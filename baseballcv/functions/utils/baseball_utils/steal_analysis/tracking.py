"""Per-frame model inference: bases, people (tracked, with pose), pitcher and ball.

The output, ``ClipTracks``, is plain data so the rest of the pipeline never touches a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import StealConfig

# COCO keypoint indices used downstream
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 0, 5, 6, 11, 12, 15, 16


@dataclass
class PersonObs:
    box: Tuple[float, float, float, float]
    kpts: np.ndarray   # (17, 2) pixels
    kconf: np.ndarray  # (17,)


@dataclass
class ClipTracks:
    fps: float
    n_frames: int
    width: int
    height: int
    shots: List[Tuple[int, int]] = field(default_factory=list)                 # inclusive frame ranges
    bases: Dict[int, list] = field(default_factory=dict)                        # frame -> [(x1,y1,x2,y2,conf)]
    people: Dict[int, Dict[int, PersonObs]] = field(default_factory=dict)       # track id -> frame -> obs
    pitcher: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    ball: Dict[int, Tuple[float, float]] = field(default_factory=dict)


def hsv_correlation(prev_hist: np.ndarray, hist: np.ndarray) -> float:
    a, b = prev_hist.ravel(), hist.ravel()
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def is_shot_cut(corr: float, mad: float, cfg: StealConfig) -> bool:
    """Hard cut by color histogram, or a cut between similar-colored views (wide shot to pitcher
    cam is ~0.75 correlation) caught by a large frame-to-frame pixel change. Pans and zooms move
    the picture far less than 25 gray levels per frame."""
    return corr < cfg.shot_cut_corr or (corr < cfg.shot_cut_soft_corr and mad > cfg.shot_cut_mad)


class ClipTracker:
    """Runs the base, person/pose, pitcher and ball models over a video."""

    def __init__(self, device: str = "cpu", config: Optional[StealConfig] = None,
                 base_model: str = "base_detector", person_model: str = "yolo11n-pose.pt",
                 pitcher_model: str = "phc_detector", ball_model: Optional[str] = "ball_trackingv4") -> None:
        self.device = device
        self.cfg = config or StealConfig()
        self._names = dict(base=base_model, person=person_model, pitcher=pitcher_model, ball=ball_model)
        self._models: Dict[str, object] = {}

    def _load(self) -> None:
        if self._models:
            return
        from ultralytics import YOLO
        from baseballcv.functions.load_tools import LoadTools
        from baseballcv.model import BaseDetector

        lt = LoadTools()
        self._models["base"] = BaseDetector(model_path=lt.load_model(self._names["base"]), device=self.device)
        self._models["person"] = YOLO(self._names["person"])
        self._models["pitcher"] = YOLO(lt.load_model(self._names["pitcher"]))
        if self._names["ball"]:
            self._models["ball"] = YOLO(lt.load_model(self._names["ball"]))

    def _best_class_box(self, model, frame, cls_name: str, conf: float):
        res = model.predict(frame, conf=conf, device=self.device, verbose=False)[0]
        best = None
        for box in res.boxes:
            if model.names[int(box.cls)].lower() == cls_name and (best is None or float(box.conf) > best[1]):
                best = (box.xyxy[0].tolist(), float(box.conf))
        return best

    def track(self, video_path: str) -> ClipTracks:
        import cv2

        self._load()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracks = ClipTracks(fps=fps, n_frames=0, width=width, height=height)

        shot_start, prev_hist, prev_gray, idx, reset = 0, None, None, 0, True
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_hist is not None and is_shot_cut(hsv_correlation(prev_hist, hist),
                                                     float(np.abs(gray - prev_gray).mean()), self.cfg):
                tracks.shots.append((shot_start, idx - 1))
                shot_start, reset = idx, True  # new shot: ByteTrack ids must not carry across a cut
            prev_hist, prev_gray = hist, gray

            dets = self._models["base"].predict(frame, conf=self.cfg.base_conf)
            if dets:
                tracks.bases[idx] = [(*d["xyxy"], d["conf"]) for d in dets]

            res = self._models["person"].track(frame, persist=not reset, classes=[0], device=self.device,
                                               verbose=False, tracker="bytetrack.yaml")[0]
            reset = False
            if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
                ids = res.boxes.id.int().tolist()
                xyxy = res.boxes.xyxy.tolist()
                kxy = res.keypoints.xy.cpu().numpy()
                kcf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                       else np.ones(kxy.shape[:2]))
                for tid, bx, kp, kc in zip(ids, xyxy, kxy, kcf):
                    tracks.people.setdefault(tid, {})[idx] = PersonObs(tuple(bx), kp, kc)

            p = self._best_class_box(self._models["pitcher"], frame, "pitcher", 0.4)
            if p:
                tracks.pitcher[idx] = tuple(p[0])
            if "ball" in self._models:
                b = self._best_class_box(self._models["ball"], frame, "baseball", 0.3)
                if b:
                    tracks.ball[idx] = ((b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2)
            idx += 1

        cap.release()
        tracks.n_frames = idx
        tracks.shots.append((shot_start, idx - 1))
        return tracks
