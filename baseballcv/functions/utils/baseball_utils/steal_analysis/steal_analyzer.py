"""StealAnalyzer: video in, per-play steal metrics, categories, CSV and annotated video out."""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from baseballcv.utilities import BaseballCVLogger

from .categorizer import aggregate_runners, categorize_play
from .config import StealConfig
from .events import (arrival_frame, interpolate_gaps, pitcher_first_move, release_frame,
                     runner_first_move, slide_start_frame)
from .geometry import (BaselineAxis, Homography, ScaleEstimate, best_scale, box_center,
                       scale_from_base_box, scale_from_height, scale_from_two_bases, stable_boxes)
from .metrics import Events, classify_first_step, compute_play_metrics
from .tracking import (L_ANKLE, L_HIP, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_SHOULDER,
                       ClipTracker, ClipTracks, PersonObs)

KP_CONF = 0.3
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def _feet(obs: PersonObs) -> Tuple[float, float]:
    """Feet midpoint from confident ankles, else bottom-center of the person box."""
    pts = [obs.kpts[i] for i in (L_ANKLE, R_ANKLE) if obs.kconf[i] > KP_CONF]
    if pts:
        p = np.mean(pts, axis=0)
        return float(p[0]), float(p[1])
    return (obs.box[0] + obs.box[2]) / 2.0, float(obs.box[3])


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StealAnalyzer:
    """Measures how a runner leaves first base on a steal of second.

    ``analyze_video`` runs the models (``ClipTracker``) and then ``analyze_tracks``, which is pure
    and can be run on any ``ClipTracks``. Every play returns a flat row; fields that could not be
    observed are ``None``, never estimated, and ``status`` says why a play was not scored.
    """

    def __init__(self, device: str = "cpu", config: Optional[StealConfig] = None,
                 results_dir: str = "results", tracker: Optional[ClipTracker] = None,
                 logger=None) -> None:
        self.cfg = config or StealConfig()
        self.results_dir = results_dir
        self.logger = logger or BaseballCVLogger.get_logger(self.__class__.__name__)
        self.tracker = tracker or ClipTracker(device=device, config=self.cfg)

    # ------------------------------------------------------------------ analysis (pure)
    def analyze_tracks(self, tracks: ClipTracks, runner_height_in: Optional[float] = None,
                       first_base_xy: Optional[Tuple[float, float]] = None,
                       homography: Optional[Homography] = None) -> Dict:
        cfg, fps = self.cfg, (tracks.fps or 30.0)
        row: Dict = {"status": "ok", "fps": fps, "n_shots": len(tracks.shots)}

        # 1. analysis shot: the shot with the most base detections
        scored = [(sum(1 for f in range(s, e + 1) if f in tracks.bases), s, e) for s, e in tracks.shots
                  if e - s + 1 >= cfg.min_shot_frames]
        if not scored or max(scored)[0] == 0:
            return {**row, "status": "no_bases_detected"}
        _, s, e = max(scored)
        frames = list(range(s, e + 1))
        row.update(shot_start_frame=s, shot_end_frame=e)

        # 2. static bases within the shot
        boxes = stable_boxes({f: tracks.bases[f] for f in frames if f in tracks.bases},
                             cfg.min_base_frames, cfg.base_cluster_frac)
        if not boxes:
            return {**row, "status": "no_stable_base"}

        # 3. base assignment candidates
        if first_base_xy is not None:
            b1 = min(boxes, key=lambda b: np.hypot(*np.subtract(box_center(b), first_base_xy)))
            rest = [b for b in boxes if b is not b1]
            assignments = [(b1, rest[0] if rest else None)]
        elif len(boxes) == 1:
            assignments = [(boxes[0], None)]
        else:
            assignments = [(boxes[0], boxes[1]), (boxes[1], boxes[0])]

        best = None
        for b1, b2 in assignments:
            ev = self._evaluate(tracks, frames, b1, b2, homography)
            if ev is not None and (best is None or ev["travel"] > best["travel"]):
                best = ev
        if best is None or best["travel"] < cfg.min_steal_travel_ft:
            return {**row, "status": "no_steal_motion"}

        b1, b2, tid = best["b1"], best["b2"], best["tid"]
        obs_map = tracks.people[tid]

        # 4. final scale and axis (re-solved now that the runner is known)
        estimates: List[ScaleEstimate] = []
        conf = cfg.scale_confidence
        if homography is not None:
            estimates.append(ScaleEstimate(float("nan"), "homography", conf["homography"]))
        if b2 is not None:
            estimates.append(scale_from_two_bases(box_center(b1), box_center(b2), cfg.base_spacing_ft,
                                                  conf["two_bases"]))
        estimates.append(scale_from_base_box(b1, cfg.base_size_in, conf["single_base"]))
        if runner_height_in:
            h = self._runner_height_px(obs_map, frames)
            if h:
                estimates.append(scale_from_height(h, runner_height_in, conf["runner_height"]))
        finite = [x for x in estimates if not np.isnan(x.px_per_ft)]
        primary = best_scale(estimates)[0]
        scalar, spread = best_scale(finite)
        axis = self._make_axis(b1, b2, scalar.px_per_ft, homography, best["direction"])

        # 5. runner trajectory over the shot
        n = len(frames)
        along = np.full(n, np.nan)
        left, right = np.full(n, np.nan), np.full(n, np.nan)
        torso = np.full(n, np.nan)
        for i, f in enumerate(frames):
            o = obs_map.get(f)
            if o is None:
                continue
            along[i] = axis.project(_feet(o))
            if o.kconf[L_ANKLE] > KP_CONF:
                left[i] = axis.project(tuple(o.kpts[L_ANKLE]))
            if o.kconf[R_ANKLE] > KP_CONF:
                right[i] = axis.project(tuple(o.kpts[R_ANKLE]))
            if all(o.kconf[k] > KP_CONF for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
                sh = (o.kpts[L_SHOULDER] + o.kpts[R_SHOULDER]) / 2
                hp = (o.kpts[L_HIP] + o.kpts[R_HIP]) / 2
                torso[i] = np.degrees(np.arctan2(abs(sh[0] - hp[0]), max(abs(sh[1] - hp[1]), 1e-6)))
        gap = int(round(0.3 * fps))
        along, left, right = (interpolate_gaps(a, gap) for a in (along, left, right))

        # 6. events
        energy = self._pitcher_energy(tracks, frames)
        pfm = pitcher_first_move(energy, fps, cfg)
        rfm = runner_first_move(along, fps, cfg)
        rel = release_frame([f - s for f in tracks.ball if s <= f <= e], after=pfm or 0)
        second_ft = axis.project(box_center(b2)) if b2 is not None else cfg.base_spacing_ft
        events = Events(
            pitcher_first_move=pfm, runner_first_move=rfm, release=rel,
            slide_start=slide_start_frame(torso, fps, rfm) if rfm is not None else None,
            arrival=arrival_frame(along, second_ft, cfg.arrival_margin_ft, rfm or 0) if rfm is not None else None,
        )
        step = classify_first_step(left, right, rfm, fps, cfg.first_step_window_s) if rfm is not None else None

        # 7. metrics, quality, categories
        row.update(compute_play_metrics(along, fps, events, second_ft, cfg, first_step=step))
        row.update(
            runner_track_id=tid,
            ambiguous_runner=bool(best["ambiguous"]),
            n_runner_frames=int(np.sum(~np.isnan(along))),
            scale_method=primary.method,
            scale_confidence=primary.confidence,
            scale_px_per_ft=None if np.isnan(primary.px_per_ft) else round(primary.px_per_ft, 2),
            scale_spread_pct=None if spread is None else round(spread, 1),
            axis_method=axis.method,
            events_observed=",".join(k for k, v in vars(events).items() if v is not None),
            **{f"frame_{k}": (None if v is None else v + s) for k, v in vars(events).items()},
        )
        row.update(categorize_play(row, cfg))
        row["_runner_obs"] = obs_map
        row["_bases"] = [b1, b2]
        return row

    def _make_axis(self, b1, b2, px_per_ft: float, homography, direction) -> BaselineAxis:
        c1 = box_center(b1)
        if b2 is not None:
            return BaselineAxis(c1, second_base=box_center(b2), px_per_ft=px_per_ft,
                                homography=homography, method="two_bases")
        return BaselineAxis(c1, direction=direction, px_per_ft=px_per_ft,
                            homography=homography, method="runner_direction")

    def _evaluate(self, tracks: ClipTracks, frames: List[int], b1, b2, homography) -> Optional[Dict]:
        """Score one base assignment: which person is the runner and how far did they travel."""
        cfg = self.cfg
        prelim = best_scale(
            ([scale_from_two_bases(box_center(b1), box_center(b2), cfg.base_spacing_ft, 0.6)] if b2 else [])
            + [scale_from_base_box(b1, cfg.base_size_in, 0.3)])[0].px_per_ft
        c1 = np.array(box_center(b1))
        fset = set(frames)
        cands = []
        for tid, obs in tracks.people.items():
            fs = sorted(f for f in obs if f in fset)
            if len(fs) < 10:
                continue
            pos = np.array([_feet(obs[f]) for f in fs])
            start = np.median(pos[:5], axis=0)
            if np.hypot(*(start - c1)) > cfg.candidate_radius_ft * prelim:
                continue
            if b2 is not None:
                unit = np.array(box_center(b2)) - c1
                unit /= np.linalg.norm(unit)
                proj = (pos - c1) @ unit / prelim
                travel, direction = float(proj.max() - proj[:5].mean()), unit
            else:
                disp = pos - start
                far = disp[np.argmax(np.hypot(disp[:, 0], disp[:, 1]))]
                norm = np.linalg.norm(far)
                if norm == 0:
                    continue
                travel, direction = float(norm / prelim), far / norm
            cands.append((travel, tid, direction))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        travel, tid, direction = cands[0]
        ambiguous = len(cands) > 1 and cands[1][0] > cfg.ambiguous_ratio * travel
        return {"travel": travel, "tid": tid, "direction": tuple(direction), "b1": b1, "b2": b2,
                "ambiguous": ambiguous}

    @staticmethod
    def _runner_height_px(obs_map: Dict[int, PersonObs], frames: List[int]) -> Optional[float]:
        """Median nose-to-ankle pixel height over early, upright frames."""
        hs = []
        for f in frames[:max(10, len(frames) // 4)]:
            o = obs_map.get(f)
            if o is None or o.kconf[NOSE] < KP_CONF or o.kconf[L_ANKLE] < KP_CONF or o.kconf[R_ANKLE] < KP_CONF:
                continue
            ankle_y = (o.kpts[L_ANKLE][1] + o.kpts[R_ANKLE][1]) / 2
            hs.append(ankle_y - o.kpts[NOSE][1])
        return float(np.median(hs)) if len(hs) >= 3 else None

    def _pitcher_energy(self, tracks: ClipTracks, frames: List[int]) -> np.ndarray:
        """Per-frame pitcher motion energy (keypoint displacement in torso-lengths), NaN if unseen."""
        energy = np.full(len(frames), np.nan)
        prev = None
        for i, f in enumerate(frames):
            pbox = tracks.pitcher.get(f)
            cur = None
            if pbox is not None:
                best, best_iou = None, 0.3
                for obs_map in tracks.people.values():
                    o = obs_map.get(f)
                    if o is not None and _iou(o.box, pbox) > best_iou:
                        best, best_iou = o, _iou(o.box, pbox)
                cur = best
            if cur is not None and prev is not None:
                torso = np.linalg.norm((cur.kpts[L_SHOULDER] + cur.kpts[R_SHOULDER]) / 2
                                       - (cur.kpts[L_HIP] + cur.kpts[R_HIP]) / 2)
                if torso >= 10:
                    ok = (cur.kconf > KP_CONF) & (prev.kconf > KP_CONF)
                    ok[:5] = False  # ignore face keypoints
                    if ok.sum() >= 4:
                        d = np.linalg.norm(cur.kpts[ok] - prev.kpts[ok], axis=1)
                        energy[i] = float(d.mean() / torso)
            prev = cur
        return energy

    # ------------------------------------------------------------------ video I/O
    def analyze_video(self, video_path: str, runner_height_in: Optional[float] = None,
                      first_base_xy: Optional[Tuple[float, float]] = None,
                      homography: Optional[Homography] = None, create_video: bool = True,
                      output_dir: Optional[str] = None) -> Dict:
        out_dir = output_dir or self.results_dir
        os.makedirs(out_dir, exist_ok=True)
        tracks = self.tracker.track(video_path)
        row = self.analyze_tracks(tracks, runner_height_in, first_base_xy, homography)
        row["video"] = os.path.basename(video_path)
        if create_video and row["status"] == "ok":
            stem = os.path.splitext(os.path.basename(video_path))[0]
            row["annotated_video"] = self._render(video_path, os.path.join(out_dir, f"steal_{stem}.mp4"), row)
        row.pop("_runner_obs", None)
        row.pop("_bases", None)
        return row

    def analyze_folder(self, input_folder: str, csv_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        files = sorted(f for ext in VIDEO_EXTS for f in glob.glob(os.path.join(input_folder, f"*{ext}")))
        rows = []
        for path in files:
            try:
                rows.append(self.analyze_video(path, **kwargs))
            except Exception as exc:  # one bad clip must not sink the batch
                self.logger.error(f"{os.path.basename(path)} failed: {exc}")
                rows.append({"video": os.path.basename(path), "status": f"error: {exc}"})
        df = pd.DataFrame(rows)
        csv_path = csv_path or os.path.join(self.results_dir, "steal_metrics.csv")
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Steal metrics for {len(df)} clips saved to {csv_path}")
        return df

    aggregate = staticmethod(aggregate_runners)

    def _render(self, video_path: str, out_path: str, row: Dict) -> str:
        import cv2

        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), row["fps"], (w, h))
        obs_map, bases = row["_runner_obs"], row["_bases"]
        marks = {row.get(f"frame_{k}"): k.replace("_", " ") for k in
                 ("pitcher_first_move", "runner_first_move", "release", "slide_start", "arrival")
                 if row.get(f"frame_{k}") is not None}
        idx, caption = 0, ""
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if row["shot_start_frame"] <= idx <= row["shot_end_frame"]:
                for b in bases:
                    if b is not None:
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 90, 0), 2)
                o = obs_map.get(idx)
                if o is not None:
                    bx = [int(v) for v in o.box]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 200, 0), 2)
                if idx in marks:
                    caption = marks[idx]
                if caption:
                    cv2.putText(frame, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            writer.write(frame)
            idx += 1
        cap.release()
        writer.release()
        return out_path
