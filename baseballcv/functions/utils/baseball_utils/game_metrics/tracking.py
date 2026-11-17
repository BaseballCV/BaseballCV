"""Blob-level tracking on cenital clips.

Players on the reference feed are 5-10 px dark blobs on turf: no pose, no
identity. Two complementary detectors:

- MOVING blobs: per-clip median background (movers only — this deliberately
  erases stationary defenders) -> absdiff -> morph open -> connected components.
- STATIC dark blobs: morphological black-hat on a single frame (defenders at a
  frozen moment) — see ``detect_dark_blobs``.

Association is greedy nearest-neighbour with constant-velocity prediction in
PIXEL space (scipy is undeclared in pyproject — pure numpy only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baseballcv.utilities import BaseballCVLogger

from ..steal_analysis.events import smooth, sustained_onset, velocity
from .config import MetricsConfig
from .metrics import to_arrays

Det = Tuple[float, float, int]   # (cx, cy, area) in pixels


@dataclass
class BlobTracks:
    """Per-clip tracks: track_id -> {frame -> (cx, cy, area)}, frames 0..n_frames-1."""
    fps: float
    n_frames: int
    tracks: Dict[int, Dict[int, Det]]
    clip_path: Optional[str] = None
    wall0_s: Optional[float] = None    # burned-clock wall time of frame 0 (OCR, may be None)
    wall1_s: Optional[float] = None    # wall time of the last frame


# ---------------------------------------------------------------- movers
def median_background(gray_frames: Sequence[np.ndarray], stride: int = 10) -> np.ndarray:
    """Median frame over a strided subsample (a full median is unnecessary: the
    feed is quiescent for most frames)."""
    sample = np.stack(list(gray_frames)[::max(1, stride)])
    return np.median(sample, axis=0).astype(np.uint8)


def extract_blobs(gray: np.ndarray, bg: np.ndarray, cfg: MetricsConfig,
                  roi_mask: Optional[np.ndarray] = None) -> List[Det]:
    import cv2

    diff = cv2.absdiff(gray, bg)
    _, mask = cv2.threshold(diff, cfg.bg_diff_thresh, 255, cv2.THRESH_BINARY)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_open_px, cfg.morph_open_px)))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.blob_min_area_px <= area <= cfg.blob_max_area_px:
            cx, cy = centroids[i]
            out.append((float(cx), float(cy), area))
    return out


def field_roi_mask(shape, homography, cfg: MetricsConfig) -> np.ndarray:
    """Image-space polygon of the field-coordinate ROI (crowd exclusion)."""
    import cv2

    x0, y0, x1, y1 = cfg.field_roi_ft
    corners = [homography.field_to_image(p)
               for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    pts = np.array([[int(round(cx)), int(round(cy))] for cx, cy in corners],
                   dtype=np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


# ---------------------------------------------------------------- static dark blobs
def detect_dark_blobs(gray: np.ndarray, cfg: MetricsConfig,
                      search_roi=None) -> List[Det]:
    """Dark blobs on ONE frozen frame (defenders standing still).

    Morphological black-hat (closing - image) isolates dark spots smaller than
    the kernel; thresholded at ``dark_blob_contrast_min`` gray levels. Pass
    ``search_roi`` = (x0, y0, x1, y1) to restrict detection to one region.
    """
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, mask = cv2.threshold(blackhat, cfg.dark_blob_contrast_min, 255, cv2.THRESH_BINARY)
    if search_roi is not None:
        x0, y0, x1, y1 = (int(v) for v in search_roi)
        m = np.zeros_like(mask)
        m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = m
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.dark_blob_min_area_px <= area <= cfg.dark_blob_max_area_px:
            cx, cy = centroids[i]
            out.append((float(cx), float(cy), area))
    return out


# ---------------------------------------------------------------- association
def associate_blobs(per_frame_dets: Sequence[List[Det]], fps: float,
                    cfg: MetricsConfig) -> Dict[int, Dict[int, Det]]:
    """Greedy NN matching with constant-velocity prediction (pixel space).

    Longer-lived tracks are matched first each frame; a detection matches at
    most one track. Tracks shorter than ``track_min_frames`` are dropped.
    """
    tracks: Dict[int, dict] = {}
    next_id = 0
    for f, dets in enumerate(per_frame_dets):
        assigned = set()
        for tid in sorted(tracks, key=lambda t: -len(tracks[t]["frames"])):
            tr = tracks[tid]
            last_f = max(tr["frames"])
            lx, ly, _ = tr["frames"][last_f]
            vx, vy = tr["vel"]
            dt = max(1, f - last_f)
            px_, py_ = lx + vx * dt, ly + vy * dt
            # frame-to-frame the gate is tight (a runner covers <1 px/frame);
            # the wide gate only bridges occlusion gaps
            best, best_d = None, cfg.assoc_gate_px if dt > 1 else cfg.assoc_gate_dt1_px
            for j, (x, y, a) in enumerate(dets):
                if j in assigned:
                    continue
                d = float(np.hypot(x - px_, y - py_))
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                x, y, a = dets[best]
                b = cfg.vel_blend
                tr["vel"] = ((1 - b) * vx + b * (x - lx) / dt,
                             (1 - b) * vy + b * (y - ly) / dt)
                tr["frames"][f] = (x, y, a)
                assigned.add(best)
        for j, (x, y, a) in enumerate(dets):
            if j not in assigned:
                tracks[next_id] = {"frames": {f: (x, y, a)}, "vel": (0.0, 0.0)}
                next_id += 1
    return {tid: tr["frames"] for tid, tr in tracks.items()
            if len(tr["frames"]) >= cfg.track_min_frames}


# ---------------------------------------------------------------- clip front-end
def track_clip(clip_path: str, cfg: MetricsConfig, homography=None,
               clock_reader=None, logger=None) -> BlobTracks:
    """Decode one per-play clip -> tracks. ``clock_reader`` (game_sync
    BurnedClockReader) anchors the clip to wall time via 2 OCR reads; failure
    leaves wall0/wall1 None rather than guessing."""
    import cv2

    log = logger or BaseballCVLogger.get_logger("game_metrics.tracking")
    # streaming two-pass: holding every frame costs ~1.4 GB per long clip
    # (S6 spike OOM), so pass 1 samples the background + edge frames and
    # pass 2 extracts detections frame by frame
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bg_frames, first_gray, last_gray = [], None, None
    n_frames, idx = 0, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if idx == 0:
            first_gray = gray
        last_gray = gray
        if idx % max(1, cfg.bg_sample_stride) == 0:
            bg_frames.append(gray)
        idx += 1
    n_frames = idx
    cap.release()
    if not n_frames:
        return BlobTracks(fps=fps, n_frames=0, tracks={}, clip_path=clip_path)
    bg = median_background(bg_frames, stride=1)   # already strided at read time
    roi = (field_roi_mask(bg.shape, homography, cfg)
           if homography is not None else None)
    dets_pf = []
    cap = cv2.VideoCapture(clip_path)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        dets_pf.append(extract_blobs(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                     bg, cfg, roi_mask=roi))
    cap.release()
    tracks = associate_blobs(dets_pf, fps, cfg)

    wall0 = wall1 = None
    if clock_reader is not None:
        try:
            for frame_idx, gray in ((0, first_gray), (n_frames - 1, last_gray)):
                anchor = clock_reader.read_frame(gray, video_time_s=frame_idx / fps)
                if anchor is not None:
                    if frame_idx == 0:
                        wall0 = anchor.wall_s
                    else:
                        wall1 = anchor.wall_s
        except Exception as exc:
            log.warning(f"clock anchor OCR failed on {clip_path}: {exc}")

    log.info(f"{clip_path}: {n_frames} frames, {len(tracks)} tracks "
             f"(>= {cfg.track_min_frames} frames)")
    return BlobTracks(fps=fps, n_frames=n_frames, tracks=tracks,
                      clip_path=clip_path, wall0_s=wall0, wall1_s=wall1)


# ---------------------------------------------------------------- runner selection
def track_field_xy(frames: Dict[int, Det], homography) -> Dict[int, Tuple[float, float]]:
    """Track positions mapped to field feet via the calibration homography."""
    out = {}
    for f, (x, y, _a) in frames.items():
        out[f] = homography.to_field((x, y))
    return out


def select_runner_track(tracks: BlobTracks, homography, cfg: MetricsConfig
                        ) -> Tuple[Optional[int], dict]:
    """The track with a sustained toward-1B onset inside the home->1B corridor.

    S6 spike lesson: near the plate the batter merges with catcher/umpire, so
    track identities chain (pile -> coach -> runner) and a "track starts in
    the home band" rule rejects the very track that holds the run. Selection
    therefore keys on the ONSET: 6 sustained frames of >8 ft/s toward 1B,
    positioned 5-30 ft from home within |y| <= 20 ft when it starts, clearing
    the pile but nowhere near the bag.

    Returns (track_id, diagnostics with ``onset_frame``); (None, diagnostics)
    when nothing qualifies.
    """
    diag = {"candidates": []}
    best_tid, best_score = None, (-1, -1)   # sanity first, length second
    best_onset = None
    x0, y0, x1, y1 = cfg.valid_field_box_ft
    for tid, frames in tracks.tracks.items():
        if not frames:
            continue
        fs = sorted(frames)
        pts = np.array([homography.to_field(frames[f][:2]) for f in fs])
        # smoothed dense velocity: raw per-detection diffs are noise-dominated
        # near home (1 px ~ 1.4 ft there; jitter alone is +-20 ft/s)
        xs, ys = to_arrays({f: (p[0], p[1]) for f, p in zip(fs, pts)},
                           tracks.n_frames, cfg)
        sm = smooth(xs, 5)
        vx = velocity(sm, tracks.fps)          # ft/s toward 1B, NaN at gaps
        ok = np.nan_to_num(vx, nan=-1.0) > cfg.runner_min_toward_1b_ft_s
        onset = sustained_onset(ok, max(3, int(round(0.17 * tracks.fps))))
        if onset is None:
            continue
        i = min(onset, len(fs) - 1)
        d_home_i = float(np.hypot(xs[fs[i]], ys[fs[i]])) \
            if np.isfinite(xs[fs[i]]) else float("nan")
        if not np.isfinite(d_home_i) \
                or not (cfg.onset_d_home_min_ft <= d_home_i <= cfg.onset_d_home_max_ft) \
                or abs(ys[fs[i]]) > cfg.corridor_y_max_ft:
            continue
        # trajectory sanity: chained identities wander into homography-blowup
        # territory; a real runner stays in the valid box after the onset
        post = pts[i:]
        in_box = ((post[:, 0] >= x0) & (post[:, 0] <= x1)
                  & (post[:, 1] >= y0) & (post[:, 1] <= y1))
        if float(in_box.mean()) < 0.95 or in_box.sum() < 2:
            continue
        post_ok = post[in_box]
        d_1b = np.hypot(post_ok[:, 0] - 90.0, post_ok[:, 1])
        reaches_1b = float(d_1b.min()) <= cfg.arrival_margin_ft + 3.0
        score = (1 if reaches_1b else 0, len(fs))
        cand = {"track_id": tid, "onset_frame": int(fs[i]),
                "onset_ft": [float(xs[fs[i]]), float(ys[fs[i]])],
                "onset_d_home_ft": d_home_i,
                "reaches_1b": reaches_1b, "n_frames": len(fs)}
        diag["candidates"].append(cand)
        if score > best_score:
            best_tid, best_score = tid, score
            best_onset = int(fs[i])
    diag["selected"] = best_tid
    diag["onset_frame"] = best_onset
    return best_tid, diag
