"""Cheap whole-video motion timeline for sync validation and warm-up flagging.

Sampled at ~5 fps on downscaled grayscale frame-diffs (spike-verified:
pitch onsets only need ~0.2-0.5 s precision, and a 394k-frame pass at 30 fps
would cost ~40 min for no gain).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import polars as pl

from baseballcv.utilities import BaseballCVLogger

from .config import GameSyncConfig


@dataclass
class MotionTimeline:
    times_s: np.ndarray
    energy: np.ndarray
    fps_sample: float

    def energy_at(self, t: float) -> float:
        return float(np.interp(t, self.times_s, self.energy))

    def slice(self, t0: float, t1: float) -> "MotionTimeline":
        m = (self.times_s >= t0) & (self.times_s <= t1)
        return MotionTimeline(self.times_s[m], self.energy[m], self.fps_sample)


def motion_timeline(video_path: str, config: Optional[GameSyncConfig] = None,
                    logger=None) -> MotionTimeline:
    """Fraction-of-changing-pixels signal sampled at activity_fps.

    Measured on the Magallanes feed: the MEAN abs frame-diff is dominated by
    compression flicker (median ~0.8 gray levels, pitch bursts peak ~3) and
    cannot be thresholded; the FRACTION of ROI pixels changing more than
    ``activity_diff_thresh`` gray levels sits near 0.03 at rest and spikes
    5x at gameplay events, which thresholds cleanly.
    """
    cfg = config or GameSyncConfig()
    logger = logger or BaseballCVLogger.get_logger("motion_timeline")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps / cfg.activity_fps)))
    x0f, y0f, x1f, y1f = cfg.activity_roi

    times, energies = [], []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                roi = frame[int(h * y0f):int(h * y1f), int(w * x0f):int(w * x1f)]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=cfg.activity_downscale, fy=cfg.activity_downscale)
                if prev is not None:
                    diff = np.abs(gray.astype(np.int16) - prev)
                    energies.append(float((diff > cfg.activity_diff_thresh).mean()))
                    times.append(idx / fps)
                prev = gray
        idx += 1
    cap.release()
    logger.info(f"motion timeline: {len(times)} samples at {fps / step:.1f} fps")
    return MotionTimeline(np.array(times), np.array(energies), fps / step)


def detect_onsets(timeline: MotionTimeline, config: Optional[GameSyncConfig] = None) -> pl.DataFrame:
    """Motion bursts over a long rolling-median floor with MAD noise scaling.

    The floor window is deliberately long (~20 s): pitch bursts must NOT lift
    their own threshold (the original 2 s window did exactly that and detected
    nothing on real footage). Threshold = floor + max(MAD*mult, floor_add).
    Same walk-back-to-floor idea as steal_analysis.events.sustained_onset.
    """
    cfg = config or GameSyncConfig()
    e = timeline.energy
    t = timeline.times_s
    if len(e) < 10:
        return pl.DataFrame({"time_s": [], "strength": []})
    k = max(3, int(round(cfg.onset_floor_window_s * timeline.fps_sample)))
    med = pl.Series(e).rolling_median(window_size=min(k, len(e))).to_numpy()
    # leading window positions come back as nulls: use nan-aware stats and let
    # the NaN thresholds simply compare False
    resid = e - med
    finite = resid[np.isfinite(resid)]
    mad = float(np.median(np.abs(finite - np.median(finite)))) * 1.4826 if len(finite) else 0.0
    thr = med + np.maximum(mad * cfg.onset_noise_mult, cfg.onset_floor_add)
    above = e > thr
    sustain = max(1, int(round(cfg.onset_sustain_s * timeline.fps_sample)))
    rows = []
    i = 1
    while i < len(e) - sustain:
        if above[i] and not above[i - 1]:
            if above[i:i + sustain].all():
                rows.append({"time_s": float(t[i]),
                             "strength": float((e[i] - med[i]) / max(thr[i] - med[i], 1e-9))})
                i += sustain
        i += 1
    return pl.DataFrame(rows)


def activity_match_score(times_db: np.ndarray, onsets: pl.DataFrame,
                         tol_s: float) -> float:
    """Fraction of DB pitch times with an onset within tol_s (video time)."""
    if onsets.is_empty() or len(times_db) == 0:
        return 0.0
    ot = onsets["time_s"].to_numpy()
    hits = 0
    for t in times_db:
        if len(ot) and np.min(np.abs(ot - t)) <= tol_s:
            hits += 1
    return hits / len(times_db)
