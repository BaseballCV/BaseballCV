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
    """Downscaled frame-diff energy sampled at activity_fps."""
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
                    energies.append(float(np.abs(gray.astype(np.int16) - prev).mean()))
                    times.append(idx / fps)
                prev = gray
        idx += 1
    cap.release()
    logger.info(f"motion timeline: {len(times)} samples at {fps / step:.1f} fps")
    return MotionTimeline(np.array(times), np.array(energies), fps / step)


def detect_onsets(timeline: MotionTimeline, config: Optional[GameSyncConfig] = None) -> pl.DataFrame:
    """Motion bursts: energy leaving a rolling-median floor, sustained briefly.

    Same walk-back-to-floor idea as steal_analysis.events.sustained_onset.
    """
    cfg = config or GameSyncConfig()
    e = timeline.energy
    t = timeline.times_s
    if len(e) < 10:
        return pl.DataFrame({"time_s": [], "strength": []})
    k = max(3, int(round(2.0 * timeline.fps_sample)))  # ~2 s rolling window
    med = pl.Series(e).rolling_median(window_size=min(k, len(e))).to_numpy()
    above = e > np.maximum(med * cfg.onset_min_strength, med + 1.0)
    sustain = max(1, int(round(cfg.onset_sustain_s * timeline.fps_sample)))
    rows = []
    i = 1
    while i < len(e) - sustain:
        if above[i] and not above[i - 1]:
            if above[i:i + sustain].all():
                rows.append({"time_s": float(t[i]), "strength": float(e[i] / max(med[i], 1e-9))})
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
