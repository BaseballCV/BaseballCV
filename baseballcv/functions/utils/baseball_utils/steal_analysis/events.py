"""Event detection on per-frame trajectories. Pure numpy: no models, fully unit-testable.

Every function returns ``None`` for an event that is not observable. Events are never guessed.
All frame indices are relative to the array that was passed in.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import StealConfig


def interpolate_gaps(x: Sequence[float], max_gap: int) -> np.ndarray:
    """Linearly fill NaN gaps no longer than ``max_gap`` frames. Edges and long gaps stay NaN."""
    a = np.asarray(x, dtype=float).copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() < 2:
        return a
    filled = np.interp(idx, idx[good], a[good])
    out = a.copy()
    i = 0
    n = len(a)
    first, last = idx[good][0], idx[good][-1]
    while i < n:
        if np.isnan(a[i]):
            j = i
            while j < n and np.isnan(a[j]):
                j += 1
            if i > first and j <= last and (j - i) <= max_gap:
                out[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return out


def smooth(x: Sequence[float], window: int) -> np.ndarray:
    """NaN-aware centered moving average (window forced odd, >= 1)."""
    a = np.asarray(x, dtype=float)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return a.copy()
    half = window // 2
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg = a[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) and not np.isnan(a[i]):
            out[i] = seg.mean()
    return out


def velocity(x: Sequence[float], fps: float) -> np.ndarray:
    """Central-difference derivative in units/second. NaN where either neighbor is NaN."""
    a = np.asarray(x, dtype=float)
    v = np.full_like(a, np.nan)
    if len(a) < 3:
        return v
    v[1:-1] = (a[2:] - a[:-2]) * fps / 2.0
    v[0] = (a[1] - a[0]) * fps
    v[-1] = (a[-1] - a[-2]) * fps
    return v


def sustained_onset(ok: np.ndarray, sustain: int, start: int = 0) -> Optional[int]:
    """First index >= start where ``ok`` is True for ``sustain`` consecutive frames."""
    sustain = max(1, sustain)
    n = len(ok)
    run = 0
    for i in range(max(0, start), n):
        run = run + 1 if ok[i] else 0
        if run >= sustain:
            return i - sustain + 1
    return None


def runner_first_move(along_ft: Sequence[float], fps: float, cfg: StealConfig,
                      start: int = 0) -> Optional[int]:
    """Frame where the runner's speed toward 2B first exceeds the threshold and stays there.

    The onset is walked back to where speed first left the noise floor (25% of threshold), so the
    returned frame is the start of the movement, not the moment it crossed the threshold.
    """
    sm = smooth(along_ft, int(round(cfg.smooth_window_s * fps)))
    speed = velocity(sm, fps)
    ok = np.nan_to_num(speed, nan=-1.0) > cfg.move_speed_ft_s
    onset = sustained_onset(ok, int(round(cfg.move_sustain_s * fps)), start)
    if onset is None:
        return None
    floor = 0.25 * cfg.move_speed_ft_s
    while onset > max(0, start) and not np.isnan(speed[onset - 1]) and speed[onset - 1] > floor:
        onset -= 1
    return int(onset)


def pitcher_first_move(energy: Sequence[float], fps: float, cfg: StealConfig) -> Optional[int]:
    """Frame where the pitcher's motion energy leaves its set-position baseline.

    ``energy`` is mean keypoint displacement per frame in torso-lengths. The baseline is taken from
    the first ``pitcher_baseline_s`` seconds with data.
    """
    e = np.asarray(energy, dtype=float)
    finite = np.where(~np.isnan(e))[0]
    if len(finite) < 5:
        return None
    base_n = max(3, int(round(cfg.pitcher_baseline_s * fps)))
    base = e[finite[:base_n]]
    thresh = base.mean() + max(cfg.pitcher_move_k * base.std(), cfg.pitcher_move_floor)
    ok = np.nan_to_num(e, nan=-1.0) > thresh
    ok[finite[:base_n]] = False  # the baseline window is, by definition, not the move
    return sustained_onset(ok, int(round(cfg.pitcher_sustain_s * fps)), int(finite[0]))


def release_frame(ball_frames: Sequence[int], min_run: int = 3, after: int = 0) -> Optional[int]:
    """First frame of the first run of >= ``min_run`` consecutive ball detections after ``after``."""
    frames = sorted({int(f) for f in ball_frames if f >= after})
    run_start, run = None, 0
    prev = None
    for f in frames:
        if prev is not None and f == prev + 1:
            run += 1
        else:
            run_start, run = f, 1
        if run >= min_run:
            return run_start
        prev = f
    return None


def arrival_frame(along_ft: Sequence[float], target_ft: float, margin_ft: float,
                  start: int = 0) -> Optional[int]:
    """First frame >= start where the runner is within ``margin_ft`` of (or past) the target."""
    a = np.asarray(along_ft, dtype=float)
    for i in range(max(0, start), len(a)):
        if not np.isnan(a[i]) and a[i] >= target_ft - margin_ft:
            return i
    return None


def slide_start_frame(torso_angle_deg: Sequence[float], fps: float, start: int = 0,
                      threshold_deg: float = 60.0, sustain_s: float = 0.1) -> Optional[int]:
    """First frame after ``start`` where the torso is > threshold degrees from vertical, sustained."""
    ang = np.asarray(torso_angle_deg, dtype=float)
    ok = np.nan_to_num(ang, nan=0.0) > threshold_deg
    return sustained_onset(ok, int(round(sustain_s * fps)), start)
