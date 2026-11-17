"""Runner metrics from blob tracks (field feet).

Semantics you must not blur:
- "contact" is NOT contact: it is the batter-blob's first observable move,
  which lags bat-ball contact by roughly one human reaction tail (~0.2 s).
  DB pitch time is a separate, clearly-named column joined in runner.py.
- Every value is None-able and never guessed: an unobserved event stays None
  and the row carries a status sentinel instead (house convention).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..steal_analysis.events import smooth, sustained_onset, velocity
from .config import MetricsConfig

FIRST_BASE = (90.0, 0.0)
SECOND_BASE = (90.0 / np.sqrt(2.0), 90.0 / np.sqrt(2.0))
THIRD_BASE = (0.0, 90.0)
HOME = (0.0, 0.0)
BASES = {0: HOME, 1: FIRST_BASE, 2: SECOND_BASE, 3: THIRD_BASE}


def to_arrays(field_xy: Dict[int, Tuple[float, float]], n_frames: int,
              cfg: Optional[MetricsConfig] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Dict[frame -> (x, y)] -> (xs, ys) dense arrays, NaN on untracked frames.

    Points outside ``cfg.valid_field_box_ft`` are projection artifacts (the
    fisheye homography blows up near the horizon) and become NaN."""
    xs = np.full(n_frames, np.nan)
    ys = np.full(n_frames, np.nan)
    box = cfg.valid_field_box_ft if cfg is not None else (-30.0, -30.0, 130.0, 130.0)
    for f, (x, y) in field_xy.items():
        if 0 <= f < n_frames and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            xs[f], ys[f] = float(x), float(y)
    return xs, ys


def first_move_frame(xs: np.ndarray, ys: np.ndarray, fps: float, cfg: MetricsConfig,
                     start: int = 0) -> Optional[int]:
    """First frame of sustained radial motion away from home (the batter's bolt).

    Walked back to where speed leaves 25% of the threshold, mirroring
    steal_analysis.events.runner_first_move, on the radial-distance trajectory.
    """
    d = np.hypot(xs, ys)
    sm = smooth(d, max(3, int(round(0.15 * fps))))
    speed = velocity(sm, fps)
    ok = np.nan_to_num(speed, nan=-1.0) > cfg.first_move_speed_ft_s
    onset = sustained_onset(ok, max(1, int(round(0.1 * fps))), start)
    if onset is None:
        return None
    floor = 0.25 * cfg.first_move_speed_ft_s
    while onset > start and not np.isnan(speed[onset - 1]) and speed[onset - 1] > floor:
        onset -= 1
    return onset


def frame_speeds(xs: np.ndarray, ys: np.ndarray, fps: float,
                 cfg: Optional[MetricsConfig] = None) -> np.ndarray:
    """NaN-aware per-frame speed in ft/s (central difference on smoothed x/y).

    Frames faster than ``cfg.max_plausible_speed_ft_s`` are identity-swap
    artifacts of chained tracks and become NaN, not data."""
    win = 5
    sx = smooth(xs, win)
    sy = smooth(ys, win)
    vx = velocity(sx, fps)
    vy = velocity(sy, fps)
    spd = np.hypot(vx, vy)
    cap = cfg.max_plausible_speed_ft_s if cfg is not None else 40.0
    spd[spd > cap] = np.nan
    return spd


def _rolling_mean(v: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware centered rolling mean."""
    out = np.full_like(v, np.nan, dtype=float)
    half = win // 2
    for i in range(len(v)):
        seg = v[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def runner_play_metrics(field_xy: Dict[int, Tuple[float, float]], fps: float,
                        n_frames: int, cfg: MetricsConfig,
                        contact_frame: Optional[int] = None) -> dict:
    """Flat dict row for one batter-runner advance (home -> 1B).

    All values None when unobserved; ``status`` says why. ``time_to_1b_s`` runs
    first move -> first frame within ``arrival_margin_ft`` of first base.
    """
    row = {
        "status": "no_track", "n_frames_tracked": len(field_xy),
        "contact_batter_move_frame": None, "contact_batter_move_s": None,
        "bolt_first_move_s": None, "arrival_frame": None,
        "time_to_1b_s": None, "peak_speed_ft_s": None,
        "speed_at_15_ft": None, "speed_at_30_ft": None, "speed_at_45_ft": None,
        "speed_at_60_ft": None, "speed_at_75_ft": None,
        "decel_at_bag_ft_s2": None, "route_efficiency": None,
        "start_d_home_ft": None, "min_d_1b_ft": None,
    }
    if not field_xy:
        return row
    xs, ys = to_arrays(field_xy, n_frames, cfg)
    fs = sorted(field_xy)
    d_home = np.hypot(xs, ys)
    d_1b = np.hypot(xs - FIRST_BASE[0], ys - FIRST_BASE[1])
    row["start_d_home_ft"] = float(d_home[fs[0]])
    row["min_d_1b_ft"] = float(np.nanmin(d_1b[fs[0]:]))

    if contact_frame is None:
        contact_frame = first_move_frame(xs, ys, fps, cfg)
    row["contact_batter_move_frame"] = int(contact_frame) if contact_frame is not None else None
    row["contact_batter_move_s"] = None if contact_frame is None else round(contact_frame / fps, 3)
    if contact_frame is None:
        row["status"] = "no_first_move"
        return row

    speed = frame_speeds(xs, ys, fps, cfg)
    valid = speed[fs[0]:fs[-1] + 1]
    fin = valid[~np.isnan(valid)]
    if fin.size:
        win = max(2, int(round(cfg.speed_window_s * fps)))
        row["peak_speed_ft_s"] = round(float(np.nanmax(_rolling_mean(valid, win))), 2)

    # arrival at 1B: first frame after contact within the margin
    arrival = None
    for f in range(contact_frame, fs[-1] + 1):
        if not np.isnan(d_1b[f]) and d_1b[f] <= cfg.arrival_margin_ft:
            arrival = f
            break
    if arrival is not None:
        row["arrival_frame"] = int(arrival)
        row["time_to_1b_s"] = round((arrival - contact_frame) / fps, 3)

    # speed checkpoints: speed at the first frame whose distance from home
    # passes each checkpoint (sprint-projection style, home->1B only). A
    # checkpoint already behind the runner at contact stays None.
    d_at_contact = d_home[contact_frame] if not np.isnan(d_home[contact_frame]) else np.inf
    for checkpoint in (15, 30, 45, 60, 75):
        if d_at_contact >= checkpoint:
            continue
        for f in range(contact_frame, fs[-1] + 1):
            if not np.isnan(d_home[f]) and d_home[f] >= checkpoint:
                v = speed[f]
                row[f"speed_at_{checkpoint}_ft"] = None if np.isnan(v) else round(float(v), 2)
                break

    # deceleration into the bag: (speed 0.5 s before arrival - speed at arrival)/0.5 s
    if arrival is not None:
        back = max(contact_frame, arrival - int(round(0.5 * fps)))
        v_arr, v_pre = speed[arrival], speed[back]
        if not (np.isnan(v_arr) or np.isnan(v_pre)):
            dt = (arrival - back) / fps
            row["decel_at_bag_ft_s2"] = round(float((v_pre - v_arr) / dt), 2)

    # route efficiency: straight-line 90 ft / path length actually run
    seg_path, prev = 0.0, None
    for f in range(contact_frame, (arrival if arrival is not None else fs[-1]) + 1):
        if np.isnan(xs[f]):
            continue
        if prev is not None:
            seg_path += float(np.hypot(xs[f] - prev[0], ys[f] - prev[1]))
        prev = (xs[f], ys[f])
    if seg_path > 0:
        row["route_efficiency"] = round(90.0 / seg_path, 3)

    # status
    problems = []
    if arrival is None:
        problems.append("no_arrival")
    if row["peak_speed_ft_s"] is not None and not (
            cfg.peak_speed_bounds[0] <= row["peak_speed_ft_s"] <= cfg.peak_speed_bounds[1]):
        problems.append("peak_speed_out_of_range")
    row["status"] = "ok" if not problems else ";".join(problems)
    return row


def baserunning_legs(field_xy: Dict[int, Tuple[float, float]], fps: float,
                     cfg: MetricsConfig) -> List[dict]:
    """Per-segment speed for multi-base advances (double/triple/HR, steals).

    A "bag arrival" is the first frame within ``arrival_margin_ft`` of each
    base, in chronological order. The FIRST leg runs from the track's start
    (home plate — this function is for batter-runner advances) to the first
    arrival; later legs run between consecutive arrivals. Speed per leg is the
    MEDIAN frame speed inside it (robust to tag-pile jitter).
    """
    if not field_xy:
        return []
    xs, ys = to_arrays(field_xy, max(field_xy) + 1, cfg)
    fs = sorted(field_xy)
    speed = frame_speeds(xs, ys, fps, cfg)
    order = [1, 2, 3, 0]   # 1B, 2B, 3B, home (HR scoring)
    arrivals: List[Tuple[int, int]] = []   # (base, frame)
    last = None
    for base in order:
        bx, by = BASES[base]
        d = np.hypot(xs - bx, ys - by)
        for f in range((last + 1) if last is not None else fs[0], fs[-1] + 1):
            if not np.isnan(d[f]) and d[f] <= cfg.arrival_margin_ft:
                arrivals.append((base, f))
                last = f
                break
    legs = []
    bounds = [(0, fs[0], *arrivals[0])] if arrivals else []
    bounds += [(b_from, f0, b_to, f1)
               for (b_from, f0), (b_to, f1) in zip(arrivals, arrivals[1:])]
    for b_from, f0, b_to, f1 in bounds:
        seg = speed[f0:f1 + 1]
        seg = seg[~np.isnan(seg)]
        legs.append({
            "from_base": b_from, "to_base": b_to,
            "start_frame": int(f0), "end_frame": int(f1),
            "duration_s": round((f1 - f0) / fps, 3),
            "median_speed_ft_s": round(float(np.median(seg)), 2) if seg.size else None,
        })
    return legs
