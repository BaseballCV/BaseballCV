"""Per-play metric computation from a runner's baseline trajectory and detected events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .config import StealConfig
from .events import smooth, velocity


@dataclass
class Events:
    """Frame indices (into the analysis arrays) of detected events; ``None`` = not observed."""

    pitcher_first_move: Optional[int] = None
    runner_first_move: Optional[int] = None
    release: Optional[int] = None
    slide_start: Optional[int] = None
    arrival: Optional[int] = None


def _r(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None or np.isnan(x) else round(float(x), nd)


def _nanmedian(a: np.ndarray) -> Optional[float]:
    a = a[~np.isnan(a)]
    return float(np.median(a)) if len(a) else None


def classify_first_step(left_along: Sequence[float], right_along: Sequence[float],
                        start: int, fps: float, window_s: float) -> Optional[str]:
    """Heuristic first-step type from the two ankles' positions along the baseline.

    ``crossover``: the back foot (lower along-baseline position at the move) passes the front foot
    inside the window before the front foot has advanced more than 1 ft.
    ``open_step``: the front foot advances more than 1 ft first, without the back foot passing it.
    ``None``: ankles not visible or no clear first step in the window.
    """
    left = np.asarray(left_along, dtype=float)
    right = np.asarray(right_along, dtype=float)
    w = max(2, int(round(window_s * fps)))
    l, r = left[start:start + w + 1], right[start:start + w + 1]
    if len(l) < 3 or np.isnan(l).all() or np.isnan(r).all():
        return None
    l0, r0 = _nanmedian(l[:3]), _nanmedian(r[:3])
    if l0 is None or r0 is None:
        return None
    back, front = (l, r) if l0 <= r0 else (r, l)
    front0 = r0 if l0 <= r0 else l0
    for b, f in zip(back, front):
        if np.isnan(b) or np.isnan(f):
            continue
        if b > f + 0.1:
            return "crossover"
        if f - front0 > 1.0:
            return "open_step"
    return None


def compute_play_metrics(along_ft: Sequence[float], fps: float, events: Events,
                         second_base_ft: Optional[float], cfg: StealConfig,
                         first_step: Optional[str] = None) -> Dict[str, Optional[float]]:
    """All numeric metrics for one steal attempt. Missing inputs yield ``None`` columns, not guesses.

    ``along_ft`` is the runner's feet-midpoint position along the 1B->2B baseline in feet from first
    base (center of bag to feet), one value per frame.
    """
    along = np.asarray(along_ft, dtype=float)
    sm = smooth(along, int(round(cfg.smooth_window_s * fps)))
    speed = velocity(sm, fps)
    rfm, pfm = events.runner_first_move, events.pitcher_first_move
    m: Dict[str, Optional[float]] = {}

    # --- lead ---
    anchor = pfm if pfm is not None else rfm
    primary = None
    if anchor is not None:
        primary = _nanmedian(sm[max(0, anchor - 4): anchor + 1])
    m["primary_lead_ft"] = _r(primary)
    secondary = None
    if events.release is not None:
        r = events.release
        secondary = _nanmedian(sm[max(0, r - 1): r + 2])
    m["secondary_lead_ft"] = _r(secondary)
    m["lead_gain_ft"] = _r(secondary - primary) if secondary is not None and primary is not None else None
    if rfm is not None and rfm > 0 and not np.isnan(sm[:rfm + 1]).all():
        m["max_lead_before_move_ft"] = _r(np.nanmax(sm[:rfm + 1]))
    else:
        m["max_lead_before_move_ft"] = None

    # --- lead style ---
    if anchor is not None:
        window = speed[max(0, anchor - int(round(1.0 * fps))): anchor + 1]
        window = window[~np.isnan(window)]
        m["pre_pitch_speed_ft_s"] = _r(float(np.mean(np.abs(window)))) if len(window) else None
    else:
        m["pre_pitch_speed_ft_s"] = None

    # --- jump ---
    if rfm is not None and pfm is not None:
        m["reaction_time_s"] = _r((rfm - pfm) / fps)
    else:
        m["reaction_time_s"] = None
    m["reaction_resolution_s"] = _r(1.0 / fps, 4)

    # --- run ---
    m["time_to_10ft_s"] = m["peak_speed_ft_s"] = m["accel_0_5s_ft_s2"] = None
    m["move_to_arrival_s"] = m["slide_start_dist_ft"] = None
    if rfm is not None and not np.isnan(sm[rfm]):
        origin = sm[rfm]
        for i in range(rfm, len(sm)):
            if not np.isnan(sm[i]) and sm[i] - origin >= 10.0:
                m["time_to_10ft_s"] = _r((i - rfm) / fps)
                break
        seg = speed[rfm: rfm + int(round(3.0 * fps))]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            m["peak_speed_ft_s"] = _r(float(seg.max()), 2)
        j = rfm + int(round(0.5 * fps))
        if j < len(speed) and not np.isnan(speed[j]) and not np.isnan(speed[rfm]):
            m["accel_0_5s_ft_s2"] = _r((speed[j] - speed[rfm]) / 0.5, 2)
        if events.arrival is not None:
            m["move_to_arrival_s"] = _r((events.arrival - rfm) / fps)
        if events.slide_start is not None and second_base_ft is not None \
                and not np.isnan(sm[events.slide_start]):
            m["slide_start_dist_ft"] = _r(second_base_ft - sm[events.slide_start], 1)

    m["first_step"] = first_step
    return m
