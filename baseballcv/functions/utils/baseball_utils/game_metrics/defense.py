"""Defender positions and movement from static dark blobs.

Per the design review: a per-clip median background ERASES stationary
defenders, so positions at pitch time come from single-frame dark-blob
detection (tracking.detect_dark_blobs) assigned to position priors by greedy
nearest-prior. Identity is POSITION-BASED ONLY (no names; substitutions are
a later pass on event_seq-ordered rows).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MetricsConfig
from .tracking import detect_dark_blobs

Point = Tuple[float, float]


def assign_positions(pts_ft: List[Point], cfg: MetricsConfig,
                     priors: Optional[Dict[str, Tuple[float, float, float]]] = None
                     ) -> Dict[str, Optional[Point]]:
    """Greedy assignment of detected field points to position priors.

    Shortest slot-to-point distances are claimed first, accepted only inside
    the prior's radius. Missing slots map to None — never guessed.
    """
    priors = priors or cfg.defender_priors_ft
    pairs = []
    for slot, (px, py, rad) in priors.items():
        for j, (x, y) in enumerate(pts_ft):
            d = float(np.hypot(x - px, y - py))
            if d <= rad:
                pairs.append((d, slot, j))
    out: Dict[str, Optional[Point]] = {slot: None for slot in priors}
    used_pts = set()
    for d, slot, j in sorted(pairs):
        if out[slot] is None and j not in used_pts:
            out[slot] = tuple(pts_ft[j])
            used_pts.add(j)
    return out


def defender_positions_at(frame_gray: np.ndarray, homography, cfg: MetricsConfig,
                          exclude_pts_ft: List[Point] = ()) -> Dict[str, Optional[Point]]:
    """One frozen frame -> position -> field-ft point (None where unobserved).

    ``exclude_pts_ft``: known non-defender blobs in field ft (batter-runner,
    baserunners) — detections within 5 ft of any are dropped before assignment.
    The batter's boxes are always excluded via config.
    """
    pts_img = detect_dark_blobs(frame_gray, cfg)
    pts_ft = [homography.to_field((x, y)) for (x, y, _a) in pts_img]
    excl = list(cfg.defender_exclusions_ft) + list(exclude_pts_ft)
    kept = [p for p in pts_ft
            if all(np.hypot(p[0] - ex, p[1] - ey) > cfg.defender_exclusion_radius_ft
                   for ex, ey in excl)]
    return assign_positions(kept, cfg)


def fielder_play_metrics(field_xy: Dict[int, Point], contact_frame: Optional[int],
                         fps: float, cfg: MetricsConfig) -> dict:
    """Movement row for one fielder track within a play window.

    ``first_move_s`` runs contact -> first sustained departure from the
    fielder's starting position (the reaction observable). ``dist_first_3s_ft``
    and ``closing_speed_ft_s`` describe the first 3 s after that move.
    All None-able.
    """
    row = {"status": "no_track", "n_frames_tracked": len(field_xy),
           "first_move_frame": None, "first_move_s": None,
           "dist_first_3s_ft": None, "closing_speed_ft_s": None}
    if len(field_xy) < 3 or contact_frame is None:
        return row
    fs = sorted(field_xy)
    xs = np.full(fs[-1] + 1, np.nan)
    ys = np.full(fs[-1] + 1, np.nan)
    for f in fs:
        xs[f], ys[f] = field_xy[f]
    d0 = np.hypot(xs - xs[fs[0]], ys - ys[fs[0]])   # distance from start point
    # sustained departure: speed away from start sustained ~0.2 s
    ok = np.zeros(len(d0), dtype=bool)
    for f in fs[1:]:
        if not np.isnan(d0[f]):
            ok[f] = d0[f] > cfg.first_move_speed_ft_s * 0.5   # displacement form
    run, onset = 0, None
    sustain = max(1, int(round(0.2 * fps)))
    for f in range(contact_frame, len(ok)):
        run = run + 1 if ok[f] else 0
        if run >= sustain:
            onset = f - sustain + 1
            break
    if onset is None:
        row["status"] = "no_first_move"
        return row
    # walk back to where the departure began (displacement at 25% of threshold)
    floor = 0.25 * cfg.first_move_speed_ft_s * 0.5
    while onset > contact_frame and not np.isnan(d0[onset - 1]) and d0[onset - 1] > floor:
        onset -= 1
    row["first_move_frame"] = int(onset)
    row["first_move_s"] = round((onset - contact_frame) / fps, 3)
    # first 3 s after the move: path length + median speed
    end = min(fs[-1], onset + int(round(3.0 * fps)))
    path, prev = 0.0, None
    speeds = []
    for f in fs:
        if f < onset or f > end:
            continue
        if prev is not None:
            path += float(np.hypot(xs[f] - prev[0], ys[f] - prev[1]))
            speeds.append(np.hypot(xs[f] - prev[0], ys[f] - prev[1]) * fps)
        prev = (xs[f], ys[f])
    row["dist_first_3s_ft"] = round(path, 2)
    fin = [s for s in speeds if not np.isnan(s)]
    row["closing_speed_ft_s"] = round(float(np.median(fin)), 2) if fin else None
    row["status"] = "ok"
    return row


def positioning_table(rows: List[dict]) -> List[dict]:
    """Mean defensive position per (batter handedness, position).

    ``rows``: one per play with ``bat_side`` ('L'/'R') and the output of
    ``defender_positions_at`` under key ``positions``. Returns one row per
    (bat_side, position) with n, mean_x_ft, mean_y_ft, std_y_ft.
    """
    import polars as pl

    flat = []
    for r in rows:
        for slot, pt in (r.get("positions") or {}).items():
            if pt is None:
                continue
            flat.append({"bat_side": r.get("bat_side"), "position": slot,
                         "x_ft": pt[0], "y_ft": pt[1]})
    if not flat:
        return []
    df = pl.DataFrame(flat).group_by(["bat_side", "position"]).agg(
        pl.len().alias("n"),
        pl.col("x_ft").mean().round(2).alias("mean_x_ft"),
        pl.col("y_ft").mean().round(2).alias("mean_y_ft"),
        pl.col("y_ft").std().round(2).alias("std_y_ft"),
    ).sort(["bat_side", "position"])
    return df.to_dicts()
