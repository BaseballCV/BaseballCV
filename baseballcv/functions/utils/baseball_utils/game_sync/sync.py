"""Offset resolution between the DB clock and the burned-in camera clock.

Spike finding (2026-11-12 Magallanes feed): the DB pitch_by_pitch.time ran a
CONSTANT 90:00 behind the camera overlay, with intervals agreeing within ~9 s
over 3.6 h. The grid below is validation, not discovery: it scores each
candidate offset by activity-onset alignment so a wrong clock assumption fails
loudly instead of silently syncing to a shifted timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import polars as pl

from baseballcv.utilities import BaseballCVLogger

from .activity import MotionTimeline, activity_match_score, detect_onsets
from .clock import ClockMap
from .config import GameSyncConfig


@dataclass
class OffsetVerdict:
    chosen_offset_s: float
    candidates: pl.DataFrame          # offset_s, matched_frac
    runner_up_offset_s: Optional[float]
    margin: float
    reason: str


@dataclass
class GameClock:
    """Final DB-time -> video-time map: clock anchors + validated constant offset."""
    clock_map: ClockMap
    offset_s: float = 0.0             # DB wall = camera wall - offset_s

    def db_to_video(self, db_wall_s: float) -> float:
        return self.clock_map.video_at(db_wall_s + self.offset_s)

    def video_to_db_wall(self, video_s: float) -> float:
        return self.clock_map.wall_at(video_s) - self.offset_s

    def db_datetime_to_video(self, dt: datetime) -> float:
        base = datetime.combine(self.clock_map.game_date, datetime.min.time())
        return self.db_to_video((dt - base).total_seconds())


def resolve_offset(clock_map: ClockMap, db_pitch_walls: Sequence[float],
                   timeline: MotionTimeline, config: Optional[GameSyncConfig] = None,
                   logger=None) -> OffsetVerdict:
    """Score each candidate constant offset by pitch/onset alignment.

    db_pitch_walls: seconds-of-day per pitch on the DB clock (from pitch_by_pitch.time).
    """
    cfg = config or GameSyncConfig()
    logger = logger or BaseballCVLogger.get_logger("resolve_offset")
    onsets = detect_onsets(timeline, cfg)
    rows = []
    for minutes in cfg.offset_candidates_min:
        off = minutes * 60.0
        video_times = np.array([clock_map.video_at(w + off) for w in db_pitch_walls])
        score = activity_match_score(video_times, onsets, cfg.onset_match_tol_s)
        rows.append({"offset_min": minutes, "offset_s": off, "matched_frac": score})
    cand = pl.DataFrame(rows).sort("matched_frac", descending=True)
    best = cand.row(0, named=True)
    runner = cand.row(1, named=True) if len(cand) > 1 else None
    margin = best["matched_frac"] - (runner["matched_frac"] if runner else 0.0)
    reason = (f"offset {best['offset_min']:+d} min matched {best['matched_frac']:.0%} of pitches "
              f"to activity onsets within {cfg.onset_match_tol_s:.0f} s")
    if runner is not None:
        reason += f" (runner-up {runner['offset_min']:+d} min at {runner['matched_frac']:.0%})"
    if margin < cfg.min_margin_frac:
        reason += " — LOW MARGIN; do not trust this sync"
        logger.warning(reason)
    else:
        logger.info(reason)
    return OffsetVerdict(chosen_offset_s=best["offset_s"], candidates=cand,
                         runner_up_offset_s=runner["offset_s"] if runner else None,
                         margin=margin, reason=reason)


def refine_offset(clock: GameClock, db_pitch_walls: Sequence[float],
                  timeline: MotionTimeline, config: Optional[GameSyncConfig] = None) -> GameClock:
    """Fine shift within +-refine_window_s minimizing median pitch-to-onset distance."""
    cfg = config or GameSyncConfig()
    onsets = detect_onsets(timeline, cfg)
    ot = onsets["time_s"].to_numpy() if not onsets.is_empty() else np.array([])
    video_times = np.array([clock.db_to_video(w) for w in db_pitch_walls])
    best_off, best_med = clock.offset_s, np.inf
    for off in np.arange(-cfg.refine_window_s, cfg.refine_window_s + 1e-9, cfg.refine_step_s):
        shifted = video_times + off
        if len(ot):
            dists = np.min(np.abs(shifted[:, None] - ot[None, :]), axis=1)
            med = float(np.median(dists))
            if med < best_med:
                best_med, best_off = med, clock.offset_s + float(off)
    return GameClock(clock_map=clock.clock_map, offset_s=best_off)


def per_play_windows(play_times: pl.DataFrame, clock: GameClock,
                     config: Optional[GameSyncConfig] = None,
                     duration_s: float = np.inf) -> pl.DataFrame:
    """Video window per play: [PA first pitch - pre_roll, PA last pitch + post_roll]."""
    cfg = config or GameSyncConfig()
    rows = []
    base = datetime.combine(clock.clock_map.game_date, datetime.min.time())
    for r in play_times.iter_rows(named=True):
        start_dt = r.get("pa_first_pitch_time_dt") or r.get("event_time_dt")
        end_dt = r.get("pa_last_pitch_time_dt") or start_dt
        if start_dt is None:
            continue
        start_s = clock.db_to_video((start_dt - base).total_seconds()) - cfg.pre_roll_s
        end_s = clock.db_to_video((end_dt - base).total_seconds()) + cfg.post_roll_s if end_dt else start_s + 5.0
        start_s = max(0.0, start_s)
        end_s = min(duration_s, max(end_s, start_s + 1.0))
        rows.append({**{k: r.get(k) for k in
                        ("event_seq", "at_bat_index", "inning", "is_top_inning", "event_type",
                         "event_text", "batter_name", "pitcher_name", "outs_before",
                         "base_state_before")},
                     "video_start_s": round(start_s, 2), "video_end_s": round(end_s, 2)})
    return pl.DataFrame(rows)


def validation_report(db_pitch_walls: Sequence[float], clock: GameClock,
                      timeline: MotionTimeline,
                      config: Optional[GameSyncConfig] = None) -> dict:
    cfg = config or GameSyncConfig()
    onsets = detect_onsets(timeline, cfg)
    video_times = np.array([clock.db_to_video(w) for w in db_pitch_walls])
    matched = activity_match_score(video_times, onsets, cfg.onset_match_tol_s)
    diag = clock.clock_map.diagnostics()
    return {
        "n_pitches": len(db_pitch_walls),
        "matched_frac": matched,
        "offset_s": clock.offset_s,
        **diag,
    }
