"""Game video <-> play-by-play synchronization for full-game cenital feeds.

See docs/steal_analysis_handoff.md and the Step-0 spike scripts for the
measured properties of the reference feed (Magallanes park, 2026-11-12).
"""
from .config import GameSyncConfig
from .pbp import GameSnapshot, PBPClient
from .clock import BurnedClockReader, ClockAnchor, ClockMap, parse_timestamp
from .activity import MotionTimeline, activity_match_score, detect_onsets, motion_timeline
from .sync import GameClock, OffsetVerdict, per_play_windows, refine_offset, resolve_offset, validation_report
from .calibration import BaseLocations, build_field_homography, fisheye_diagnostics, locate_bases_from_priors, median_frame
from .clips import cut_clips
from .runner import GameSyncRunner

__all__ = [
    "GameSyncConfig",
    "PBPClient", "GameSnapshot",
    "BurnedClockReader", "ClockAnchor", "ClockMap", "parse_timestamp",
    "MotionTimeline", "motion_timeline", "detect_onsets",
    "GameClock", "OffsetVerdict", "resolve_offset", "refine_offset",
    "per_play_windows", "validation_report",
    "BaseLocations", "build_field_homography", "fisheye_diagnostics",
    "locate_bases_from_priors", "median_frame",
    "cut_clips",
    "GameSyncRunner",
]
