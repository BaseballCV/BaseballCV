"""Runner and defender metrics from synced per-play clips (cenital feeds).

Loads game_sync's calibration.npz / sync_state.json + manifest, tracks the
batter-runner and defenders as blobs, exports per-play metric rows. See
docs/steal_analysis_handoff.md for the measured feed properties and gates.
"""
from .config import MetricsConfig
from .tracking import (
    BlobTracks, associate_blobs, detect_dark_blobs, extract_blobs,
    median_background, select_runner_track, track_clip, track_field_xy,
)
from .metrics import (
    baserunning_legs, first_move_frame, runner_play_metrics, to_arrays,
)
from .defense import (
    assign_positions, defender_positions_at, fielder_play_metrics,
    positioning_table,
)
from .runner import GameMetricsRunner

__all__ = [
    "MetricsConfig",
    "BlobTracks", "track_clip", "select_runner_track", "track_field_xy",
    "extract_blobs", "detect_dark_blobs", "associate_blobs", "median_background",
    "runner_play_metrics", "baserunning_legs", "first_move_frame", "to_arrays",
    "defender_positions_at", "assign_positions", "fielder_play_metrics",
    "positioning_table",
    "GameMetricsRunner",
]
