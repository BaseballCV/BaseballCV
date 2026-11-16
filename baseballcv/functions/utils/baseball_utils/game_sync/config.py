"""Configuration for game video <-> play-by-play synchronization.

All thresholds in one dataclass, mirroring `steal_analysis.config.StealConfig`.
Values are calibrated on the Magallanes cenital feed (2026-11-12); see
scripts/spike_s1*.py findings in the branch history.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameSyncConfig:
    # --- burned-in clock (clock.py) ---
    ocr_crop_x0_frac: float = 0.60     # top-right crop left edge, width fraction
    ocr_crop_y1_frac: float = 0.09     # top-right crop bottom edge, height fraction
    ocr_upscale: int = 4
    ocr_step_s: float = 60.0           # anchor spacing across the file
    ocr_retry_offset_s: float = 1.0    # retry a failed read 1 s later
    clock_jump_s: float = 1200.0       # anchor-to-anchor wall delta beyond this folds mod 1 h
    clock_fold_s: int = 3600           # the recorder's display-oscillation period (NTP/DST toggle)
    clock_max_seg_dev_s: float = 3.0   # per-segment residual alarm threshold

    # --- offset resolution (sync.py) ---
    offset_candidates_min: tuple = (-120, -90, -60, -30, 0, 30, 60, 90, 120)  # minutes
    onset_match_tol_s: float = 3.0
    refine_window_s: float = 5.0
    refine_step_s: float = 0.25
    min_margin_frac: float = 0.25      # winner must beat runner-up by this fraction
    # interval-fit feasibility (full-game recording: first pitch near the start,
    # last pitch near EOF). Measured 2026-11-12 feed: video starts 9 s before
    # the DB first pitch and ends 102 s after the DB last pitch.
    sync_pad_max_s: float = 1200.0     # max video lead-in before the DB first pitch
    sync_pad_end_s: float = 1200.0     # max video tail after the DB last pitch

    # --- play windows (sync.py) ---
    pre_roll_s: float = 20.0
    post_roll_s: float = 8.0

    # --- activity timeline (activity.py) ---
    activity_fps: float = 5.0
    activity_downscale: float = 0.5
    activity_roi: tuple = (0.30, 0.30, 0.75, 0.85)  # infield box, frame fractions (x0,y0,x1,y1)
    activity_diff_thresh: float = 3.0    # gray levels: a pixel counts as "changed" above this
    onset_floor_window_s: float = 20.0   # rolling-median floor; long so bursts don't lift it
    onset_noise_mult: float = 3.0        # threshold = floor + max(mad*mult, floor_add)
    onset_floor_add: float = 0.004       # absolute floor (fraction of ROI pixels)
    onset_sustain_s: float = 0.3

    # --- calibration (calibration.py) ---
    calib_median_window_s: float = 72.0
    calib_n_frames: int = 36
    search_radius_px: int = 12
    blob_contrast_min: float = 5.0

    # --- clips (clips.py) ---
    ffmpeg_bin: str = "ffmpeg"
    clip_copy: bool = True

    # --- remote DB (pbp.py) ---
    ssh_host: str = "212.28.189.37"
    ssh_user: str = "carlos"
    db_path: str = "/var/www/navegantes/baseball_data.db"
    db_read_user: str = "www-data"     # WAL safety: always read as the DB owner
