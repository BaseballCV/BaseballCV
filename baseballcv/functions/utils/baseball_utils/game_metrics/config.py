"""Thresholds for blob-level runner and defender metrics on cenital feeds.

Every number here was sized against the reference feed (854x480 @30 fisheye,
Magallanes park, game 869992 2026-11-12): players are 5-10 px dark blobs on
turf, compression flicker is ~0.8 gray levels, post-homography scale is
~2.3 px/ft on the 1B line and ~0.72 px/ft toward home/CF (see
docs/steal_analysis_handoff.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsConfig:
    # ---- background subtraction + blob extraction (movers) ----
    bg_sample_stride: int = 10          # median bg samples every Nth frame
    bg_diff_thresh: float = 14.0        # gray levels; flicker is ~0.8, blobs >> 14
    blob_min_area_px: int = 8
    blob_max_area_px: int = 400
    morph_open_px: int = 3

    # ---- tracking / association ----
    assoc_gate_px: float = 14.0         # NN gate when bridging gaps (dt > 1)
    assoc_gate_dt1_px: float = 4.0      # tight gate frame-to-frame: a runner
    #   moves <1 px/frame at 27 ft/s (2.3 px/ft), so a 14 px single-step gate
    #   lets tracks hop between people (S6 spike: 44 ft/s chain artifacts)
    vel_blend: float = 0.3              # EMA weight for the new displacement
    track_min_frames: int = 15          # shorter tracks are noise
    max_plausible_speed_ft_s: float = 40.0   # faster than any human sprint ->
    #   a tracking artifact; treated as unobserved

    # field-coordinate ROI for mover detection: the crowd in the stands is a
    # constant-motion source (one spike clip yielded 1305 spurious tracks
    # full-frame) — everything outside the diamond is ignored
    field_roi_ft: tuple = (-20.0, -20.0, 115.0, 115.0)   # x0, y0, x1, y1 (ft)

    # ---- runner selection (field ft; home=(0,0), +x toward 1B) ----
    # The batter merges with catcher/umpire at the plate (S6 spike: chained
    # identities; a start-band rule rejects the track that holds the run), so
    # selection keys on the sustained TOWARD-1B ONSET inside the corridor
    # rather than on where the track begins.
    runner_min_toward_1b_ft_s: float = 8.0   # sustained toward-1B speed to qualify
    # S6 spike v7/v8 lesson: below ~20 ft from home the batter is 2-3 px wide
    # (0.72 px/ft) and merged into the plate pile blob — no onset is observable
    # there (play_0005: the runner emerges from the pile as a solid 20-60 px
    # blob only at 18-26 ft). The onset corridor therefore starts where the
    # batter physically separates.
    onset_d_home_min_ft: float = 20.0   # onset where the blob is trackable...
    onset_d_home_max_ft: float = 40.0   # ...but still well before 1B
    corridor_y_max_ft: float = 20.0     # onset within the 1B corridor |y|

    # ---- trajectory sanity ----
    # homography projections blow up near the horizon (kite_ratio 3.73): any
    # field point outside this box is a projection artifact, not a player
    valid_field_box_ft: tuple = (-30.0, -30.0, 130.0, 130.0)

    # ---- runner metrics ----
    peak_speed_bounds: tuple = (18.0, 35.0)  # sanity band; outside -> suspect status
    arrival_margin_ft: float = 3.0      # "at the bag" = within this of (90, 0)
    speed_window_s: float = 1.0         # best-window peak speed window
    contact_search_s: float = 3.0       # first move must occur within this of clip start

    # ---- static dark-blob detection (defenders at a frozen frame) ----
    dark_blob_contrast_min: float = 10.0    # local dark contrast vs ring
    dark_blob_min_area_px: int = 6
    dark_blob_max_area_px: int = 120
    dark_blob_radius_px: int = 9        # detection neighborhood around each prior

    # ---- defender region priors (field ft): name -> (x, y, radius_ft) ----
    # field frame matches game_sync.calibration.FIELD_POINTS: home (0,0),
    # +x toward 1B, 2B on the 45-degree diagonal
    defender_priors_ft: dict = field(default_factory=lambda: {
        "P": (42.7, 42.7, 8.0),
        "C": (0.0, 0.0, 5.0),
        "1B": (92.0, 2.0, 6.0),
        "2B": (63.6, 63.6, 8.0),
        "3B": (2.0, 92.0, 6.0),
        "SS": (48.0, 58.0, 12.0),
        "LF": (60.0, 130.0, 25.0),
        "CF": (0.0, 155.0, 25.0),
        "RF": (-60.0, 130.0, 25.0),
    })
    # batter's box + plate pile-up: never report a defender here
    defender_exclusions_ft: tuple = ((-5.0, 8.0), (5.0, 8.0))   # L/R batter boxes
    defender_exclusion_radius_ft: float = 5.0

    # ---- defense movement ----
    first_move_speed_ft_s: float = 6.0  # sustained speed that counts as a move
