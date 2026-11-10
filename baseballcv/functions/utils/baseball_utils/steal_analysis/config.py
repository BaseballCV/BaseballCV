from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StealConfig:
    """Every threshold used by the steal analysis pipeline, in one place.

    Distances are feet, times are seconds, speeds are feet/second unless noted.
    """

    # --- base detection ---
    base_conf: float = 0.40            # v2.1 loses wide-shot recall at 0.5 (#183)
    base_size_in: float = 18.0         # 18 in bags (2023+); use 15 for older footage
    base_spacing_ft: float = 90.0
    min_base_frames: int = 8           # frames a base cluster must persist in a shot
    base_cluster_frac: float = 0.6     # center distance (x box width) merged into one cluster

    # --- shots ---
    shot_cut_corr: float = 0.55        # HSV histogram correlation below this = cut
    min_shot_frames: int = 20

    # --- runner identification ---
    min_steal_travel_ft: float = 10.0  # runner must travel this far toward 2B to count as a steal
    ambiguous_ratio: float = 0.6       # runner-up travel / best travel above this => ambiguous
    candidate_radius_ft: float = 30.0  # candidates must start within this of first base

    # --- event detection ---
    smooth_window_s: float = 0.12
    move_speed_ft_s: float = 4.0       # runner "first move" speed threshold
    move_sustain_s: float = 0.15
    pitcher_move_k: float = 3.0        # std-devs above baseline motion energy
    pitcher_move_floor: float = 0.02   # minimum energy rise (torso-lengths / frame)
    pitcher_sustain_s: float = 0.10
    pitcher_baseline_s: float = 0.5
    arrival_margin_ft: float = 1.5

    # --- first-step classification ---
    first_step_window_s: float = 0.4

    # --- scale ---
    min_scale_conf: float = 0.4        # distance labels need at least this
    scale_confidence: dict = field(default_factory=lambda: {
        "homography": 0.9, "two_bases": 0.6, "runner_height": 0.5, "single_base": 0.3,
    })

    # --- categories ---
    lead_short_ft: float = 11.0        # < short = "short"
    lead_long_ft: float = 13.5         # >= long = "long"
    walking_speed_ft_s: float = 1.0    # pre-pitch mean speed above this = "walking"
    jump_early_s: float = -0.05        # reaction below this = "early"
    jump_late_s: float = 0.15          # reaction above this = "late"
