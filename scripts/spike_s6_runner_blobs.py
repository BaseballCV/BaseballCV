"""S6 spike: can we track the batter-runner as a dark blob on the cenital feed?

Throwaway gate-check for the game_metrics plan. Runs the REAL package path
(game_metrics.tracking.track_clip + select_runner_track) on ~10 ground-ball
clips and reports the plan's gate numbers.

GATE (from the plan): >= 7/10 clips track from the home band to within 3 ft of
1B, >= 80% of frames tracked, peak speed 20-35 ft/s.

Run AFTER GameSyncRunner has emitted run1/game_sync/calibration.npz.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    select_runner_track, track_clip,
)
from baseballcv.functions.utils.baseball_utils.game_metrics.metrics import (
    FIRST_BASE, runner_play_metrics, to_arrays,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

RUN_DIR = "run1/game_sync"


def main():
    z = np.load(os.path.join(RUN_DIR, "calibration.npz"))
    hom = Homography.from_matrix(z["matrix"])
    cfg = MetricsConfig()
    cfg = MetricsConfig()
    manifest = pl.read_csv(os.path.join(RUN_DIR, "manifest.csv"))
    gbs = manifest.filter(
        (pl.col("event_text") == "Groundout") & (pl.col("base_state_before") == 0)
    ).head(10)
    print(f"{len(gbs)} groundout clips, empty base state")
    gate = []
    for row in gbs.iter_rows(named=True):
        path = row["clip_path"]
        try:
            tracks = track_clip(path, cfg, homography=hom, logger=None)
        except Exception as exc:
            print(f"{os.path.basename(path)}: track_clip FAILED: {exc}")
            gate.append(False)
            continue
        tid, diag = select_runner_track(tracks, hom, cfg)
        if tid is None:
            print(f"{os.path.basename(path)}: NO runner track "
                  f"({len(tracks.tracks)} tracks >= {cfg.track_min_frames} f; "
                  f"cands={diag['candidates']})")
            gate.append(False)
            continue
        frames = tracks.tracks[tid]
        # convert to FIELD feet before the box-sensitive array build (raw
        # pixels would all fall outside valid_field_box_ft)
        xs, ys = to_arrays({f: hom.to_field((p[0], p[1])) for f, p in frames.items()},
                           tracks.n_frames, cfg)
        fs = np.array(sorted(frames))
        d_1b = np.hypot(xs[fs] - FIRST_BASE[0], ys[fs] - FIRST_BASE[1])
        span = (fs[0], fs[-1])
        frac = len(fs) / (span[1] - span[0] + 1)
        m = runner_play_metrics({f: hom.to_field(frames[f][:2]) for f in frames},
                                tracks.fps, tracks.n_frames, cfg)
        ok = (d_1b.min() <= cfg.arrival_margin_ft and frac >= 0.8
              and m["peak_speed_ft_s"] is not None
              and 20.0 <= m["peak_speed_ft_s"] <= 35.0)
        gate.append(ok)
        onset_d = next((c["onset_d_home_ft"] for c in diag["candidates"]
                        if c["track_id"] == tid), float("nan"))
        print(f"{os.path.basename(path)}: track {tid} onset_d_home="
              f"{onset_d:.1f}ft "
              f"min_d_1b={d_1b.min():.1f}ft frac={frac:.2f} "
              f"peak={m['peak_speed_ft_s']} t2_1b={m['time_to_1b_s']} "
              f"frames={len(fs)} [{span[0]}..{span[1]}] status={m['status']} -> "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nGATE: {sum(gate)}/{len(gate)} (need >= 7/{len(gate)})")


if __name__ == "__main__":
    main()
