"""S6 v10: the spike path on RE-CUT clips (generous tail, frame-exact).

Reads run1/s6v10/*.mp4 (from scripts/recut_s6_clips.py) and runs the real
package path (track_clip -> select_runner_track -> runner_play_metrics).
Gate unchanged: arrival <= 3 ft of 1B, frac >= 0.8, peak 20-35 ft/s.

Usage: python scripts/spike_s6_v10.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    select_runner_track, track_clip,
)
from baseballcv.functions.utils.baseball_utils.game_metrics.metrics import (
    FIRST_BASE, runner_play_metrics, to_arrays,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

RUN_DIR = "run1/game_sync"
CLIPS = sorted(glob.glob("run1/s6v10/*.mp4"))


def main():
    z = np.load(os.path.join(RUN_DIR, "calibration.npz"))
    hom = Homography.from_matrix(z["matrix"])
    cfg = MetricsConfig()
    gate = []
    for path in CLIPS:
        base = os.path.basename(path)
        try:
            tracks = track_clip(path, cfg, homography=hom, logger=None)
        except Exception as exc:
            print(f"{base}: track_clip FAILED: {exc}")
            gate.append(False)
            continue
        tid, diag = select_runner_track(tracks, hom, cfg)
        if tid is None:
            print(f"{base}: NO runner track ({len(tracks.tracks)} tracks; "
                  f"cands={diag['candidates']})")
            gate.append(False)
            continue
        frames = tracks.tracks[tid]
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
        print(f"{base}: track {tid} onset={onset_d:.1f}ft min_d_1b={d_1b.min():.1f} "
              f"frac={frac:.2f} peak={m['peak_speed_ft_s']} t2_1b={m['time_to_1b_s']} "
              f"frames={len(fs)} [{span[0]}..{span[1]}] status={m['status']} -> "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nGATE: {sum(gate)}/{len(gate)}")


if __name__ == "__main__":
    main()
