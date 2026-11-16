"""Locate the 23 s clock-map deviation and sweep the pitch/onset lag."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from datetime import date

from baseballcv.functions.utils.baseball_utils.game_sync import GameSyncConfig, detect_onsets
from baseballcv.functions.utils.baseball_utils.game_sync.activity import MotionTimeline
from baseballcv.functions.utils.baseball_utils.game_sync.clock import ClockAnchor, ClockMap

RD = "run1/game_sync"
cfg = GameSyncConfig()

z = np.load(f"{RD}/clock_anchors.npz")
anchors = [ClockAnchor(float(v), float(w), "") for v, w in zip(z["video_s"], z["wall_s"])]
cmap = ClockMap.from_anchors(anchors, date(2026, 11, 12), cfg)

# per-segment deviation: observed wall step vs 1.0 slope
vt, wt = cmap.video_times, cmap.wall_times
dv, dw = np.diff(vt), np.diff(wt)
dev = (wt[1:] - wt[:-1]) - dv  # wall advance minus video advance
bad = np.argsort(-np.abs(dev))[:12]
print("worst segments (video_from, video_to, wall_step-video_step s):")
for i in sorted(bad, key=lambda j: vt[j]):
    print(f"  {vt[i]:7.0f} -> {vt[i+1]:7.0f}   dev={dev[i]:+6.1f}   dw={dw[i]:7.0f} dv={dv[i]:6.0f}")

# lag sweep under off=+90min: matched_frac vs extra lag
pitches = pl.read_parquet(f"{RD}/pbp_cache/game_869992.parquet.pitches")
walls = []
for r in pitches.filter(pl.col("time").is_not_null()).iter_rows(named=True):
    h, m, s = (int(p) for p in r["time"].split(":")[:3])
    walls.append(h * 3600 + m * 60 + s)
walls = np.array(sorted(set(walls)))

tz = np.load(f"{RD}/motion_timeline.npz")
timeline = MotionTimeline(tz["time_s"], tz["energy"], 5.0)
on = detect_onsets(timeline, cfg)["time_s"].to_numpy()

from baseballcv.functions.utils.baseball_utils.game_sync.activity import activity_match_score
base = np.array([cmap.video_at(w + 5400.0) for w in walls])
print("\nlag sweep (extra lag applied to mapped pitch times):")
for lag in (-20, -15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15, 20):
    mf = activity_match_score(base + lag, pl.DataFrame({"time_s": on}), 3.0)
    print(f"  lag {lag:+3d} s -> matched {mf:.1%}")

d = np.min(np.abs(base[:, None] - on[None, :]), axis=1)
print(f"\nnearest-onset distance under +90min, lag 0: median={np.median(d):.1f} "
      f"p25={np.percentile(d, 25):.1f} p75={np.percentile(d, 75):.1f}")
