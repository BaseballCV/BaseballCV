"""Score offsets by aggregate timeline energy at mapped pitch times (no onsets)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from datetime import date

from baseballcv.functions.utils.baseball_utils.game_sync.clock import ClockAnchor, ClockMap

RD = "run1/game_sync"
z = np.load(f"{RD}/clock_anchors.npz")
anchors = [ClockAnchor(float(v), float(w), "") for v, w in zip(z["video_s"], z["wall_s"])]
cmap = ClockMap.from_anchors(anchors, date(2026, 11, 12), GameSyncConfig := __import__(
    "baseballcv.functions.utils.baseball_utils.game_sync.config", fromlist=["GameSyncConfig"]).GameSyncConfig())

pitches = pl.read_parquet(f"{RD}/pbp_cache/game_869992.parquet.pitches")
walls = []
for r in pitches.filter(pl.col("time").is_not_null()).iter_rows(named=True):
    h, m, s = (int(p) for p in r["time"].split(":")[:3])
    walls.append(h * 3600 + m * 60 + s)
walls = np.array(sorted(set(walls)))

tz = np.load(f"{RD}/motion_timeline.npz")
t, e = tz["time_s"], tz["energy"]
med, noise = np.median(e), np.median(np.abs(e - np.median(e))) * 1.4826

def energy_score(off):
    vt = np.array([cmap.video_at(w + off) for w in walls])
    vt = vt[(vt > t[0] + 2) & (vt < t[-1] - 2)]
    idx = np.searchsorted(t, vt)
    z = (e[idx] - med) / noise          # energy AT the mapped pitch instant
    # and in a small neighborhood (delivery motion spans ~1-2 samples)
    acc = []
    for k in (-1, 0, 1):
        j = np.clip(idx + k, 0, len(e) - 1)
        acc.append((e[j] - med) / noise)
    return np.mean(z), np.max(np.stack(acc), axis=1).mean(), len(vt)

print(f"global: median={med:.4f} noise(MAD)={noise:.4f} n_pitches={len(walls)}")
print(f"{'offset':>8} {'z@t':>6} {'z@max(+/-1)':>12}")
for minutes in (-120, -90, -60, -30, 0, 30, 60, 90, 120):
    z0, zm, n = energy_score(minutes * 60.0)
    print(f"{minutes:+4d}min {z0:6.2f} {zm:12.2f}")

# where in the game does +90 elevate? split into 6 segments
vt = np.array([cmap.video_at(w + 5400.0) for w in walls])
idx = np.clip(np.searchsorted(t, vt), 0, len(e) - 1)
zz = (e[idx] - med) / noise
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)):
    a, b = np.percentile(vt, [lo, hi])
    m = (vt >= a) & (vt <= b)
    print(f"video {a:6.0f}-{b:6.0f}: mean z={zz[m].mean():+.2f} frac>2: {np.mean(zz[m] > 2):.0%} n={m.sum()}")
