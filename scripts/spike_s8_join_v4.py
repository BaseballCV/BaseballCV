"""S8 joiner v4: monotone DTW of PBP pitches to DISCRIMINATED delivery bursts.

v4 differs from the v3-era joiners: inputs are the discriminator's clean
candidates (quiet-mound-before runs; long delivery+contact chains use the
windup midpoint), lag window tightens to +-15 s, and the run produces
per-pitch wall/video instants plus diagnostics keyed to the clock-verified
anchors:
  - play_0005 (pitch no.20): play in-clip at f1320-1364 -> video_s ~674.2,
    wall ~67438 (lag ~0)
  - play_0003 (pitch no.11 G43): clip-verified burst wall ~67057.5 (lag +1.5)
  - play_0017 (pitches no.60/61): bursts wall ~69077.5 (lag +0.5) and
    ~69110 (lag -4)

Usage: python scripts/spike_s8_join_v4.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

OUT_CSV = "run1/s8_join_v4.csv"
DB_TO_VIDEO_WALL_S = 5400.0
LAGWIN = 15.0
SKIP_PITCH = 25.0
SKIP_BURST = 1.0

a = np.load("run1/game_sync/clock_anchors.npz")
av, aw = a["video_s"].astype(float), a["wall_s"].astype(float)

c = np.load("run1/s8_clean.npz")
cand_video_s = c["instant_s"].astype(float)
cand_wall = np.interp(cand_video_s, av, aw)
print(f"{len(cand_wall)} clean delivery candidates")

p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
p = p.with_columns(
    (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1_000_000_000
     + DB_TO_VIDEO_WALL_S).alias("wall")
).sort("no")
walls = p["wall"].to_numpy().astype(float)
nos = p["no"].to_numpy()
res = p["play_result"].to_numpy()
n, m = len(walls), len(cand_wall)

INF = 1e18
dp = np.full((n + 1, m + 1), INF)
bp = np.zeros((n + 1, m + 1), dtype=np.int8)   # 1 skip pitch, 2 match
dp[0, :] = 0.0
for i in range(1, n + 1):
    am = int(np.argmin(dp[i - 1, :]))
    dp[i, :] = dp[i - 1, am] + SKIP_PITCH
    bp[i, :] = 1
    lo = np.searchsorted(cand_wall, walls[i - 1] - LAGWIN)
    hi = np.searchsorted(cand_wall, walls[i - 1] + LAGWIN)
    for j in range(lo, hi):
        cst = dp[i - 1, j] + abs(cand_wall[j] - walls[i - 1])
        if cst < dp[i, j]:
            dp[i, j] = cst
            bp[i, j] = 2
j = int(np.argmin(dp[n, :]))
i = n
matches = {}
while i > 0:
    if bp[i, j] == 2:
        matches[i - 1] = j     # dp row i = after pitch index i-1
        i -= 1
    else:
        j = int(np.argmin(dp[i - 1, :]))
        i -= 1

lags = np.array([cand_wall[jj] - walls[ii] for ii, jj in matches.items()])
print(f"matched {len(matches)}/{n} pitches")
if len(lags):
    print("lag p5/p25/p50/p75/p95:",
          np.round(np.percentile(lags, [5, 25, 50, 75, 95]), 1).tolist())

# per-PA lag consistency
pa = (p["inning"].to_numpy(), p["is_top_inning"].to_numpy(),
      p["plate_appearance"].to_numpy())
import collections
per = collections.defaultdict(list)
for ii, jj in matches.items():
    per[(pa[0][ii], pa[1][ii], pa[2][ii])].append(cand_wall[jj] - walls[ii])
stds = [np.std(v) for v in per.values() if len(v) >= 2]
print("per-PA lag std median:", round(float(np.median(stds)), 2),
      "over", len(stds), "PAs;  >5 s:", sum(1 for s in stds if s > 5))

res_matched = [ii for ii in matches if res[ii] != ""]
res_tot = int((res != "").sum())
if res_matched:
    rl = [abs(cand_wall[jj] - walls[ii]) for ii, jj in matches.items()
          if res[ii] != ""]
    print(f"result pitches matched: {len(res_matched)}/{res_tot}; "
          f"|lag| median {np.median(rl):.1f} s")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["pitch_no", "db_time", "burst_wall_s", "burst_video_s",
                "lag_s", "status"])
    for ii in range(n):
        if ii in matches:
            jj = matches[ii]
            w.writerow([nos[ii], p["time"][ii], f"{cand_wall[jj]:.1f}",
                        f"{cand_video_s[jj]:.1f}",
                        f"{cand_wall[jj] - walls[ii]:.1f}", "ok"])
        else:
            w.writerow([nos[ii], p["time"][ii], "", "", "", "skip"])

print("\nanchors:")
for no, expect in ((20, "~0 (play_0005 f1340 in-clip)"), (11, "+1.5 (play_0003)"),
                   (60, "+0.5 (play_0017)"), (61, "-4 (play_0017)")):
    ii = int(np.where(nos == no)[0][0])
    if ii in matches:
        jj = matches[ii]
        print(f"  pitch {no}: lag {cand_wall[jj] - walls[ii]:+.1f} (expect {expect})")
    else:
        print(f"  pitch {no}: SKIPPED (expect {expect})")
print(f"wrote {OUT_CSV}")
