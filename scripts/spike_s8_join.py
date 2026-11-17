"""S8 join v3: corroborated-lag matching of PBP pitches to delivery bursts.

History: v1 (greedy earliest) saturated the lag cap on noise; v2 (exact
gap-matched segments) collapsed on long PAs where one delivery is missed.
Key measured fact (v2 output): where a PA matched, its per-pitch lag was
near-constant (median std 1.07 s). So per PA:

  1. For each candidate last-pitch burst b in [db_last-150, db_last+5]:
     hypothesized lag L = db_last - b.
  2. Corroboration = # earlier pitches with a burst within 8 s of db_i - L.
  3. Viterbi across PAs: maximize corroboration - 0.35*|L - L_prev|,
     enforcing monotone bursts and lag jumps <= 60 s.

Then each pitch maps to its nearest burst to db_i - L (misses allowed ->
interpolated video position). Validation: pitch no.20 (play_0005) must land
at lag ~32.7 s (journal 2026-11-16).

Outputs run1/s8_join.csv (per-pitch) + run1/s8_join_pa.csv (per-PA).
Usage: python scripts/spike_s8_join.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

OUT_CSV = "run1/s8_join.csv"
OUT_PA_CSV = "run1/s8_join_pa.csv"
DB_TO_VIDEO_WALL_S = 5400.0
MAX_LAG_S = 150.0
CORROB_TOL_S = 8.0
LAG_PENALTY = 0.35
MAX_LAG_JUMP_S = 60.0
MIN_PA_GAP_S = 6.0        # bursts of consecutive PAs cannot be closer

anchors = np.load("run1/game_sync/clock_anchors.npz")
av, aw = anchors["video_s"].astype(float), anchors["wall_s"].astype(float)

d = np.load("run1/s8_deliveries.npz")
cf = d["center_f"] / 30.0
merged = []
for v in sorted(cf):
    if merged and v - merged[-1] < 1.5:   # one windup can split into 2 runs
        merged[-1] = (merged[-1] + v) / 2.0
    else:
        merged.append(v)
cand_video_s = np.array(merged)
cand_wall = np.interp(cand_video_s, av, aw)
print(f"{len(cand_wall)} bursts after 1.5 s merge")

p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
p = p.with_columns(
    (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1_000_000_000
     + DB_TO_VIDEO_WALL_S).alias("wall")
).sort("no")

pas = []
for pitch in p.iter_rows(named=True):
    key = (pitch["inning"], pitch["is_top_inning"], pitch["plate_appearance"])
    if not pas or pas[-1]["key"] != key:
        pas.append({"key": key, "rows": []})
    pas[-1]["rows"].append(pitch)

NB = len(cand_wall)


def corrob(j, lag):
    """How many of the last K earlier pitches corroborate lag for burst j."""
    rows = pas_cur["rows"]
    n = min(len(rows), 6)
    hit = 0
    for i in range(len(rows) - n, len(rows) - 1):
        target = rows[i]["wall"] - lag
        k = np.searchsorted(cand_wall, target - CORROB_TOL_S)
        if k < NB and cand_wall[k] <= target + CORROB_TOL_S:
            hit += 1
    return hit


def pa_candidates(pi):
    """[(burst_idx, lag, corroboration)] for PA pi's last-pitch burst."""
    global pas_cur
    pas_cur = pas[pi]
    wN = pas[pi]["rows"][-1]["wall"]
    out = []
    j0 = np.searchsorted(cand_wall, wN - MAX_LAG_S)
    for j in range(j0, NB):
        if cand_wall[j] > wN + 5:
            break
        lag = wN - cand_wall[j]
        if lag < -5:
            continue
        out.append((j, lag, corrob(j, lag)))
    return out


# Viterbi over PAs; state = burst index
cands = [pa_candidates(pi) for pi in range(len(pas))]
best = [dict() for _ in pas]
for j, lag, c in cands[0]:
    best[0][j] = (-c, lag, -1)     # cost, lag, backpointer
for pi in range(1, len(pas)):
    prev_end = {j: cand_wall[j] for j in best[pi - 1]}
    for j, lag, c in cands[pi]:
        if cand_wall[j] < min(prev_end, default=0) + MIN_PA_GAP_S:
            continue
        bestc, bp = None, -1
        for pj, (pc, plag, _) in best[pi - 1].items():
            if cand_wall[j] - prev_end[pj] < MIN_PA_GAP_S:
                continue
            jump = abs(lag - plag)
            if jump > MAX_LAG_JUMP_S:
                continue
            cc = pc - c + LAG_PENALTY * jump
            if bestc is None or cc < bestc:
                bestc, bp = cc, pj
        if bestc is not None:
            best[pi][j] = (bestc, lag, bp)

chosen = [None] * len(pas)
fin = min(best[-1], key=lambda j: best[-1][j][0], default=None)
if fin is not None:
    j = fin
    for pi in range(len(pas) - 1, -1, -1):
        chosen[pi] = j
        j = best[pi][j][2]

rows_out, pa_out = [], []
for pi, pa in enumerate(pas):
    j = chosen[pi]
    if j is None:
        cs = cands[pi]
        if cs:   # fallback: max corroboration, then lag nearest 60
            j, lag, c = max(cs, key=lambda t: (t[2], -abs(t[1] - 60)))
        else:
            for r in pa["rows"]:
                rows_out.append((r["no"], r["time"], pi + 1, "", "", "", "NO_CAND"))
            pa_out.append((pi + 1, len(pa["rows"]), "", "", "NO_CAND"))
            continue
    else:
        lag = best[pi][j][1]
    cor = corrob(j, lag)
    pa_out.append((pi + 1, len(pa["rows"]), f"{cand_video_s[j]:.1f}",
                   f"{lag:.1f}", cor, "ok"))
    for i, r in enumerate(pa["rows"]):
        target = r["wall"] - lag
        k = np.searchsorted(cand_wall, target - CORROB_TOL_S)
        if k < NB and cand_wall[k] <= target + CORROB_TOL_S:
            rows_out.append((r["no"], r["time"], pi + 1, f"{cand_wall[k]:.1f}",
                             f"{cand_video_s[k]:.1f}", f"{r['wall'] - cand_wall[k]:.1f}",
                             "ok"))
        else:   # missed delivery: interpolate from the PA lag
            vs = float(np.interp(target, aw, av))
            rows_out.append((r["no"], r["time"], pi + 1, f"{target:.1f}",
                             f"{vs:.1f}", f"{lag:.1f}", "interp"))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["pitch_no", "db_time", "pa_seq", "burst_wall_s", "burst_video_s",
                "lag_s", "status"])
    w.writerows(rows_out)
with open(OUT_PA_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["pa_seq", "n_pitches", "last_burst_video_s", "lag_s", "corrob", "status"])
    w.writerows(pa_out)

ok = [r for r in rows_out if r[-1] == "ok"]
print(f"\nmatched {len(ok)}/{len(rows_out)} pitches "
      f"({sum(1 for r in rows_out if r[-1] == 'interp')} interpolated)")
lags = np.array([float(r[5]) for r in ok if r[-1] == "ok"])
print(f"lag: median {np.median(lags):.1f}  p10 {np.percentile(lags, 10):.1f}  "
      f"p90 {np.percentile(lags, 90):.1f}  min {lags.min():.1f}  max {lags.max():.1f}")
cors = [p_[4] for p_ in pa_out if len(p_) > 5 and p_[5] == "ok"]
print(f"corroboration: median {np.median(cors):.0f}, "
      f"PAs with corrob>=2: {sum(1 for c in cors if c >= 2)}/{len(cors)}")
stds = []
for pi, pa in enumerate(pas):
    v = [float(r[5]) for r in rows_out if r[2] == pi + 1 and r[-1] == "ok"]
    if len(v) >= 2:
        stds.append(np.std(v))
print(f"per-PA lag std: median {np.median(stds):.2f} over {len(stds)} PAs; "
      f">5 s: {sum(1 for x in stds if x > 5)}")
r20 = next((r for r in rows_out if r[0] == 20), None)
if r20:
    print(f"VALIDATION pitch no.20 (play_0005): lag {r20[5]} s (expect ~32.7)")
print(f"wrote {OUT_CSV}, {OUT_PA_CSV}")
