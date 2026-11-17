"""S8 discriminator: separate real mound deliveries from noise bursts.

A real delivery has a QUIET mound before it: the pitcher stands still at the
circle for ~1 s, then the windup moves pixels. Fielding chains (delivery +
contact + fielding merged into one long run) also start with a quiet mound —
their run START is still the delivery instant. Overlays / crowd pans /
mid-fielding bursts have elevated pre-window energy. So:

  candidate delivery = run whose pre-window (start-40 .. start-5 frames)
  mean count-energy is below PRE_THRESH; instant = center (short runs) or
  start + 7 frames (long runs — windup midpoint).

Outputs run1/s8_clean.npz + csv, with the feature table for calibration.
Usage: python scripts/spike_s8_discriminate.py [PRE_THRESH]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT_NPZ = "run1/s8_clean.npz"
OUT_CSV = "run1/s8_clean.csv"
PRE_FRAMES = 40          # ~1.3 s of quiet mound before the windup
PRE_SKIP = 5             # stop the window this many frames before the run
FPS = 30.0

pre_thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

d = np.load("run1/s8_deliveries.npz")
energies = d["energies"]
starts = d["start_f"].astype(int)
ends = d["end_f"].astype(int)
n_frames = int(d["n_frames"])

rows = []
for s, e in zip(starts, ends):
    ln = e - s + 1
    lo = max(0, s - PRE_FRAMES)
    hi = max(0, s - PRE_SKIP)
    pre = float(energies[lo:hi].mean()) if hi > lo else float("nan")
    instant_f = float(s + e) / 2.0 if ln <= 22 else float(s) + 7.0
    rows.append((s, e, ln, pre, instant_f))

n_pass = 0
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["start_f", "end_f", "len", "pre_energy", "instant_f",
                "instant_s", "delivery"])
    for s, e, ln, pre, inst in rows:
        ok = pre < pre_thresh
        n_pass += ok
        w.writerow([s, e, ln, f"{pre:.2f}", f"{inst:.0f}", f"{inst / FPS:.2f}",
                    int(ok)])
print(f"pre_energy < {pre_thresh}: {n_pass}/{len(rows)} runs pass")

inst = np.array([r[4] for r in rows if r[3] < pre_thresh])
np.savez(OUT_NPZ, instant_f=inst, instant_s=inst / FPS)
print(f"wrote {OUT_NPZ} ({len(inst)} delivery candidates), {OUT_CSV}")

# calibration view: distribution summary
pres = np.array([r[3] for r in rows if not np.isnan(r[3])])
print("pre_energy percentiles p10/p25/p50/p75/p90:",
      np.round(np.percentile(pres, [10, 25, 50, 75, 90]), 2).tolist())
lens = np.array([r[2] for r in rows])
print("run length percentiles p50/p90/p99:",
      np.percentile(lens, [50, 90, 99]).tolist())
