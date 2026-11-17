"""S8 alignment spike: scan the WHOLE source feed for mound-delivery bursts.

The DB pitch times are scorekeeper mush (variable 21-97 s lags, journal 2026-11-16).
A mound delivery is video ground truth: the pitcher's windup moves pixels inside
a small circle at the mound. Scan every frame, threshold the circle diff energy,
report burst runs. Later steps clock-anchor each burst and join to PBP pitches.

Mound px comes from the persisted homography: field_to_image((42.8, 42.8)) --
verified = (505, 221) on this feed, matching the play_0005 prototype bursts.

Output: run1/s8_deliveries.npz (start_f, end_f, center_f, peak_e) + a csv.

Usage: python scripts/spike_s8_delivery_scan.py [VIDEO]
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
OUT_NPZ = "run1/s8_deliveries.npz"
OUT_CSV = "run1/s8_deliveries.csv"

# Calibrated on play_0005 (known deliveries f327-342, f351-366; weak f789;
# fielding sequence f1325-1360): burst frames have 10-27 circle pixels above
# diff 20 (quiet: 0-7), and a delivery run is 10-16 frames while the follow-up
# fielding action is one continuous ~36-frame run — length separates them.
PIX_DIFF_THRESH = 20    # per-pixel gray diff that counts as motion
PIX_COUNT_THRESH = 8    # burst when >= 9 circle pixels exceed it
LAG_FRAMES = 5          # f vs f+5 (prototype)
MIN_RUN = 4
MAX_RUN = 22            # longer = fielding play / replay, not a delivery
MERGE_GAP = 3           # merge runs separated by <3 quiet frames
FPS = 30.0

z = np.load("run1/game_sync/calibration.npz")
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography
hom = Homography.from_matrix(z["matrix"])
mx, my = hom.field_to_image((42.8, 42.8))   # mound center
R = 9
print(f"mound circle: px({mx:.0f},{my:.0f}) r{R}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video: fps={fps:.2f} frames={total} ({total / fps / 60:.0f} min)")

yy, xx = np.mgrid[:480, :854]
mask = (xx - mx) ** 2 + (yy - my) ** 2 <= R * R

prev = None
energies = np.zeros(max(total, 1), dtype=np.float32)
t0 = time.time()
f = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if prev is not None and f >= LAG_FRAMES:
        energies[f - LAG_FRAMES] = (cv2.absdiff(g, prev)[mask] > PIX_DIFF_THRESH).sum()
    prev = g
    f += 1
    if f % 30000 == 0:
        el = time.time() - t0
        print(f"  {f}/{total} frames ({el:.0f} s, {f / el:.0f} fps)", flush=True)
cap.release()
n = f
if n == 0:
    raise SystemExit(f"no frames decoded from {VIDEO}")
energies = energies[:n]
print(f"scanned {n} frames in {time.time() - t0:.0f} s")

# burst runs
burst = energies > PIX_COUNT_THRESH
runs = []
f = 0
while f < n:
    if burst[f]:
        s = f
        while f < n and burst[f]:
            f += 1
        runs.append([s, f - 1])
    else:
        f += 1
# merge near runs
merged = []
for r in runs:
    if merged and r[0] - merged[-1][1] <= MERGE_GAP:
        merged[-1][1] = r[1]
    else:
        merged.append(r)
# keep plausible deliveries
kept = []
for s, e in merged:
    ln = e - s + 1
    if MIN_RUN <= ln <= MAX_RUN:
        seg = energies[s:e + 1]
        kept.append((s, e, float(s + e) / 2.0, float(seg.max())))
print(f"{len(merged)} runs -> {len(kept)} plausible deliveries")

with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["start_f", "end_f", "center_f", "peak_e", "center_s"])
    for s, e, c, pk in kept:
        w.writerow([s, e, int(c), f"{pk:.1f}", f"{c / FPS:.2f}"])

if kept:
    arr = np.array([(s, e, c, pk) for s, e, c, pk in kept])
    np.savez(OUT_NPZ, start_f=arr[:, 0], end_f=arr[:, 1],
             center_f=arr[:, 2], peak_e=arr[:, 3])
else:
    np.savez(OUT_NPZ, start_f=np.zeros(0), end_f=np.zeros(0),
             center_f=np.zeros(0), peak_e=np.zeros(0))
print(f"wrote {OUT_NPZ} and {OUT_CSV}")
print("first 20 centers (s):", [f"{c / FPS:.1f}" for _, _, c, _ in kept[:20]])
