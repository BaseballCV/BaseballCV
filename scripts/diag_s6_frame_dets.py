"""All raw detections (image px + field ft) for single frames, plus a zoomed
crop of the 1B-line region saved next to the log — to see whether the visible
runner is detected, and if not, why (contrast? area?)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    extract_blobs, field_roi_mask, median_background,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

CLIP = sys.argv[1] if len(sys.argv) > 1 else r"run1\game_sync\play_0005_inn1B.mp4"
FRAMES = [int(v) for v in sys.argv[2:]] or [700, 720, 740, 760, 780, 800]

z = np.load("run1/game_sync/calibration.npz")
hom = Homography.from_matrix(z["matrix"])
cfg = MetricsConfig()

cap = cv2.VideoCapture(CLIP)
grays = []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
cap.release()
bg = median_background(grays[::cfg.bg_sample_stride], stride=1)
roi = field_roi_mask(bg.shape, hom, cfg)
print("ROI nonzero px:", int(roi.sum() / 255))

diffs = []
for f in FRAMES:
    dets = extract_blobs(grays[f], bg, cfg, roi_mask=roi)
    print(f"\nframe {f}: {len(dets)} dets")
    for (x, y, a) in sorted(dets, key=lambda d: -d[2]):
        fx, fy = hom.to_field((x, y))
        print(f"  px({x:5.0f},{y:5.0f}) a{a:4d} -> field({fx:6.1f},{fy:6.1f})")
    # what does the raw diff look like along the visible 1B line?
    diff = cv2.absdiff(grays[f], bg)
    diffs.append((f, diff))

# zoom crops of the basepath region for visual check
out = "run1/diag3"
os.makedirs(out, exist_ok=True)
for f, diff in diffs:
    vis = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(out, f"diff_{f:05d}.png"), vis)
    crop = grays[f][180:320, 380:720]
    cv2.imwrite(os.path.join(out, f"gray_{f:05d}.png"), crop)
print(f"\ncrops -> {out}")
