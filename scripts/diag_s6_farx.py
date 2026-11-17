"""When does ANYTHING get farther than 30 ft down the 1B line in this clip?

Whole-clip scan: per frame, corridor dets (0<x<100, |y|<20), drop the static
94-97 ft pair, print frames whose max remaining x exceeds 30 ft.
"""
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

for f, g in enumerate(grays):
    dets = extract_blobs(g, bg, cfg, roi_mask=roi)
    xs = []
    for (x, y, a) in dets:
        fx, fy = hom.to_field((x, y))
        if 0 < fx < 90 and abs(fy) < 20:
            xs.append(fx)
    if xs and max(xs) > 30:
        print(f"{f:5d} | max_x={max(xs):5.1f}  xs={sorted(round(v) for v in xs)}")
