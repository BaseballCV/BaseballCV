"""Are the runner's raw detections present at all during the run?

For frames in a window, print every raw detection mapped to field feet,
restricted to the home->1B corridor (0<x<100, |y|<20). If the runner shows up
raw, association is the culprit; if not, detection is.
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
F0 = int(sys.argv[2]) if len(sys.argv) > 2 else 650
F1 = int(sys.argv[3]) if len(sys.argv) > 3 else 860

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

print("frame | raw dets in corridor (field ft: x,y,area)")
for f in range(F0, min(F1, len(grays))):
    dets = extract_blobs(grays[f], bg, cfg, roi_mask=roi)
    cor = []
    for (x, y, a) in dets:
        fx, fy = hom.to_field((x, y))
        if 0 < fx < 100 and abs(fy) < 20:
            cor.append(f"{fx:.0f},{fy:.0f},a{a}")
    if cor:
        print(f"{f:5d} | {'  '.join(cor)}")
