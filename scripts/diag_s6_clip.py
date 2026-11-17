"""One-clip S6 diagnostic: what does detection + association actually see?

Renders annotated frames (ROI polygon, per-frame detections, the longest
tracks) to run1/diag/ and dumps field-space trajectories of the longest
home-band tracks around the bolt. Eyeball material.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    field_roi_mask, select_runner_track, track_field_xy, track_clip,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

CLIP = sys.argv[1] if len(sys.argv) > 1 else r"run1\game_sync\play_0003_inn1T.mp4"
OUT = "run1/diag"
os.makedirs(OUT, exist_ok=True)

z = np.load("run1/game_sync/calibration.npz")
hom = Homography.from_matrix(z["matrix"])
cfg = MetricsConfig()

tracks = track_clip(CLIP, cfg, homography=hom, logger=None)
print(f"{len(tracks.tracks)} tracks, {tracks.n_frames} frames")

# longest 12 tracks
top = sorted(tracks.tracks.items(), key=lambda kv: -len(kv[1]))[:12]
for tid, frames in top:
    fs = sorted(frames)
    start = hom.to_field(frames[fs[0]][:2])
    end = hom.to_field(frames[fs[-1]][:2])
    print(f"track {tid}: {len(frames)} f, [{fs[0]}..{fs[-1]}] "
          f"start_ft=({start[0]:.1f},{start[1]:.1f}) end_ft=({end[0]:.1f},{end[1]:.1f})")

# annotate sampled frames
cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
colors = [(int(c) for c in cv2.applyColorMap(np.array([[i * 23 % 255]], np.uint8),
                                            cv2.COLORMAP_HSV)[0, 0]) for i in range(20)]
fi = 0
palette = [(60, 60, 240), (60, 200, 240), (60, 240, 60), (240, 200, 60),
           (240, 60, 200), (240, 240, 60), (120, 60, 240), (60, 240, 200)]
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    if fi % 25 == 0:
        vis = bgr.copy()
        cv2.polylines(vis, [np.array([[int(p[0]), int(p[1])]
                                      for p in (hom.field_to_image(c)
                                                for c in ((*cfg.field_roi_ft[:2],),
                                                          (cfg.field_roi_ft[2], cfg.field_roi_ft[1]),
                                                          (cfg.field_roi_ft[2], cfg.field_roi_ft[3]),
                                                          (cfg.field_roi_ft[0], cfg.field_roi_ft[3])))],
                                      np.int32)], True, (255, 255, 255), 1)
        for k, (tid, frames) in enumerate(top):
            col = palette[k % len(palette)]
            pts = [frames[f] for f in sorted(frames) if fi - 40 <= f <= fi]
            for p in pts:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, col, -1)
            if pts:
                cv2.putText(vis, str(tid), (int(pts[-1][0]) + 5, int(pts[-1][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.imwrite(os.path.join(OUT, f"clip_{fi:05d}.png"), vis)
    fi += 1
cap.release()
print(f"annotated frames -> {OUT}")

# field-space dump around the presumed bolt for the top home-band track
for tid, frames in top[:4]:
    field = track_field_xy(frames, hom)
    fs = sorted(field)
    print(f"\ntrack {tid} field trajectory (every 5 frames):")
    for f in fs[::5]:
        p = field[f]
        print(f"  f={f:4d} ({p[0]:6.1f},{p[1]:6.1f})")
