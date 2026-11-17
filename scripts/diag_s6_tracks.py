"""Why do most clips have ZERO onset candidates? Dump every track's best
smoothed toward-1B velocity + corridor occupancy for one clip, and render
annotated frames."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    field_roi_mask, track_clip,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.events import smooth, velocity
from baseballcv.functions.utils.baseball_utils.game_metrics.metrics import to_arrays
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

CLIP = sys.argv[1] if len(sys.argv) > 1 else r"run1\game_sync\play_0005_inn1B.mp4"
OUT = "run1/diag2"
os.makedirs(OUT, exist_ok=True)

z = np.load("run1/game_sync/calibration.npz")
hom = Homography.from_matrix(z["matrix"])
cfg = MetricsConfig()

tracks = track_clip(CLIP, cfg, homography=hom, logger=None)
print(f"{len(tracks.tracks)} tracks, {tracks.n_frames} frames")

rows = []
for tid, frames in tracks.tracks.items():
    if len(frames) < 10:
        continue
    fs = sorted(frames)
    pts = np.array([hom.to_field(frames[f][:2]) for f in fs])
    xs, ys = to_arrays({f: (p[0], p[1]) for f, p in zip(fs, pts)},
                       tracks.n_frames, cfg)
    vx = velocity(smooth(xs, 5), 30.0)
    fin = np.isfinite(vx)
    best = float(np.nanmax(vx)) if fin.any() else float("-inf")
    i_best = int(np.nanargmax(vx)) if fin.any() else -1
    # corridor occupancy: fraction of frames with 0<x<100, |y|<25
    corr = float(((pts[:, 0] > 0) & (pts[:, 0] < 100) & (np.abs(pts[:, 1]) < 25)).mean())
    rows.append((best, tid, len(frames), fs[0], fs[-1],
                 float(np.hypot(xs[fs[0]], ys[fs[0]])) if np.isfinite(xs[fs[0]]) else -1,
                 corr))
rows.sort(reverse=True)
print("best_vx | tid | frames | span | start_d_home | corridor_frac")
for best, tid, n, a, b, sdh, corr in rows[:20]:
    print(f"  {best:6.1f} | {tid:5d} | {n:5d} | [{a}..{b}] | {sdh:6.1f} | {corr:.2f}")

# annotated frames every 25 f: ROI + all track points in the last 40 frames
cap = cv2.VideoCapture(CLIP)
palette = [(60, 60, 240), (60, 200, 240), (60, 240, 60), (240, 200, 60),
           (240, 60, 200), (240, 240, 60), (120, 60, 240), (60, 240, 200)]
top = sorted(tracks.tracks.items(), key=lambda kv: -len(kv[1]))[:8]
fi = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    if fi % 25 == 0:
        vis = bgr.copy()
        for k, (tid, frames) in enumerate(top):
            col = palette[k % len(palette)]
            for f, p in frames.items():
                if fi - 40 <= f <= fi:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 4, col, -1)
            fs = [f for f in sorted(frames) if fi - 40 <= f <= fi]
            if fs:
                p = frames[fs[-1]]
                cv2.putText(vis, str(tid), (int(p[0]) + 5, int(p[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.imwrite(os.path.join(OUT, f"{os.path.basename(CLIP)}_{fi:05d}.png"), vis)
    fi += 1
cap.release()
print(f"annotated -> {OUT}")
