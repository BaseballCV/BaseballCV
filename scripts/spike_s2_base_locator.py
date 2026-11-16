"""Spike S2: locate bases on the cenital view via median-of-frames + bright-square
detection. Produces an annotated overlay to eyeball, plus candidate list."""
import sys

import cv2
import numpy as np

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
CENTER_T = float(sys.argv[1]) if len(sys.argv) > 1 else 1333.0  # video seconds
N = 36
STEP_S = 2.0

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# median over N frames sampled around CENTER_T (suppresses moving players)
frames = []
for k in range(N):
    t = CENTER_T + (k - N // 2) * STEP_S
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, f = cap.read()
    if ok:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
med = np.median(np.stack(frames), axis=0).astype(np.uint8)
cv2.imwrite(r"C:\Users\Carlos Marcano\AppData\Local\Temp\magprobes\median_frame.png", med)

# bright mask: white-ish, reasonably bright, low saturation in color? (median is gray)
# adaptive: top percentile within the infield ROI
roi = np.zeros_like(med, bool)
roi[140:420, 280:720] = True   # generous infield box from the 4x crop inspection
vals = med[roi]
thr = np.percentile(vals, 99.0)
mask = np.zeros_like(med, np.uint8)
mask[(med >= max(thr, 170)) & roi] = 255

# connected components -> candidates by size/shape
n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
cands = []
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 2 or area > 400:
        continue
    if w > 60 or h > 60:
        continue
    ar = max(w, h) / max(1, min(w, h))
    if ar > 3.0:
        continue
    cx, cy = cents[i]
    cands.append((cx, cy, w, h, area))

print(f"median thr={thr:.0f}, {len(cands)} candidates:")
for c in sorted(cands, key=lambda c: -c[4])[:30]:
    print(f"  cx={c[0]:7.1f} cy={c[1]:7.1f} w={c[2]:3d} h={c[3]:3d} area={c[4]}")

# annotated overlay (4x for eyeballing)
vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
vis = cv2.resize(vis, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
for j, c in enumerate(sorted(cands, key=lambda c: -c[4])[:30]):
    cx, cy = c[0] * 4, c[1] * 4
    cv2.circle(vis, (int(cx), int(cy)), 14, (0, 0, 255), 2)
    cv2.putText(vis, str(j), (int(cx) + 16, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.imwrite(r"C:\Users\Carlos Marcano\AppData\Local\Temp\magprobes\base_candidates.png", vis)
print("overlay saved")
