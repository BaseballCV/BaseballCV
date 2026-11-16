"""Spike S2c: save zoomed patches around predicted base positions for direct eyeballing."""
import cv2
import numpy as np

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
OUT = r"C:\Users\Carlos Marcano\AppData\Local\Temp\magprobes"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
for k in range(36):
    t = 1333 + (k - 18) * 2.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, f = cap.read()
    if ok:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
med = np.median(np.stack(frames), axis=0).astype(np.uint8)

# prediction: home=(487,277), 2B=(516,190); diamond axis perp corrected:
# screen-right of home->2B axis = 1B side
home = np.array([487.0, 277.0])
second = np.array([516.0, 190.0])
diag = second - home
L = np.hypot(*diag)
perp_r = np.array([-diag[1], diag[0]]) / L      # rotate diag by +90 (image coords, y down): right side
side = L / np.sqrt(2)
preds = {
    "1B": home + side * perp_r,
    "3B": home - side * perp_r,
}
preds["home"] = home
preds["2B"] = second
print(f"diag px {L:.1f}  scale {L/127.3:.3f} px/ft  side {side:.1f}px")
for k, v in preds.items():
    print(k, v.round(1))

# find bright blob near each prediction, save 21x21 patch at 16x
for name, p in preds.items():
    x0, y0 = int(p[0]), int(p[1])
    r = 10
    win = med[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.float32)
    loc = cv2.blur(win, (7, 7))
    resp = win - loc
    resp[resp < 5] = 0
    pos, score = None, 0.0
    if resp.max() >= 5:
        ys, xs = np.nonzero(resp > resp.max() * 0.5)
        w = resp[ys, xs]
        pos = ((xs * w).sum() / w.sum() + (x0 - r), (ys * w).sum() / w.sum() + (y0 - r))
        score = float(resp.max())
    print(f"{name}: found={None if pos is None else np.round(pos, 1)} score={score:.1f}")
    patch = med[max(0, y0 - 14):y0 + 15, max(0, x0 - 14):x0 + 15]
    big = cv2.resize(patch, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
    cv2.putText(big, name, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(f"{OUT}\\patch_{name}.png", big)
print("patches saved")
