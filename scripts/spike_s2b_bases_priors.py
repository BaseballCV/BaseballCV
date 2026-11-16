"""Spike S2b: precision base localization using diamond-geometry priors on a
quiescent median frame. Search bright blobs in +-10px windows around predicted
base positions (derived from home + 2B), verify square fit, save overlay."""
import cv2
import numpy as np

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
OUT = r"C:\Users\Carlos Marcano\AppData\Local\Temp\magprobes"

# --- median over a quiescent window (72 s around t=1333) ---
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

# priors measured on the 8x zoom
home = np.array([487.0, 277.0])
second = np.array([516.0, 190.0])
diag = second - home
L = np.hypot(*diag)                     # px per 127.3 ft
perp = np.array([-diag[1], diag[0]]) / L   # unit perpendicular (points toward 3B? sign to check)
# 1B should be right of home->2B axis: with home below 2B, right-hand perpendicular:
perp_r = np.array([diag[1], -diag[0]]) / L
side_px = L / np.sqrt(2)                # 90 ft side in px

first_pred = home + side_px * perp_r * 1.0
third_pred = home - side_px * perp_r * 1.0
print(f"priors: home={home} 2B={second} scale={L/127.3:.4f} px/ft side={side_px:.1f}px")
print(f"predicted 1B={first_pred.round(1)} 3B={third_pred.round(1)}")


def find_bright_blob(center, radius=10):
    """Brightest small blob centroid in a window; weighted by brightness over local mean."""
    x0, y0 = int(center[0]), int(center[1])
    win = med[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1].astype(np.float32)
    loc = cv2.blur(win, (7, 7))
    resp = win - loc                       # local contrast
    resp[resp < 6] = 0                     # bright-on-surroundings only
    if resp.max() < 6:
        return None, 0.0
    ys, xs = np.nonzero(resp > resp.max() * 0.5)
    w = resp[ys, xs]
    cx = (xs * w).sum() / w.sum() + (x0 - radius)
    cy = (ys * w).sum() / w.sum() + (y0 - radius)
    return (cx, cy), float(resp.max())


found = {}
for name, pred in [("home", home), ("first", first_pred), ("second", second), ("third", third_pred)]:
    pos, score = find_bright_blob(pred)
    found[name] = pos
    print(f"{name:7s} pred={pred.round(1)} -> found={None if pos is None else np.round(pos,1)} score={score:.1f}")

# square-fit check on the found points (if all 4)
pts = [found[k] for k in ("home", "first", "second", "third")]
if all(p is not None for p in pts):
    pts = np.array(pts)
    names = ["home", "1B", "2B", "3B"]
    print("\nsquare check (ft, scale from home-2B diagonal):")
    for i in range(4):
        for j in range(i + 1, 4):
            d_px = np.hypot(*(pts[i] - pts[j]))
            expect = 127.3 if {i, j} in ({0, 2}, {1, 3}) else 90.0
            print(f"  {names[i]}-{names[j]}: {d_px/(L/127.3):6.1f} ft (expect {expect})")

# overlay at 8x
vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
vis = cv2.resize(vis, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
for name, pred, col in [("home", home, (0, 255, 255)), ("first", first_pred, (0, 165, 255)),
                        ("second", second, (255, 0, 255)), ("third", third_pred, (255, 0, 0))]:
    p = found[name]
    px = (p if p is not None else pred) * 8
    cv2.circle(vis, (int(px[0]), int(px[1])), 22, (0, 0, 255), 2)
    cv2.putText(vis, name, (int(px[0]) + 24, int(px[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
cv2.imwrite(f"{OUT}\\base_priors_overlay.png", vis)
print("\noverlay saved")
