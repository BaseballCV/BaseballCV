"""Test sharper activity signal: fraction of ROI pixels with |diff| > thr."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
cfg_x0, cfg_y0, cfg_x1, cfg_y1 = 0.30, 0.30, 0.75, 0.85
FPS_KEEP = 5.0

def window(t0, dur=120.0):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(fps / FPS_KEEP))
    prev = None
    frac6, frac3, mean_e, times = [], [], [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    n = int(dur * fps)
    idx = 0
    while idx < n:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                roi = frame[int(h*cfg_y0):int(h*cfg_y1), int(w*cfg_x0):int(w*cfg_x1)]
                g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, None, fx=0.5, fy=0.5)
                if prev is not None:
                    d = np.abs(g.astype(np.int16) - prev)
                    frac6.append(float((d > 6).mean()))
                    frac3.append(float((d > 3).mean()))
                    mean_e.append(float(d.mean()))
                    times.append(t0 + idx / fps)
                prev = g
        idx += 1
    cap.release()
    return (np.array(times), np.array(frac6), np.array(frac3), np.array(mean_e))

def scan_onsets(t, sig, k_s=20.0, mult=3.0, floor_add=0.004, sustain=2):
    """Rolling-median floor over k_s seconds (bursts don't lift it), MAD noise."""
    k = max(3, int(k_s * 5.0))
    pad = np.pad(sig, (k // 2, k // 2), mode="edge")
    # rolling median via strided windows
    med = np.array([np.median(pad[i:i + k]) for i in range(len(sig))])
    resid = sig - med
    noise = max(float(np.median(np.abs(resid)) * 1.4826) * mult, floor_add)
    above = sig > med + noise
    rows = []
    i = 1
    while i < len(above) - sustain:
        if above[i] and not above[i - 1] and above[i:i + sustain].all():
            rows.append(t[i])
            i += sustain
        i += 1
    return rows, med, noise

# known pitch video times (from spot checks): HR 2732, HR 5896, first pitch ~9
CASES = (("hr_muzziotti", 2700, [2732]), ("hr_odor", 5870, [5896]),
         ("game_start", 0, [9]), ("between_inn", 6100, []))
for label, t0, known in CASES:
    t, f6, f3, me = window(t0)
    for name, sig in (("frac6", f6), ("frac3", f3)):
        rows, med, noise = scan_onsets(t, sig)
        hits = [kt for kt in known if any(abs(r - kt) < 3 for r in rows)]
        print(f"{label:14s} {name:5s} n_samp={len(sig):4d} med={np.median(sig):.4f} "
              f"p99={np.percentile(sig, 99):.4f} noise_thr={noise:.4f} onsets={len(rows):3d} "
              f"known_hits={len(hits)}/{len(known)} "
              + (f"first_onsets={[round(r,1) for r in rows[:6]]}" if rows else ""))
    print()
