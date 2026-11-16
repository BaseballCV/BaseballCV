"""Statistical offset validation: frac3 energy at DB pitch times under +90min vs decoys."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import polars as pl

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
X0, Y0, X1, Y1 = 0.30, 0.30, 0.75, 0.85
WALL0 = 18 * 3600 + 32 * 60 + 44  # camera wall at video t=0 (spike-verified)

pitches = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
times = (pitches.filter(pl.col("time").is_not_null())["time"].to_list())
walls = []
for hhmmss in times:
    h, m, s = (int(p) for p in hhmmss.split(":")[:3])
    walls.append(h * 3600 + m * 60 + s)
walls = np.array(sorted(set(walls)))
# subsample ~24 spread over the game
sel = np.linspace(0, len(walls) - 1, 24).astype(int)
walls = walls[sel]

def frac_signal(t0, dur):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps / 5.0)))
    prev, tt, sig = None, [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t0 * fps)))
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
                roi = frame[int(h*Y0):int(h*Y1), int(w*X0):int(w*X1)]
                g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, None, fx=0.5, fy=0.5)
                if prev is not None:
                    sig.append(float((np.abs(g.astype(np.int16) - prev) > 3).mean()))
                    tt.append(t0 + idx / fps)
                prev = g
        idx += 1
    cap.release()
    return np.array(tt), np.array(sig)

def z_at(tt, sig, t_pitch):
    m = (tt >= t_pitch - 30) & (tt <= t_pitch + 30)
    if m.sum() < 20:
        return None
    base, noise = np.median(sig[m]), np.std(sig[m]) + 1e-6
    hit = (tt >= t_pitch - 2) & (tt <= t_pitch + 2)
    return float((sig[hit].mean() - base) / noise)

results = {}
for off_s, name in ((5400.0, "+90min"), (0.0, "+0min"), (3600.0, "+60min"), (7200.0, "+120min")):
    zs = []
    for w in walls:
        t_pitch = w + off_s - WALL0
        if t_pitch < 30:
            continue
        tt, sig = frac_signal(t_pitch - 30, 60)
        z = z_at(tt, sig, t_pitch)
        if z is not None:
            zs.append(z)
    zs = np.array(zs)
    results[name] = zs
    print(f"{name:8s} n={len(zs):2d}  mean z={np.mean(zs):+.2f}  median z={np.median(zs):+.2f}  "
          f"frac z>1: {np.mean(zs > 1):.0%}  frac z>2: {np.mean(zs > 2):.0%}")
