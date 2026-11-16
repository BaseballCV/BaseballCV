"""Spike S1b: is the +6 s/hr apparent drift real? Dense OCR reads in short
windows at the very start and very end; compare local slope per window."""
import re
import sys

import cv2
import numpy as np

VIDEO = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
CROP_X0, CROP_Y1 = int(854 * 0.60), int(480 * 0.09)
TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\s*(AM|PM)?", re.I)


def parse_sec(text):
    m = TS_RE.search(text)
    if not m:
        return None
    hh, mi, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ap = (m.group(4) or "").upper()
    if ap == "PM" and hh != 12:
        hh += 12
    if ap == "AM" and hh == 12:
        hh = 0
    return hh * 3600 + mi * 60 + ss


def window_reads(cap, fps, t0, t1, step_s=5.0):
    xs, ys = [], []
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            crop = frame[0:CROP_Y1, CROP_X0:854]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if bw.mean() > 127:
                bw = 255 - bw
            result, _ = OCR(bw)
            text = " ".join(r[1] for r in result) if result else ""
            sec = parse_sec(text)
            if sec is not None:
                xs.append(t)
                ys.append(sec)
        t += step_s
    return np.array(xs), np.array(ys)


from rapidocr_onnxruntime import RapidOCR  # noqa: E402
OCR = RapidOCR()

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = n_frames / fps

for label, t0, t1 in [("start", 0, 240), ("end", duration - 240, duration)]:
    xs, ys = window_reads(cap, fps, t0, t1)
    if len(xs) >= 4:
        coef = np.polyfit(xs, ys, 1)
        resid = ys - np.polyval(coef, xs)
        print(f"{label}: n={len(xs)} slope={coef[0]:.6f} ({(coef[0]-1)*3600:+.2f}s/hr) "
              f"rmse={np.sqrt(np.mean(resid**2)):.2f}s")
        for x, y, r in list(zip(xs, ys, ys - np.polyval(coef, xs)))[:6]:
            print(f"   t={x:8.1f} wall={y:.0f} resid={r:+.2f}")
    else:
        print(f"{label}: only {len(xs)} reads")
cap.release()
