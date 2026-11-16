"""Spike S1c: full-file scan of the burned-in clock to map video-time -> wall-time
piecewise (frame-drop gaps). Reads every ~60 s of video, reports segments where
local slope deviates from 1.0 (recording gaps) and total accumulated offset."""
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


from rapidocr_onnxruntime import RapidOCR  # noqa: E402
OCR = RapidOCR()

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = n_frames / fps
print(f"fps={fps} n_frames={n_frames:.0f} duration={duration:.1f}s")

xs, ys = [], []
t = 0.0
STEP = 60.0
fails = 0
while t < duration:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    sec = None
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
    if sec is None:
        # retry at +1s before giving up on this anchor
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + 1) * fps))
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
    if sec is None:
        fails += 1
    else:
        xs.append(t)
        ys.append(sec)
    t += STEP
cap.release()

xs, ys = np.array(xs), np.array(ys)
print(f"\nreads ok: {len(xs)}/{len(xs)+fails} ({100*len(xs)/(len(xs)+fails):.0f}%)")

# local slope between consecutive anchors
print("\nsegments with |slope-1| > 0.02 (accumulating gaps):")
gap_total = 0.0
prev_s = None
for i in range(1, len(xs)):
    dv = xs[i] - xs[i - 1]
    dw = ys[i] - ys[i - 1]
    slope = dw / dv
    if abs(slope - 1.0) > 0.02:
        gap_total += dw - dv
        print(f"  video {xs[i-1]:8.0f}->{xs[i]:8.0f}s  wall delta {dw:6.0f}s  "
              f"slope {slope:.4f}  gap {dw-dv:+.0f}s")
    prev_s = ys[i]
# discontinuities (wall goes backwards or jumps)
print("\nwall-clock discontinuities (|dw-dv| > 120s at otherwise slope~1):")
for i in range(1, len(xs)):
    dv, dw = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
    if abs(dw - dv) > 120 and abs(dw / dv - 1) <= 0.02:
        print(f"  video {xs[i-1]:8.0f}->{xs[i]:8.0f}s  wall jump {dw-dv:+.0f}s")
print(f"\ntotal slow-motion gap time: {gap_total:+.0f}s")
print(f"wall at video 0: {ys[0]:.0f}s-of-day; at end: {ys[-1]:.0f}; video span {xs[-1]-xs[0]:.0f}s, wall span {ys[-1]-ys[0]:.0f}s")

np.savez(r"C:\Users\Carlos Marcano\AppData\Local\Temp\magprobes\clock_anchors.npz", xs=xs, ys=ys)
print("anchors saved")
