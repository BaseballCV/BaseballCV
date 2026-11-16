"""Step-0 spike S1: burned-in timestamp OCR on the Magallanes cenital video.

Gate: >=80% of sampled frames parse to a valid timestamp, rapidocr vs
pixel-threshold binarization agree on the seconds digits, linear fit RMSE < 2 s.

Run:  python scripts/spike_s1_timestamp_ocr.py <video.mp4> [n_frames]
"""
import re
import sys
from datetime import datetime

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# 854x480: timestamp occupies roughly x >= 0.62*W, y <= 0.07*H (top-right)
CROP_X0, CROP_Y1 = int(854 * 0.60), int(480 * 0.09)


def grab_frames(path, times_s):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        out.append((t, frame if ok else None))
    cap.release()
    return out


def sample_times(duration_s, n):
    # spread over the game, denser at both ends
    inner = np.linspace(60, duration_s - 60, n - 8)
    dense = np.concatenate([np.linspace(1, 60, 4), np.linspace(duration_s - 60, duration_s - 1, 4)])
    return np.sort(np.concatenate([inner, dense]))


def preprocess(crop):
    """Grayscale, 4x upscale, Otsu binarize (white text on varied background)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure text is white-on-black
    if bw.mean() > 127:
        bw = 255 - bw
    return big, bw


TS_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})\s*(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)?", re.I)


def parse_text(text):
    m = TS_RE.search(text.replace("\n", " "))
    if not m:
        return None
    mm, dd, yy, hh, mi, ss, ampm = m.groups()
    hh, mi, ss = int(hh), int(mi), int(ss)
    if ampm:
        ampm = ampm.upper()
        if hh == 12:
            hh = 0 if ampm == "AM" else 12
        elif ampm == "PM":
            hh += 12
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31 and hh < 24 and mi < 60 and ss < 60):
        return None
    return (int(yy), int(mm), int(dd), hh, mi, ss)


def main():
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    cap = cv2.VideoCapture(VIDEO)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"duration: {duration:.0f}s")

    times = sample_times(duration, N)
    rows = []
    for t, frame in grab_frames(VIDEO, times):
        if frame is None:
            rows.append((t, None, None, None))
            continue
        crop = frame[0:CROP_Y1, CROP_X0:854]
        big, bw = preprocess(crop)
        result, _ = ocr(bw)
        text_r = " ".join(r[1] for r in result) if result else ""
        parsed_r = parse_text(text_r)
        rows.append((t, text_r.strip(), parsed_r, None))

    ok = [r for r in rows if r[2] is not None]
    print(f"\nparsed {len(ok)}/{len(rows)} frames ({100*len(ok)/len(rows):.0f}%)")
    for t, text, parsed, _ in rows[:8]:
        print(f"  t={t:8.1f}s  text={text!r:45s}  parsed={parsed}")

    if len(ok) < 0.8 * len(rows):
        print("GATE FAIL: parse rate < 80%")
        return 1

    # linear fit: video seconds -> wall seconds-of-day
    xs, ys = [], []
    for t, _, parsed, _ in ok:
        yy, mo, dd, hh, mi, ss = parsed
        wall = hh * 3600 + mi * 60 + ss
        xs.append(t)
        ys.append(wall)
    xs, ys = np.array(xs), np.array(ys)
    # robust: two-pass LSQ with residual rejection
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ coef
    keep = np.abs(resid - np.median(resid)) < 3 * 1.4826 * np.median(np.abs(resid - np.median(resid)) + 1e-9)
    coef, *_ = np.linalg.lstsq(A[keep], ys[keep], rcond=None)
    resid = (ys - A @ coef)[keep]
    slope, intercept = coef
    rmse = float(np.sqrt(np.mean(resid**2)))
    drift_s_per_hr = (slope - 1.0) * 3600

    w0 = datetime.now()
    print(f"\nfit: slope={slope:.8f} intercept={intercept:.1f}s rmse={rmse:.2f}s "
          f"n_used={keep.sum()}/{len(ok)} drift={drift_s_per_hr:+.2f}s/hr")
    print(f"video t=0      -> {(intercept/3600):.3f}h of day")
    print(f"video t={duration:.0f} -> {((slope*duration+intercept)/3600):.3f}h of day")
    print(f"gate: rmse {'PASS' if rmse < 2.0 else 'FAIL'} (< 2.0), "
          f"drift {'PASS' if abs(drift_s_per_hr) < 5 else 'WARN'} (< 5 s/hr)")
    return 0 if rmse < 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
