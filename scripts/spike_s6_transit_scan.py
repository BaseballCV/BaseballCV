"""S6 v11: full-game transit scan on the SOURCE video (db times not trusted).

Round finding (handoff journal, S6 v11): db-derived video_s is off by a
CONSTANT ~+100 s all game (db clock ~5505 s behind the burned clock; the
5400 offset guess is ~105 s short), so db-anchored clip windows miss their
plays by that fixed margin (play_0042's single transits at video 6506.7, 53 s
past the re-cut clip). Detection itself works fine game-long.

This script scans the whole source video ONCE with the validated recipe
(median bg per chunk, absdiff > 14, morph open 3x3 = package default, area
5-300), keeps corridor dets (5 < fx < 98 ft, |fy| < 12), then chains them into
transit candidates (24 -> 80+ ft at 15-35 ft/s, |dy| < 6). Outputs:

  run1/s6_transit_dets.parquet   raw corridor dets (video_s, fx, fy, area)
  run1/s6_transits.csv           transit chains + per-transit metrics

Usage: python scripts/spike_s6_transit_scan.py [SRC_MP4] [LO_S] [HI_S]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import polars as pl

from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
LO_S = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
HI_S = float(sys.argv[3]) if len(sys.argv) > 3 else None
OUT_PARQUET = "run1/s6_transit_dets.parquet"
OUT_CSV = "run1/s6_transits.csv"
FPS = 30.0
STEP = 2                    # scan every 2nd frame
CHUNK = 600                 # frames per bg chunk
BG_STRIDE = 10              # bg sample every 10th frame in chunk
DIFF_T = 14
AREA_LO, AREA_HI = 5, 300
K3 = np.ones((3, 3), np.uint8)

CHUNK_LO, CHUNK_HI = 30, 120   # bg chunk overlap (frames)


def main():
    z = np.load("run1/game_sync/calibration.npz")
    hom = Homography.from_matrix(z["matrix"])
    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - FPS) < 0.1, f"source fps {fps}"
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi_f = n_src if HI_S is None else min(n_src, int(HI_S * FPS))
    lo_f = int(LO_S * FPS)
    rows = []
    t0 = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo_f)
    chunk_start = lo_f
    buf = []
    done = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (lo_f + done) >= hi_f:
            break
        buf.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        done += 1
        if len(buf) == CHUNK:
            _process_chunk(buf, chunk_start, hom, rows)
            buf = buf[CHUNK - CHUNK_LO:]
            chunk_start += CHUNK - CHUNK_LO
            el = time.time() - t0
            fr = chunk_start - lo_f
            print(f"{chunk_start / FPS:.0f}s  ({fr / (hi_f - lo_f) * 100:.0f}%) "
                  f"{el:.0f}s elapsed, {len(rows)} dets", flush=True)
    if len(buf) >= BG_STRIDE * 2:
        _process_chunk(buf, chunk_start, hom, rows)
    cap.release()
    d = pl.DataFrame(rows, schema={"video_s": pl.Float64, "fx": pl.Float32,
                                   "fy": pl.Float32, "area": pl.Int32},
                     orient="row")
    d.write_parquet(OUT_PARQUET)
    print(f"{len(d)} corridor dets -> {OUT_PARQUET} ({time.time() - t0:.0f}s)")

    transits = find_transits(d)
    transits.write_csv(OUT_CSV)
    print(transits)


def _process_chunk(buf, start_f, hom, rows):
    idx = np.arange(0, len(buf), BG_STRIDE)
    bg = np.median(np.stack([buf[i] for i in idx]), axis=0).astype(np.uint8)
    for i in range(0, len(buf), STEP):
        g = buf[i]
        f = start_f + i
        d = cv2.absdiff(g, bg)
        m = (d > DIFF_T).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K3)
        nb, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for k in range(1, nb):
            a = stats[k, cv2.CC_STAT_AREA]
            if AREA_LO <= a <= AREA_HI:
                fx, fy = hom.to_field((float(cent[k][0]), float(cent[k][1])))
                if 5 < fx < 98 and -12 < fy < 12:
                    rows.append((f / FPS, fx, fy, a))


def find_transits(d: pl.DataFrame) -> pl.DataFrame:
    t = d["video_s"].to_numpy()
    fx = d["fx"].to_numpy()
    fy = d["fy"].to_numpy()
    n = len(d)
    out = []
    for i in range(n):
        if not (24 < fx[i] < 40):
            continue
        chain = [i]
        while True:
            j = chain[-1]
            found = None
            for k in range(j + 1, min(j + 400, n)):   # <= 12 s ahead in det rows
                dt = t[k] - t[j]
                if dt > 2.5:
                    break
                if dt <= 0.05:
                    continue
                sp = (fx[k] - fx[j]) / dt
                if 8 <= sp <= 40 and abs(fy[k] - fy[j]) < 6:
                    found = k
                    break
            if found is None:
                break
            chain.append(found)
            if fx[found] > 80:
                break
        if len(chain) >= 8 and fx[chain[-1]] > 80:
            dur = t[chain[-1]] - t[chain[0]]
            sp = (fx[chain[-1]] - fx[chain[0]]) / dur
            if 15 <= sp <= 35:
                out.append((t[chain[0]], t[chain[-1]], fx[chain[0]],
                            fx[chain[-1]], sp, len(chain)))
    kept = []
    for c in sorted(out):
        if not kept or c[0] - kept[-1][0] > 10:
            kept.append(c)
    return pl.DataFrame(kept, schema={
        "t_start_s": pl.Float64, "t_end_s": pl.Float64,
        "x_start_ft": pl.Float64, "x_end_ft": pl.Float64,
        "speed_ft_s": pl.Float64, "n_pts": pl.Int32}, orient="row")


if __name__ == "__main__":
    main()
