"""Annotate the source game video with the S6 v11 findings.

Draws, in place as they appear:
  - raw corridor blob detections (faint cyan dots)
  - validated home->1B transit chains (green comet trail + yellow head) with a
    banner carrying PA event, inning, speed, and db lag
  - the 1B basepath + bases, HUD (game clock from CLOCK0, video time, transit
    counter), and an intro legend panel over the first seconds

Inputs: run1/s6_transit_dets.parquet, run1/s6_transits.csv (from
spike_s6_transit_scan.py), run1/game_sync/ (manifest, pitches, calibration).
Output: libx264 mp4 via ffmpeg stdin pipe.

Usage: python scripts/annotate_s6_overlays.py [SRC_MP4] [OUT_MP4] [--sample t1,t2,...]
       --sample renders single frames at the given video times for QA instead
       of the full video.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import polars as pl

from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

SRC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
    else r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12_annotated.mp4"
SAMPLE = None
if "--sample" in sys.argv:
    SAMPLE = [float(x) for x in
              sys.argv[sys.argv.index("--sample") + 1].split(",")]

FPS = 30.0
CLOCK0 = 66764.0
DB_OFFSET = 5400.0
W, H = 854, 480

# colors (BGR)
CYAN_DIM = (200, 180, 60)
GREEN = (80, 255, 120)
YELLOW = (60, 230, 255)
WHITE = (255, 255, 255)
TEAL = (255, 200, 60)
RED = (80, 80, 230)

CHUNK_LO, CHUNK_HI = 30, 120  # bg chunk overlap (frames), as in the scan


def load_data():
    d = pl.read_parquet("run1/s6_transit_dets.parquet")
    fr = (d["video_s"].to_numpy() * FPS).round().astype(np.int64)
    order = np.argsort(fr, kind="stable")
    dets = {"fr": fr[order], "fx": d["fx"].to_numpy()[order],
            "fy": d["fy"].to_numpy()[order], "area": d["area"].to_numpy()[order]}
    tr = pl.read_csv("run1/s6_transits.csv").to_dicts()
    events = load_events()
    for k, t in enumerate(tr):
        t["idx"] = k + 1
        e = nearest_event(events, t["t_start_s"])
        t["event"] = e
    return dets, tr


def load_events():
    m = pl.read_csv("run1/game_sync/manifest.csv").sort("video_start_s")
    p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
    p = p.with_columns(
        (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1e9
         + DB_OFFSET - CLOCK0).alias("db")).sort("no")
    evs = []
    for r in m.iter_rows(named=True):
        cand = p.filter((pl.col("db") >= r["video_start_s"] - 30)
                        & (pl.col("db") <= r["video_end_s"] + 30))
        res = cand.filter(pl.col("play_result") != "")
        row = res if len(res) else cand
        if len(row):
            evs.append({"db": float(row["db"].to_list()[-1]),
                        "label": f"{r['event_text']} "
                                 f"{r['inning']}{'T' if r['is_top_inning'] else 'B'}"})
    return evs


def nearest_event(evs, transit_s):
    best, best_d = None, 1e9
    for e in evs:
        lag = transit_s - e["db"]
        if 40 <= lag <= 180 and abs(lag - 100) < best_d:
            best, best_d = e, abs(lag - 100)
    return best


def chain_indices(dets, tr):
    """Re-run the scan's greedy chain finder, keeping det row indices."""
    t = dets["fr"].astype(np.float64) / FPS
    fx, fy = dets["fx"], dets["fy"]
    n = len(t)
    chains = {k: [] for k in range(len(tr))}
    for i in range(n):
        if not (24 < fx[i] < 40):
            continue
        chain = [i]
        while True:
            j = chain[-1]
            found = None
            for k in range(j + 1, min(j + 400, n)):
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
            if 15 <= sp <= 35 and t[chain[0]] > 0:
                # match to a recorded transit by start time
                for k2, trt in enumerate(tr):
                    if abs(trt["t_start_s"] - t[chain[0]]) < 1.0:
                        if not chains[k2]:
                            chains[k2] = chain
                        break
    return chains


def base_overlay(hom):
    """Static translucent layer: 1B basepath, bases, corridor band edges."""
    layer = np.zeros((H, W, 3), np.uint8)
    pts = [hom.field_to_image((x, 0.0)) for x in np.linspace(0, 95, 60)]
    pts = [(int(round(px)), int(round(py))) for px, py in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(layer, a, b, (90, 60, 20), 2)
    for ft, size in (((0, 0), 8), ((90, 0), 9), ((63.6, 63.6), 7)):
        px, py = hom.field_to_image(ft)
        cv2.drawMarker(layer, (int(px), int(py)), (90, 60, 20), cv2.MARKER_SQUARE,
                       size + 6, 2)
    return layer


def label(img, text, pos, color, scale=0.55, thick=1, bg=(15, 15, 15), alpha=0.65,
          pad=4):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = pos
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad + 2
    x, y = max(2, x), max(th + 2 * pad + 2, y)
    x2 = min(W - 2, x2)
    sub = img[y - th - pad - 2:y2, x:x2]
    if sub.shape[0] and sub.shape[1]:
        cv2.addWeighted(np.full_like(sub, bg), alpha, sub, 1 - alpha, 0, sub)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def wall_clock(video_s):
    w = CLOCK0 + video_s
    h = int(w // 3600) % 24
    m = int(w // 60) % 60
    s = int(w) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    z = np.load("run1/game_sync/calibration.npz")
    hom = Homography.from_matrix(z["matrix"])
    print("loading dets + transits...", flush=True)
    dets, tr = load_data()
    print(f"{len(dets['fr'])} dets, {len(tr)} transits; chaining...", flush=True)
    chains = chain_indices(dets, tr)
    n_chained = sum(1 for c in chains.values() if c)
    print(f"chained {n_chained}/{len(tr)} transits", flush=True)
    # precompute image px for chain dets
    chain_px = {}
    for k, c in chains.items():
        if c:
            pts = [hom.field_to_image((float(dets["fx"][j]), float(dets["fy"][j])))
                   for j in c]
            chain_px[k] = (np.array([p[0] for p in pts], np.float32),
                           np.array([p[1] for p in pts], np.float32),
                           np.array([dets["fr"][j] for j in c], np.float64),
                           np.array([dets["area"][j] for j in c], np.int32))
    # det image px for the raw-cyan layer, precomputed once (vectorized)
    inv = np.linalg.inv(z["matrix"])
    v = inv @ np.stack([dets["fx"], dets["fy"],
                        np.ones_like(dets["fx"])], axis=0)
    det_px = (v[0] / v[2], v[1] / v[2])
    layer = base_overlay(hom)

    if SAMPLE:
        cap = cv2.VideoCapture(SRC)
        for tt in SAMPLE:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * FPS))
            ok, bgr = cap.read()
            if not ok:
                print(f"sample {tt}: read failed")
                continue
            fi = int(tt * FPS)
            img = render(bgr, fi, dets, tr, chain_px, det_px, layer, tt)
            outp = f"run1/annot_sample_{tt:.0f}.png"
            cv2.imwrite(outp, img)
            print(f"sample {tt} -> {outp}")
        cap.release()
        return

    n_src = int(cv2.VideoCapture(SRC).get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"rendering {n_src} frames -> {OUT}", flush=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(int(FPS)), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(SRC)
    fi = 0
    import time
    t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok or fi >= n_src:
            break
        img = render(bgr, fi, dets, tr, chain_px, det_px, layer, fi / FPS)
        ff.stdin.write(img.tobytes())
        fi += 1
        if fi % 30000 == 0:
            el = time.time() - t0
            print(f"{fi}/{n_src} ({fi * 100 // n_src}%) {el:.0f}s "
                  f"({fi / el:.0f} fps)", flush=True)
    cap.release()
    ff.stdin.close()
    ff.wait()
    print(f"done: {fi} frames, rc={ff.returncode} ({time.time() - t0:.0f}s)")


def render(bgr, fi, dets, tr, chain_px, det_px, layer, video_s):
    img = cv2.addWeighted(bgr, 1, layer, 0.45, 0)

    # raw corridor dets this frame: faint cyan dots
    i0 = np.searchsorted(dets["fr"], fi - 1, "left")
    i1 = np.searchsorted(dets["fr"], fi + 1, "right")
    if i1 > i0:
        xs, ys = det_px[0][i0:i1], det_px[1][i0:i1]
        for x, y in zip(xs, ys):
            cv2.circle(img, (int(x), int(y)), 1, CYAN_DIM, -1, cv2.LINE_AA)

    # active / recent transits
    active = None
    for k, t in enumerate(tr):
        if chain_px.get(k) is None:
            continue
        cpx = chain_px[k]
        cs = cpx[2] / FPS
        if t["t_start_s"] - 0.4 <= video_s <= t["t_end_s"] + 0.6:
            active = (k, t, cpx, cs)
            break
    if active is not None:
        k, t, cpx, cs = active
        m = cs <= video_s + 0.01
        xs, ys, areas = cpx[0][m], cpx[1][m], cpx[3][m]
        if len(xs):
            head = int(np.argmax(xs))
            for x, y, a in zip(xs, ys, areas):
                r = 1 + int(min(3, a / 60))
                cv2.circle(img, (int(x), int(y)), r, GREEN, -1, cv2.LINE_AA)
            hx, hy = int(xs[head]), int(ys[head])
            cv2.circle(img, (hx, hy), 4, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(img, (hx, hy), 6, YELLOW, 1, cv2.LINE_AA)

    # HUD
    done = sum(1 for t in tr
               if chain_px.get(t["idx"] - 1) is not None
               and t["t_end_s"] < video_s)
    label(img, f"GAME CLOCK {wall_clock(video_s)}   video {int(video_s // 60)}:"
               f"{int(video_s) % 60:02d}", (10, 24), WHITE, 0.55, 1)
    label(img, f"transits {done}/{len(tr)}", (W - 128, 52), TEAL, 0.55, 1)

    if active is not None:
        k, t, cpx, cs = active
        e = t["event"]
        ev = e["label"] if e else "PA unmatched"
        lag = f"{t['t_start_s'] - e['db']:+.0f}s" if e else "--"
        label(img, f"TRANSIT {t['idx']}/{len(tr)}  {ev}  "
                   f"{t['speed_ft_s']:.1f} ft/s  24->{t['x_end_ft']:.0f} ft  "
                   f"db lag {lag}", (10, H - 16), GREEN, 0.62, 2)
    elif video_s < 14:
        lines = [
            "S6 v11 full-game annotation",
            f"{len(tr)} home->1B transit chains (green), raw corridor blobs (cyan)",
            "db clock runs ~+100 s behind video: DB_OFFSET fix 5400 -> ~5505",
            "speeds 15-32 ft/s; windows anchored on transits, not db times",
        ]
        y = 60
        for ln in lines:
            label(img, ln, (10, y), WHITE, 0.55, 1, alpha=0.75)
            y += 26
    return img


if __name__ == "__main__":
    main()
