"""S7 spike: can we see defenders as static dark blobs at pitch time, and do
infielders' first moves show up on grounders?

Stage 1 (positions): for every pitch of one inning-half, map DB pitch time ->
video frame via the persisted anchors (offset +90 min, piecewise wall->video
match restricted to anchors near the expected video time), grab the frame from
the SOURCE video, run dark-blob detection + prior assignment, count filled
slots.

Stage 2 (first moves): on groundout clips, the fielder who reacts is a MOVER,
so median-background tracking sees him: select tracks starting within 15 ft of
an infield prior that sustain >= 6 ft/s after contact.

GATE (from the plan): 6-7 positions stable on >= 80% of pitches; infielder
first move on >= 60% of grounders.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import polars as pl

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.defense import (
    defender_positions_at,
)
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import track_clip
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

RUN_DIR = "run1/game_sync"
SOURCE = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
INFIELD = ("P", "1B", "2B", "3B", "SS")
ANCHOR_WIN_S = 120.0
ANCHOR_TOL_S = 5.0


def load():
    z = np.load(os.path.join(RUN_DIR, "calibration.npz"))
    hom = Homography.from_matrix(z["matrix"])
    state = json.load(open(os.path.join(RUN_DIR, "sync_state.json"), encoding="utf-8"))
    a = np.load(os.path.join(RUN_DIR, "clock_anchors.npz"))
    return hom, state, a["video_s"], a["wall_s"]


def wall_to_video(db_wall, offset, video_s, wall_s, base_wall):
    target = db_wall + offset
    expected = target - base_wall
    cand = np.abs(video_s - expected) <= ANCHOR_WIN_S
    if not cand.any():
        return None
    i = int(np.argmin(np.abs(wall_s[cand] - target)))
    if abs(wall_s[cand][i] - target) > ANCHOR_TOL_S:
        return None
    return float(video_s[cand][i])


def stage1(hom, state, video_s, wall_s, cfg):
    import json
    pitches = pl.read_parquet(os.path.join(RUN_DIR, "pbp_cache", "game_869992.parquet.pitches"))
    base_wall = float(wall_s[0])
    half = pitches.filter((pl.col("inning") == 4) & (pl.col("is_top_inning") == True)
                          ).filter(pl.col("time").is_not_null())
    print(f"stage 1: {len(half)} pitches, top 4th; base_wall={base_wall:.0f}")
    cap = cv2.VideoCapture(SOURCE)
    counts, filled_by_pitch = [], []
    for row in half.iter_rows(named=True):
        h, m, s = (int(x) for x in row["time"].split(":")[:3])
        db_wall = h * 3600 + m * 60 + s
        v = wall_to_video(db_wall, state["chosen_offset_s"], video_s, wall_s, base_wall)
        if v is None:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, v * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos = defender_positions_at(gray, hom, cfg)
        n = sum(p is not None for p in pos.values())
        counts.append(n)
        filled_by_pitch.append((row["no"], n, {k: (round(p[0], 1), round(p[1], 1))
                                               for k, p in pos.items() if p is not None}))
    cap.release()
    counts = np.array(counts)
    if len(counts) == 0:
        print("stage 1: NO frames matched — spike inconclusive")
        return
    for n in range(0, 8):
        print(f"  {n} positions filled: {(counts == n).sum()} pitches")
    frac67 = float(((counts >= 6) & (counts <= 7)).mean())
    print(f"  6-7 positions on {frac67:.0%} of pitches "
          f"(gate >= 80%); matched {len(counts)}/{len(half)}")
    if filled_by_pitch:
        print("  example:", filled_by_pitch[len(filled_by_pitch) // 2])


def stage2(hom, cfg, max_clips=5):
    manifest = pl.read_csv(os.path.join(RUN_DIR, "manifest.csv"))
    gbs = manifest.filter(
        (pl.col("event_text") == "Groundout") & (pl.col("base_state_before") == 0)
    ).head(max_clips)
    print(f"\nstage 2: {len(gbs)} groundout clips")
    hits = 0
    for row in gbs.iter_rows(named=True):
        tracks = track_clip(row["clip_path"], cfg, logger=None)
        # contact = earliest sustained bolt among tracks (batter-runner); reuse
        # the selected-runner diagnostics instead of re-deriving
        from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
            select_runner_track)
        tid, _ = select_runner_track(tracks, hom, cfg)
        if tid is None:
            print(f"  {os.path.basename(row['clip_path'])}: no runner -> skip")
            continue
        contact_f = min(tracks.tracks[tid])
        found = None
        for t_id, frames in tracks.tracks.items():
            if t_id == tid or len(frames) < 15:
                continue
            fs = sorted(frames)
            start = hom.to_field(frames[fs[0]][:2])
            for slot in INFIELD:
                px, py, rad = cfg.defender_priors_ft[slot]
                if np.hypot(start[0] - px, start[1] - py) > 15.0:
                    continue
                # sustained departure after contact: displacement > 4 ft
                for f in fs:
                    if f <= contact_f:
                        continue
                    p = hom.to_field(frames[f][:2])
                    if np.hypot(p[0] - start[0], p[1] - start[1]) > 4.0:
                        found = (slot, t_id, f - contact_f)
                        break
                if found:
                    break
            if found:
                break
        if found:
            hits += 1
            print(f"  {os.path.basename(row['clip_path'])}: mover {found[0]} "
                  f"(track {found[1]}, {found[2] / 30:.1f} s after contact)")
        else:
            print(f"  {os.path.basename(row['clip_path'])}: NO infield mover")
    print(f"  infielder first move on {hits}/{len(gbs)} grounders (gate >= 60%)")


if __name__ == "__main__":
    import json
    cfg = MetricsConfig()
    hom, state, video_s, wall_s = load()
    stage1(hom, state, video_s, wall_s, cfg)
    stage2(hom, cfg)
