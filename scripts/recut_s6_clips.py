"""Re-cut S6 clips around the batted pitch with a generous tail.

Lesson from play_0033 (journal 2026-11-16): the manifest clip window can end
seconds after the batted pitch — the runner's whole transit is cut off. This
re-cuts clips from the source video centered on the PLAY: from 15 s before the
result pitch's db wall to 45 s after (60 s), re-encoded (frame-exact, no
keyframe slop). The runner pipeline OCR-anchors each clip anyway, so the
+-10-15 s db accuracy only shifts the window, not the measurement.

Usage: python scripts/recut_s6_clips.py [EVENT_TEXT]  (default: Single)
Writes run1/s6v10/*.mp4 and run1/s6v10/recut.csv.
"""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

EVENT = sys.argv[1] if len(sys.argv) > 1 else "Single"
SRC = r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4"
OUT_DIR = "run1/s6v10"
CLOCK0 = 66764.0        # wall_s of video_s 0 (clock_anchors.npz)
DB_OFFSET = 5400.0
PRE_S, POST_S = 15.0, 45.0

os.makedirs(OUT_DIR, exist_ok=True)
m = pl.read_csv("run1/game_sync/manifest.csv")
p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
p = p.with_columns(
    (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1_000_000_000
     + DB_OFFSET - CLOCK0).alias("video_s")
).sort("no")

clips = m.filter(pl.col("event_text") == EVENT).unique(subset=["video_start_s"])
rows = []
for row in clips.iter_rows(named=True):
    bname = row["batter_name"]
    # result pitch of this PA: last pitch for this batter before the event's
    # manifest window end (batter names share mojibake across files)
    cand = p.filter(
        (pl.col("batter_name") == bname)
        & (pl.col("video_s") >= row["video_start_s"] - 90)
        & (pl.col("video_s") <= row["video_end_s"] + 120)
    )
    if len(cand) == 0:
        print(f"{row['clip_path']}: no PA pitch found ({bname!r})")
        continue
    res = cand.filter(pl.col("play_result") != "")
    play_vs = (res if len(res) else cand)["video_s"][-1]
    ss = max(0.0, play_vs - PRE_S)
    t = PRE_S + POST_S
    name = os.path.basename(row["clip_path"])
    out = os.path.join(OUT_DIR, name)
    if not os.path.exists(out):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}",
               "-i", SRC, "-t", f"{t:.2f}", "-c:v", "libx264", "-preset",
               "veryfast", "-crf", "23", "-an", out]
        subprocess.run(cmd, check=True)
    rows.append((name, row["event_text"], f"{play_vs:.1f}", f"{ss:.1f}",
                 f"{ss + t:.1f}", f"{(play_vs - ss):.1f}"))
    print(f"{name}: play at video {play_vs:.1f} -> clip {ss:.1f}..{ss + t:.1f} "
          f"(play at +{play_vs - ss:.1f}s)")

with open(os.path.join(OUT_DIR, "recut.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "event", "play_video_s", "clip_start", "clip_end",
                "play_offset_s"])
    w.writerows(rows)
print(f"wrote {len(rows)} clips -> {OUT_DIR}")
