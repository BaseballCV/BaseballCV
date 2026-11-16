"""End-to-end sync run on the real Magallanes feed (game 869992)."""
import json
import os
import sys

# running a script puts scripts/ on sys.path instead of the repo root, which
# would silently import an older installed baseballcv from site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseballcv.functions import BaseballTools

res = BaseballTools(device="cpu").sync_game_video(
    video_path=r"C:\Users\Carlos Marcano\Downloads\1_juego27_2026-11-12.mp4",
    game_pk=869992,
    cut_clips=True,
    results_dir="run1/game_sync",
)

out = {k: v for k, v in res.items() if k not in ("windows", "manifest")}
if "windows" in res and res["windows"] is not None:
    out["n_windows"] = len(res["windows"])
if "manifest" in res and res["manifest"] is not None:
    out["n_clips"] = sum(1 for p in res["manifest"]["clip_path"].to_list() if p)
print("=== RESULT ===")
print(json.dumps(out, indent=1, default=str)[:4000])
