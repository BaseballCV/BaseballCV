"""Per-play clip cutting and the manifest CSV."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import polars as pl

from baseballcv.utilities import BaseballCVLogger

from .config import GameSyncConfig

MANIFEST_COLUMNS = [
    "event_seq", "at_bat_index", "inning", "is_top_inning", "event_type", "event_text",
    "batter_name", "pitcher_name", "outs_before", "base_state_before",
    "clip_path", "video_start_s", "video_end_s",
]


def cut_clips(video_path: str, windows: pl.DataFrame, out_dir: str,
              config: Optional[GameSyncConfig] = None, logger=None) -> pl.DataFrame:
    """Cut one clip per play window.

    ffmpeg -ss/-to before -i with stream copy: fast, and keyframe-inexact by up
    to one GOP (~1-2 s), which the 20 s pre-roll / 8 s post-roll absorb.
    """
    cfg = config or GameSyncConfig()
    logger = logger or BaseballCVLogger.get_logger("cut_clips")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, r in enumerate(windows.iter_rows(named=True)):
        clip = out / f"play_{r.get('event_seq', i):04d}_inn{r.get('inning', 0)}{'T' if r.get('is_top_inning') else 'B'}.mp4"
        cmd = [
            cfg.ffmpeg_bin, "-y", "-v", "error",
            "-ss", f"{r['video_start_s']:.2f}", "-to", f"{r['video_end_s']:.2f}",
            "-i", video_path, "-c", "copy" if cfg.clip_copy else "libx264",
            str(clip),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            logger.warning(f"clip {clip.name} failed: {proc.stderr.strip()[:200]}")
            paths.append(None)
            continue
        paths.append(str(clip))
    manifest = windows.with_columns(pl.Series("clip_path", paths))
    manifest_path = out / "manifest.csv"
    manifest.select([c for c in MANIFEST_COLUMNS if c in manifest.columns]).write_csv(manifest_path)
    logger.info(f"{sum(p is not None for p in paths)}/{len(paths)} clips -> {manifest_path}")
    return manifest
