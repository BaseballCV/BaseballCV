"""End-to-end per-play runner metrics from synced clips.

Consumes game_sync's outputs (manifest.csv, calibration.npz, sync_state.json,
clock_anchors.npz) — never re-derives the sync. Per clip: 2-read OCR anchor
(first+last frame, sanity-checked against the manifest duration; fallback =
manifest video_start_s, distinguished by a status column), blob tracking,
runner selection, metric rows. Per-clip try/except: one bad clip never kills
the batch. Writes game_metrics.csv (house convention: flat rows, quality
columns, status sentinels, None-not-guess).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from baseballcv.utilities import BaseballCVLogger

from ..steal_analysis.geometry import Homography
from .config import MetricsConfig
from .metrics import baserunning_legs, runner_play_metrics, to_arrays
from .tracking import select_runner_track, track_clip

CLIP_ANCHOR_TOL_S = 2.0     # wall match + duration sanity tolerance
ANCHOR_VIDEO_WINDOW_S = 30.0


class GameMetricsRunner:
    def __init__(self, sync_results_dir: str = "run1/game_sync",
                 results_dir: str = "results/game_metrics",
                 config: Optional[MetricsConfig] = None, logger=None,
                 device: str = "cpu") -> None:
        self.sync_dir = sync_results_dir
        self.results_dir = results_dir
        self.config = config or MetricsConfig()
        self.device = device
        self.logger = logger or BaseballCVLogger.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _load_sync_artifacts(self) -> Dict:
        manifest_path = os.path.join(self.sync_dir, "manifest.csv")
        calib_path = os.path.join(self.sync_dir, "calibration.npz")
        state_path = os.path.join(self.sync_dir, "sync_state.json")
        anchors_path = os.path.join(self.sync_dir, "clock_anchors.npz")
        missing = [p for p in (manifest_path, calib_path, state_path, anchors_path)
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "run sync_game_video first; missing artifacts: " + ", ".join(missing))
        manifest = pl.read_csv(manifest_path)
        z = np.load(calib_path)
        homography = Homography.from_matrix(z["matrix"])
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        a = np.load(anchors_path)
        anchors = {"video_s": a["video_s"], "wall_s": a["wall_s"]}
        return {"manifest": manifest, "homography": homography,
                "state": state, "anchors": anchors}

    def _match_anchor_wall(self, wall: Optional[float], video_start_s: float,
                           anchors: Dict) -> Optional[float]:
        """Clip-frame wall time -> video time, via the persisted OCR anchors.

        Only anchors within +-30 s of the manifest's video_start_s are
        considered (the burned clock oscillates +-3600 s in one stretch of the
        reference feed; nearest-wall GLOBALLY would match the wrong branch).
        """
        if wall is None:
            return None
        cand = np.abs(anchors["video_s"] - video_start_s) <= ANCHOR_VIDEO_WINDOW_S
        if not cand.any():
            return None
        best = int(np.argmin(np.abs(anchors["wall_s"][cand] - wall)))
        if abs(anchors["wall_s"][cand][best] - wall) > CLIP_ANCHOR_TOL_S:
            return None
        return float(anchors["video_s"][cand][best])

    def _wall_to_video(self, wall: float, anchors: Dict) -> Optional[float]:
        """DB wall seconds-of-day -> video time via the persisted anchors.

        Piecewise by proximity: candidates restricted to anchors near the
        expected video time (wall slope is 1.0; the overlay folds +-3600 s in
        one stretch, so a global nearest-wall match would hit the wrong
        branch)."""
        video_s, wall_s = anchors["video_s"], anchors["wall_s"]
        base_wall = float(wall_s[0])
        expected = wall - base_wall
        cand = np.abs(video_s - expected) <= ANCHOR_VIDEO_WINDOW_S * 4
        if not cand.any():
            return None
        i = int(np.argmin(np.abs(wall_s[cand] - wall)))
        if abs(wall_s[cand][i] - wall) > 5.0:
            return None
        return float(video_s[cand][i])

    # ------------------------------------------------------------------
    def run(self, defense: bool = False) -> Dict:
        art = self._load_sync_artifacts()
        manifest, hom = art["manifest"], art["homography"]
        anchors = art["anchors"]

        # DB pitch times for the contact cross-check column (cached parquet;
        # no ssh when the snapshot is cached)
        from ..game_sync.config import GameSyncConfig
        from ..game_sync.pbp import PBPClient
        client = PBPClient(GameSyncConfig(),
                           cache_dir=os.path.join(self.sync_dir, "pbp_cache"),
                           logger=self.logger)
        snap = client.fetch_game(art["state"]["game_pk"])
        times = snap.play_times()
        pitch_video_by_event = {}
        offset = art["state"]["chosen_offset_s"]
        for r in times.iter_rows(named=True):
            dt = r.get("pa_last_pitch_time_dt") or r.get("event_time_dt")
            if dt is None:
                continue
            wall = dt.hour * 3600 + dt.minute * 60 + dt.second
            v = self._wall_to_video(wall + offset, anchors)
            if v is not None:
                pitch_video_by_event[r["event_seq"]] = v

        from ..game_sync.clock import BurnedClockReader
        reader = BurnedClockReader(logger=self.logger)

        rows: List[dict] = []
        for mrow in manifest.iter_rows(named=True):
            row = {k: mrow.get(k) for k in
                   ("event_seq", "at_bat_index", "inning", "is_top_inning",
                    "event_type", "event_text", "batter_name", "pitcher_name",
                    "outs_before", "base_state_before")}
            try:
                row.update(self._process_clip(mrow, hom, anchors, reader))
            except Exception as exc:   # one bad clip never kills the batch
                self.logger.warning(f"clip failed (event_seq={mrow['event_seq']}): {exc}")
                row.update({"status": f"clip_error: {exc}", "track_id": None})
            # DB cross-check: clip-local seconds of the PA's last pitch
            pv = pitch_video_by_event.get(mrow["event_seq"])
            row["contact_db_pitch_s"] = (
                round(pv - row["clip_video_start_s"], 2)
                if pv is not None and row.get("clip_video_start_s") is not None else None)
            rows.append(row)

        df = pl.DataFrame(rows)
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, "game_metrics.csv")
        df.write_csv(csv_path)
        n_ok = int((df["status"] == "ok").sum()) if len(df) else 0
        summary = {"n_clips": len(df), "n_ok": n_ok, "csv_path": csv_path,
                   "game_pk": art["state"]["game_pk"]}
        self.logger.info(f"game_metrics: {n_ok}/{len(df)} rows ok -> {csv_path}")
        return {"summary": summary, "rows": rows, "metrics": df}

    # ------------------------------------------------------------------
    def _process_clip(self, mrow: dict, hom, anchors: Dict, reader) -> dict:
        clip_path = mrow["clip_path"]
        if not clip_path or not os.path.exists(clip_path):
            return {"status": "clip_missing", "track_id": None}
        tracks = track_clip(clip_path, self.config, homography=hom,
                            clock_reader=reader, logger=self.logger)
        if tracks.n_frames == 0:
            return {"status": "clip_unreadable", "track_id": None}

        # clip -> video-time anchor: OCR wall of frame 0 matched to the
        # persisted anchors near the manifest seek; fallback = the seek itself
        video_start = float(mrow["video_start_s"])
        clip_start = self._match_anchor_wall(tracks.wall0_s, video_start, anchors)
        anchor_status = "ocr_matched" if clip_start is not None else "ocr_fallback"
        if clip_start is None:
            clip_start = video_start
        dur = (tracks.n_frames - 1) / tracks.fps
        duration_ok = (tracks.wall0_s is None or tracks.wall1_s is None
                       or abs((tracks.wall1_s - tracks.wall0_s) - dur) <= CLIP_ANCHOR_TOL_S)

        out = {"clip_path": clip_path, "n_frames_clip": tracks.n_frames,
               "clip_video_start_s": round(clip_start, 2),
               "clip_anchor_status": anchor_status,
               "clip_duration_ok": bool(duration_ok),
               "n_tracks": len(tracks.tracks)}

        tid, diag = select_runner_track(tracks, hom, self.config)
        out["track_id"] = tid
        out["n_runner_candidates"] = len(diag["candidates"])
        if tid is None:
            out["status"] = "no_runner_track"
            return out
        onset_frame = diag.get("onset_frame")
        field_xy = {f: hom.to_field((p[0], p[1]))
                    for f, p in tracks.tracks[tid].items()}
        m = runner_play_metrics(field_xy, tracks.fps, tracks.n_frames, self.config,
                                contact_frame=onset_frame)
        row = dict(m)
        row["bolt_first_move_s"] = row.get("contact_batter_move_s")
        row["bolt_timing_basis"] = "onset_after_pile"   # pile merge: bolt is the
        # corridor onset, not bat-ball contact (DB pitch column is the
        # coarse cross-check)
        row.update(out)
        # multi-base advances: legs as a json column (csv-safe)
        if mrow.get("event_type") in ("double", "triple", "home_run",
                                      "stolen_base_2b", "stolen_base_3b",
                                      "stolen_base_home"):
            legs = baserunning_legs(field_xy, tracks.fps, self.config)
            row["legs"] = json.dumps(legs) if legs else None
            row["n_legs"] = len(legs)
        return row


# legacy alias name kept out of __all__ on purpose: the runner is surfaced
# through BaseballTools.compute_game_metrics (see baseball_tools.py)
