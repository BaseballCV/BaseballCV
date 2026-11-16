"""End-to-end game-video <-> PBP sync pipeline.

video + game_pk in -> clock verdict, calibration, per-play windows and clips out.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from baseballcv.utilities import BaseballCVLogger

from .activity import motion_timeline
from .calibration import build_field_homography, fisheye_diagnostics, locate_bases_from_priors, median_frame
from .clips import cut_clips as _cut_clips
from .clock import BurnedClockReader, ClockMap
from .config import GameSyncConfig
from .pbp import PBPClient
from .sync import GameClock, per_play_windows, resolve_offset, validation_report

# Spike-verified base priors for Magallanes park (854x480 cenital feed, 2026-11-12).
# Other venues: annotate one frame and add an entry here (venue key = venue_name
# from play_log_games, lowercased).
VENUE_BASE_PRIORS = {
    "estadio jose bernardo perez": {  # Magallanes, Valencia
        "home": (487.0, 277.0),
        "second": (516.0, 190.0),
        "first": (685.0, 211.0),
        "third": (342.0, 220.0),
    },
}


class GameSyncRunner:
    def __init__(self, device: str = "cpu", config: Optional[GameSyncConfig] = None,
                 results_dir: str = "results/game_sync", logger=None) -> None:
        self.device = device
        self.config = config or GameSyncConfig()
        self.results_dir = results_dir
        self.logger = logger or BaseballCVLogger.get_logger(self.__class__.__name__)

    def run(self, video_path: str, game_pk: int, cut_clips: bool = True,
            refresh_pbp: bool = False) -> Dict:
        if not os.path.exists(video_path):
            return {"error": f"Video file not found at {video_path}"}
        import cv2
        cap = cv2.VideoCapture(video_path)
        duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 1. PBP snapshot (read-only, cached)
        client = PBPClient(self.config, cache_dir=os.path.join(self.results_dir, "pbp_cache"),
                           logger=self.logger)
        snap = client.fetch_game(game_pk, refresh=refresh_pbp)
        play_times = snap.play_times()
        pitch_walls = [self._wall_of(r["time"]) for r in
                       snap.pitches.filter(snap.pitches["time"].is_not_null()).iter_rows(named=True)]

        # 2. Burned-in clock anchors -> piecewise map (persisted for debugging:
        #    anchor OCR is the slowest stage, never recompute to diagnose)
        reader = BurnedClockReader(self.config, logger=self.logger)
        anchors = reader.build_anchors(video_path, duration_s=duration_s,
                                       cache_dir=self.results_dir)
        self._persist("clock_anchors.npz", video_s=np.array([a.video_time_s for a in anchors]),
                      wall_s=np.array([a.wall_s for a in anchors]))
        clock_map = ClockMap.from_anchors(anchors, snap.game_date, self.config)

        # 3. Activity timeline + offset resolution/validation.
        #    The timeline is a reported diagnostic: on a continuous single-camera
        #    feed activity cannot discover the offset (see sync.resolve_offset).
        #    Cached per video: the full-file grab pass costs ~6 min.
        from .activity import MotionTimeline
        tcache = os.path.join(self.results_dir,
                              f"motion_timeline_{os.path.splitext(os.path.basename(video_path))[0]}_"
                              f"{os.path.getsize(video_path)}.npz")
        timeline = None
        if os.path.exists(tcache):
            try:
                z = np.load(tcache)
                timeline = MotionTimeline(z["time_s"], z["energy"], float(z["fps"]))
                self.logger.info(f"motion timeline (cached) from {os.path.basename(tcache)}")
            except (OSError, KeyError, ValueError):
                timeline = None
        if timeline is None:
            timeline = motion_timeline(video_path, self.config, logger=self.logger)
            self._persist(os.path.basename(tcache), time_s=timeline.times_s,
                          energy=timeline.energy, fps=np.float64(timeline.fps_sample))
        verdict = resolve_offset(clock_map, pitch_walls, duration_s, timeline,
                                 self.config, logger=self.logger)
        if verdict.margin < self.config.min_margin_frac:
            # never cut clips against an untrusted sync: fail loudly with the
            # candidate table so the offset question can be settled offline
            return {"error": f"offset unresolved: {verdict.reason}",
                    "offset_verdict": {"candidates": verdict.candidates.to_pandas().to_dict("records")},
                    "artifacts_dir": self.results_dir}
        clock = GameClock(clock_map=clock_map, offset_s=verdict.chosen_offset_s)
        report = validation_report(pitch_walls, clock, timeline, self.config)

        # 4. Calibration (bases near venue priors on a quiescent median frame)
        calibration = None
        venue_key = str(snap.header["venue_name"][0]).lower()
        priors = VENUE_BASE_PRIORS.get(venue_key)
        if priors:
            try:
                med = median_frame(video_path, duration_s * 0.1, self.config, logger=self.logger)
                locs = locate_bases_from_priors(med, priors, self.config)
                hom, notes = build_field_homography(locs)
                calibration = {
                    "pixels": {k: [round(v[0], 1), round(v[1], 1)] for k, v in locs.pixels.items()},
                    "confidence": locs.confidence,
                    "homography_notes": notes,
                    "fisheye": fisheye_diagnostics(locs),
                }
                # persisted so downstream passes (game_metrics) never re-derive it
                self._persist_calibration(locs, hom, venue_key)
            except Exception as exc:  # calibration must not kill the sync
                self.logger.warning(f"calibration failed (sync continues without it): {exc}")
                calibration = {"error": str(exc)}
        else:
            calibration = {"error": f"no base priors for venue '{venue_key}'"}

        # sync provenance for downstream passes (game_metrics): chosen offset,
        # game identity, venue — small json, written on every successful run
        self._persist_sync_state(game_pk=game_pk, game_date=str(snap.game_date),
                                 venue_name=str(snap.header["venue_name"][0]),
                                 chosen_offset_s=verdict.chosen_offset_s,
                                 offset_margin=verdict.margin,
                                 duration_s=duration_s, video_path=video_path)

        # 5. Windows + clips
        windows = per_play_windows(play_times, clock, self.config, duration_s=duration_s)
        manifest = None
        if cut_clips and not windows.is_empty():
            manifest = _cut_clips(video_path, windows, self.results_dir, self.config,
                                  logger=self.logger)

        return {
            "game_pk": game_pk,
            "game_date": str(snap.game_date),
            "offset_verdict": {"chosen_offset_s": verdict.chosen_offset_s,
                               "margin": verdict.margin, "reason": verdict.reason,
                               "candidates": verdict.candidates.to_pandas().to_dict("records")},
            "sync_report": report,
            "clock_diagnostics": clock_map.diagnostics(),
            "calibration": calibration,
            "windows": windows,
            "manifest": manifest,
        }

    @staticmethod
    def _wall_of(hhmmss: str) -> float:
        h, m, s = (int(p) for p in hhmmss.split(":")[:3])
        return h * 3600 + m * 60 + s

    def _persist_calibration(self, locs, hom, venue_key: str) -> None:
        """Homography + base pixels, so game_metrics never re-derives calibration."""
        self._persist("calibration.npz",
                      matrix=hom.matrix,
                      base_names=np.array(sorted(locs.pixels)),
                      base_pixels=np.array([locs.pixels[k] for k in sorted(locs.pixels)]),
                      confidence=np.float64(locs.confidence),
                      venue=np.array(venue_key))

    def _persist_sync_state(self, *, game_pk: int, game_date: str, venue_name: str,
                            chosen_offset_s: float, offset_margin: float,
                            duration_s: float, video_path: str) -> None:
        state = {"game_pk": game_pk, "game_date": game_date, "venue_name": venue_name,
                 "chosen_offset_s": chosen_offset_s, "offset_margin": offset_margin,
                 "duration_s": duration_s, "video_path": os.path.abspath(video_path)}
        try:
            with open(os.path.join(self.results_dir, "sync_state.json"), "w",
                      encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except OSError as exc:
            self.logger.warning(f"could not persist sync_state.json: {exc}")

    def _persist(self, name: str, **arrays) -> None:
        try:
            np.savez_compressed(os.path.join(self.results_dir, name), **arrays)
        except OSError as exc:
            self.logger.warning(f"could not persist {name}: {exc}")
