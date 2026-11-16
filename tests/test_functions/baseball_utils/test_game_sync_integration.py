"""Integration test: real Magallanes cenital feed + real LVBP DB (game 869992).

Runs only with `--network` AND a local copy of the video:
    BASEBALLCV_SYNC_VIDEO="C:/path/1_juego27_2026-11-12.mp4" pytest -m network tests/test_functions/baseball_utils/test_game_sync_integration.py

Expectations (spike-verified, 2026-11-12 feed):
- DB pitch_by_pitch.time runs a constant ~90 min behind the burned-in clock.
- Clock anchors cover the game with small per-segment deviations.
- One offset candidate wins clearly; most pitches align with activity onsets.
"""
import os

import polars as pl
import pytest

from baseballcv.functions.utils.baseball_utils.game_sync import (
    BurnedClockReader, ClockMap, GameSyncConfig, GameSyncRunner,
)

pytestmark = pytest.mark.network

VIDEO = os.environ.get("BASEBALLCV_SYNC_VIDEO")
GAME_PK = 869992

requires_video = pytest.mark.skipif(not VIDEO or not os.path.exists(VIDEO),
                                    reason="set BASEBALLCV_SYNC_VIDEO to a local game video")


@requires_video
def test_end_to_end_sync(tmp_path):
    runner = GameSyncRunner(config=GameSyncConfig(), results_dir=str(tmp_path))
    res = runner.run(VIDEO, GAME_PK, cut_clips=False)
    assert "error" not in res, res.get("error")

    verdict = res["offset_verdict"]
    assert verdict["chosen_offset_s"] == pytest.approx(90 * 60, abs=30)
    assert verdict["margin"] >= 0.25

    report = res["sync_report"]
    assert report["matched_frac"] >= 0.6
    assert report["n_anchors"] >= 50
    assert report["median_slope"] == pytest.approx(1.0, abs=0.01)

    windows = res["windows"]
    assert len(windows) > 60          # 80 PAs in this game
    dur = windows["video_end_s"].max()
    assert dur <= 13138.1 + 1.0       # clamped to the video duration
    assert windows["video_start_s"].min() >= 0.0
