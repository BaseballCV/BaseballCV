"""Unit tests for game_sync (video <-> PBP synchronization).

All tests run offline: synthetic clock anchors, motion timelines, median
frames and PBP frames. The real-feed integration lives in
test_game_sync_integration.py under the `network` marker.
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from baseballcv.functions.utils.baseball_utils.game_sync import (
    ClockAnchor, ClockMap, GameSyncConfig, GameSnapshot,
    activity_match_score, build_field_homography, detect_onsets,
    fisheye_diagnostics, locate_bases_from_priors,
    parse_timestamp, per_play_windows, refine_offset, resolve_offset,
    validation_report,
)
from baseballcv.functions.utils.baseball_utils.game_sync.activity import MotionTimeline
from baseballcv.functions.utils.baseball_utils.game_sync.calibration import (
    _bright_blob, FIELD_POINTS,
)
from baseballcv.functions.utils.baseball_utils.game_sync.sync import GameClock

CFG = GameSyncConfig()
GAME_DATE = date(2026, 11, 12)


# ---------------------------------------------------------------- clock parsing
class TestParseTimestamp:
    def test_formats(self):
        assert parse_timestamp("11-12-2026 06:33:14 PM") == 18 * 3600 + 33 * 60 + 14
        assert parse_timestamp("11-12-2026 08:12:49PM") == 20 * 3600 + 12 * 60 + 49
        assert parse_timestamp("06:33:14 AM") == 6 * 3600 + 33 * 60 + 14
        assert parse_timestamp("12:05:00 AM") == 5 * 60
        assert parse_timestamp("12:05:00 PM") == 12 * 3600 + 5 * 60
        assert parse_timestamp("18:40:00") == 18 * 3600 + 40 * 60

    def test_garbage(self):
        assert parse_timestamp("") is None
        assert parse_timestamp("spray chart") is None
        assert parse_timestamp("25:99:99PM") is None


def _anchor(t, wall):
    return ClockAnchor(video_time_s=t, wall_s=wall, raw_text="x")


def _cmap():
    """Anchors every 60 s at a fixed 1 s/s clock, base wall 18:00:00."""
    base = 18 * 3600
    anchors = [_anchor(float(t), float(base + t)) for t in range(0, 720, 60)]
    return ClockMap.from_anchors(anchors, GAME_DATE, CFG), base


class TestClockMap:
    def _map(self, with_flip=True):
        """Anchors every 60 s; between t=360-600 the recorder displays +1 h."""
        base = 18 * 3600 + 32 * 60 + 44  # 18:32:44
        anchors = []
        for t in range(0, 900, 60):
            wall = base + t
            if with_flip and 360 <= t <= 600:
                wall += 3600
            anchors.append(_anchor(float(t), float(wall)))
        return ClockMap.from_anchors(anchors, GAME_DATE, CFG), base

    def test_fold_removes_display_oscillation(self):
        cmap, base = self._map()
        assert np.allclose(cmap.wall_times, base + cmap.video_times, atol=2.0)

    def test_piecewise_map_accuracy(self):
        cmap, base = self._map()
        for t in (10, 200, 450, 700, 830):  # within the anchor span (np.interp clamps beyond)
            assert abs(cmap.wall_at(t) - (base + t)) <= 2.0

    def test_inverse_and_datetime(self):
        cmap, base = self._map()
        t = cmap.video_at(base + 450)
        assert abs(t - 450) <= 2.0
        dt = cmap.datetime_at(450)
        assert dt == datetime(2026, 11, 12, 18, 32, 44) + timedelta(seconds=450)

    def test_needs_anchors(self):
        with pytest.raises(ValueError):
            ClockMap.from_anchors([_anchor(0, 100), _anchor(1, 101)], GAME_DATE, CFG)


# ---------------------------------------------------------------- activity
def _timeline_with_bursts(times_s):
    fps = 5.0
    t = np.arange(0, 600, 1 / fps)
    e = np.full_like(t, 1.0)
    for bt in times_s:
        e += 5.0 * np.exp(-0.5 * ((t - bt) / 0.25) ** 2)
    return MotionTimeline(t, e, fps)


class TestActivity:
    def test_onsets_at_known_times(self):
        bursts = [100.0, 200.0, 300.0]
        onsets = detect_onsets(_timeline_with_bursts(bursts), CFG)
        got = onsets["time_s"].to_numpy()
        assert len(got) == len(bursts)
        assert np.allclose(np.sort(got), bursts, atol=1.0)

    def test_flat_timeline_no_onsets(self):
        t = np.arange(0, 300, 0.2)
        onsets = detect_onsets(MotionTimeline(t, np.full_like(t, 1.0), 5.0), CFG)
        assert onsets.is_empty()

    def test_match_score(self):
        onsets = detect_onsets(_timeline_with_bursts([100, 200, 300, 400]), CFG)
        hits = np.array([100.5, 199.0, 300.2, 700.0])
        assert activity_match_score(hits, onsets, 3.0) == pytest.approx(0.75)


# ---------------------------------------------------------------- offset resolution
class TestResolveOffset:
    def _pitch_walls(self, base, video_times, offset_s):
        # db_wall such that video_at(db_wall + offset) == video_time
        return [base + t - offset_s for t in video_times]

    def _wide_cmap(self, span=7200.0):
        """Anchors every 60 s over `span` s, base wall 18:00:00."""
        base = 18 * 3600
        anchors = [_anchor(float(t), float(base + t)) for t in np.arange(0, span + 1, 60.0)]
        return ClockMap.from_anchors(anchors, GAME_DATE, CFG), base

    def test_interval_fit_picks_true_offset_and_rejects_decoys(self):
        cmap, base = self._wide_cmap()
        true_off = 90 * 60.0
        # full-game shape: first pitch 10 s in, last pitch 20 s before EOF
        walls = self._pitch_walls(base, [10.0, 700.0], true_off)
        verdict = resolve_offset(cmap, walls, 720.0, None, CFG)
        assert verdict.chosen_offset_s == true_off
        assert verdict.margin == pytest.approx(1.0)          # sole feasible candidate
        feas = {r["offset_min"]: r["feasible"] for r in verdict.candidates.to_dicts()}
        assert feas[90] is np.True_ or feas[90] is True
        for m in (-120, -60, -30, 0, 30, 60, 120):
            assert not feas[m], f"offset {m:+d} must be infeasible"

    def test_no_feasible_candidate_refuses(self):
        cmap, base = self._wide_cmap()
        # DB span wider than the recording: nothing can fit
        walls = self._pitch_walls(base, [10.0, 7000.0], 90 * 60.0)
        verdict = resolve_offset(cmap, walls, 720.0, None, CFG)
        assert verdict.margin == 0.0
        assert "NO candidate" in verdict.reason

    def test_multiple_feasible_activity_tiebreak(self):
        import dataclasses
        cmap, base = self._wide_cmap()
        true_off = 90 * 60.0
        # candidates are 1800 s apart, so two can fit only when the tail pad
        # exceeds that spacing; duration 2000 puts both (+90 -> 100..160 s,
        # +120 -> 1900..1960 s) inside the anchor range and the pads
        cfg2 = dataclasses.replace(CFG, sync_pad_max_s=3000.0, sync_pad_end_s=3600.0)
        walls = self._pitch_walls(base, [100.0, 160.0], true_off)
        tl = _timeline_with_bursts([100.0, 160.0])           # bursts at the true mapping
        verdict = resolve_offset(cmap, walls, 2000.0, tl, cfg2)
        assert verdict.chosen_offset_s == true_off
        assert verdict.margin >= CFG.min_margin_frac

    def test_multiple_feasible_low_margin_flagged(self):
        import dataclasses
        cmap, base = self._wide_cmap()
        cfg2 = dataclasses.replace(CFG, sync_pad_max_s=3000.0, sync_pad_end_s=3600.0)
        walls = self._pitch_walls(base, [100.0, 160.0], 90 * 60.0)
        flat = MotionTimeline(np.arange(0, 2000, 0.2), np.full(10000, 1.0), 5.0)
        verdict = resolve_offset(cmap, walls, 2000.0, flat, cfg2)
        assert "LOW MARGIN" in verdict.reason

    def test_refine_offsets_small_shift(self):
        cmap, base = _cmap()
        true_off, shift = 90 * 60.0, 1.5
        pitch_video = [100.0, 200.0, 300.0, 400.0]
        walls = self._pitch_walls(base, pitch_video, true_off + shift)
        tl = _timeline_with_bursts(pitch_video)
        clock = refine_offset(GameClock(clock_map=cmap, offset_s=true_off), walls, tl, CFG)
        # refine minimizes pitch->onset distance, so it converges to the shift
        # up to onset quantization (first above-floor sample ~1 s before the
        # burst peak at 5 fps), not to the 0.25 s refine grid
        assert abs(clock.offset_s - (true_off + shift)) <= 1.0


class TestPerPlayWindows:
    def test_margins_and_clamping(self):
        cmap, base = _cmap()
        clock = GameClock(clock_map=cmap, offset_s=90 * 60.0)
        rows = [
            {"event_seq": 1, "at_bat_index": 1, "inning": 1, "is_top_inning": True,
             "event_type": "strikeout", "event_text": "KS", "batter_name": "B", "pitcher_name": "P",
             "outs_before": 0, "base_state_before": 0,
             "pa_first_pitch_time_dt": datetime(2026, 11, 12, 19, 30, 0),
             "pa_last_pitch_time_dt": datetime(2026, 11, 12, 19, 30, 40)},
            {"event_seq": 2, "at_bat_index": 2, "inning": 1, "is_top_inning": True,
             "event_type": "single", "event_text": "1B7", "batter_name": "B2", "pitcher_name": "P",
             "outs_before": 1, "base_state_before": 1,
             "pa_first_pitch_time_dt": datetime(2026, 11, 12, 19, 31, 30),
             "pa_last_pitch_time_dt": datetime(2026, 11, 12, 19, 31, 50)},
        ]
        df = per_play_windows(pl.DataFrame(rows), clock, CFG, duration_s=200.0)
        w1, w2 = df.to_dicts()
        # first window: first pitch at wall base + (19:30:00 - 18:00:00 + 5400 offset) ...
        assert w1["video_start_s"] == CFG.pre_roll_s or w1["video_start_s"] > 0
        assert w2["video_start_s"] >= w1["video_start_s"]
        # window 2 clamped at duration
        assert w2["video_end_s"] <= 200.0

    def test_skips_rows_without_times(self):
        cmap, _ = _cmap()
        clock = GameClock(clock_map=cmap, offset_s=0.0)
        rows = [{"event_seq": 1, "inning": 1, "is_top_inning": True, "event_type": "x",
                 "pa_first_pitch_time_dt": None, "pa_last_pitch_time_dt": None,
                 "event_time_dt": None}]
        df = per_play_windows(pl.DataFrame(rows), clock, CFG)
        assert df.is_empty()


# ---------------------------------------------------------------- pbp
def _snapshot():
    events = pl.DataFrame({
        "event_seq": [1, 2], "at_bat_index": [0, 0], "inning": [1, 1],
        "is_top_inning": [True, True], "event_type": ["strikeout", None],
        "event_text": ["KS", "SB"], "is_pa": [True, False], "is_ab": [True, False],
        "batter_id": [5, 5], "batter_name": ["B", "B"], "bat_side": ["R", "R"],
        "pitcher_id": [9, 9], "pitcher_name": ["P", "P"], "pitch_hand": ["R", "R"],
        "outs_before": [0, 0], "outs_after": [1, 1], "r1_before": [None, 5],
        "r2_before": [None, None], "r3_before": [None, None], "r1_after": [None, 5],
        "r2_after": [None, None], "r3_after": [None, None],
        "base_state_before": [0, 1], "base_state_after": [0, 1],
        "away_score_before": [0, 0], "home_score_before": [0, 0],
        "away_score_after": [0, 0], "home_score_after": [0, 0],
        "balls": [1, 1], "strikes": [2, 2],
    })
    pitches = pl.DataFrame({
        "no": [1, 2, 3], "time": ["19:00:10", "19:00:35", "19:01:00"],
        "plate_appearance": [1, 1, 1], "inning": [1, 1, 1],
        "half_inning": ["Top", "Top", "Top"], "is_top_inning": [True, True, True],
        "pitcher_name": ["P"] * 3, "batter_name": ["B"] * 3,
        "balls": [0, 1, 1], "strikes": [0, 0, 1], "outs": [0, 0, 0],
        "pitch_type": ["FF"] * 3, "pitch_call": ["B", "S", "X"],
        "play_result": ["", "", "KS"],
    })
    header = pl.DataFrame({
        "game_pk": [869992], "official_date": ["2026-11-12"], "game_datetime": ["2026-11-12T23:00:00Z"],
        "day_night": ["N"], "game_type": ["R"], "season_stage": ["regular"],
        "venue_name": ["X"], "away_team_abbrev": ["LAG"], "home_team_abbrev": ["MAG"],
        "away_runs": [1], "home_runs": [2], "innings": [9],
    })
    return GameSnapshot(game_pk=869992, game_date=GAME_DATE, header=header,
                        events=events, pitches=pitches)


class TestPlayTimes:
    def test_join_takes_pa_first_pitch(self):
        pt = _snapshot().play_times()
        r = pt.filter(pl.col("event_seq") == 1).row(0, named=True)
        assert r["pa_first_pitch_time_dt"] == datetime(2026, 11, 12, 19, 0, 10)
        assert r["pa_last_pitch_time_dt"] == datetime(2026, 11, 12, 19, 1, 0)

    def test_plate_appearance_is_1based_vs_at_bat_index(self):
        # regression for the real-game bug: pitch_by_pitch.plate_appearance
        # runs 1..80 while play_log_events.at_bat_index runs 0..79; the join
        # must align pa-1 <-> abi, and consecutive PAs must both get times
        snap = _snapshot()
        snap.events = pl.DataFrame({
            "event_seq": [1, 2], "at_bat_index": [0, 1], "inning": [1, 1],
            "is_top_inning": [True, True], "event_type": ["strikeout", "single"],
            "event_text": ["KS", "1B7"], "is_pa": [True, True], "is_ab": [True, True],
            "batter_id": [5, 6], "batter_name": ["B", "B2"], "bat_side": ["R", "R"],
            "pitcher_id": [9, 9], "pitcher_name": ["P", "P"], "pitch_hand": ["R", "R"],
            "outs_before": [0, 1], "outs_after": [1, 1], "r1_before": [None, None],
            "r2_before": [None, None], "r3_before": [None, None], "r1_after": [None, 5],
            "r2_after": [None, None], "r3_after": [None, None],
            "base_state_before": [0, 1], "base_state_after": [1, 1],
            "away_score_before": [0, 0], "home_score_before": [0, 0],
            "away_score_after": [0, 0], "home_score_after": [0, 0],
            "balls": [1, 0], "strikes": [2, 0],
        })
        snap.pitches = pl.DataFrame({
            "no": [1, 2, 3, 4], "time": ["19:00:10", "19:01:00", "19:02:10", "19:02:40"],
            "plate_appearance": [1, 1, 2, 2], "inning": [1, 1, 1, 1],
            "half_inning": ["Top"] * 4, "is_top_inning": [True] * 4,
            "pitcher_name": ["P"] * 4, "batter_name": ["B", "B", "B2", "B2"],
            "balls": [0, 1, 0, 0], "strikes": [0, 1, 0, 1], "outs": [0, 0, 1, 1],
            "pitch_type": ["FF"] * 4, "pitch_call": ["B", "X", "B", "S"],
            "play_result": ["", "KS", "", ""],
        })
        pt = snap.play_times().sort("event_seq")
        assert pt["pa_first_pitch_time_dt"].to_list() == [
            datetime(2026, 11, 12, 19, 0, 10), datetime(2026, 11, 12, 19, 2, 10)]

    def test_pa_continues_across_half_innings(self):
        # regression for the real-game bug: pitch_by_pitch.plate_appearance is
        # a lineup slot that keeps counting across halves (top-2nd ended at
        # pa 7, bottom-2nd starts at pa 7), so no arithmetic on the value can
        # map it to at_bat_index — the PA ordinal within the half must be used
        snap = _snapshot()
        snap.events = pl.DataFrame({
            "event_seq": [1, 2, 3], "at_bat_index": [0, 1, 2], "inning": [2, 2, 2],
            "is_top_inning": [True, True, False], "event_type": ["strikeout", "single", "single"],
            "event_text": ["KS", "1B7", "1B7"], "is_pa": [True, True, True],
            "is_ab": [True, True, True], "batter_id": [5, 6, 7],
            "batter_name": ["B", "B2", "B3"], "bat_side": ["R", "R", "R"],
            "pitcher_id": [9, 9, 9], "pitcher_name": ["P", "P", "P"], "pitch_hand": ["R", "R", "R"],
            "outs_before": [0, 1, 0], "outs_after": [1, 2, 1],
            "r1_before": [None, None, None], "r2_before": [None, None, None],
            "r3_before": [None, None, None], "r1_after": [None, 5, 6],
            "r2_after": [None, None, None], "r3_after": [None, None, None],
            "base_state_before": [0, 1, 0], "base_state_after": [1, 1, 1],
            "away_score_before": [0, 0, 0], "home_score_before": [0, 0, 0],
            "away_score_after": [0, 0, 0], "home_score_after": [0, 0, 0],
            "balls": [1, 0, 0], "strikes": [2, 0, 0],
        })
        snap.pitches = pl.DataFrame({
            "no": [1, 2, 3, 4, 5],
            "time": ["18:00:10", "18:01:00", "18:05:10", "18:05:40", "18:06:00"],
            "plate_appearance": [6, 7, 7, 8, 8],       # 7 spans both halves
            "inning": [2, 2, 2, 2, 2],
            "half_inning": ["Top", "Top", "Bottom", "Bottom", "Bottom"],
            "is_top_inning": [True, True, False, False, False],
            "pitcher_name": ["P"] * 5, "batter_name": ["B", "B2", "B3", "B3", "B4"],
            "balls": [0] * 5, "strikes": [0] * 5, "outs": [0, 1, 0, 0, 0],
            "pitch_type": ["FF"] * 5, "pitch_call": ["X", "X", "B", "X", "B"],
            "play_result": ["KS", "1B7", "", "KS", ""],
        })
        pt = snap.play_times().sort("event_seq")
        assert pt["pa_first_pitch_time_dt"].to_list() == [
            datetime(2026, 11, 12, 18, 0, 10),   # top 1st PA (pa 6, ordinal 1)
            datetime(2026, 11, 12, 18, 1, 0),    # top 2nd PA (pa 7, ordinal 2)
            datetime(2026, 11, 12, 18, 5, 10),   # bottom 1st PA (pa 7 again, ordinal 1)
        ]

    def test_midnight_rollover_pushes_next_day(self):
        snap = _snapshot()
        # two PAs: one pitching at 23:59:00, the next at 00:00:30 next day
        snap.pitches = pl.DataFrame({
            "no": [1, 2], "time": ["23:59:00", "00:00:30"],
            "plate_appearance": [10, 11], "inning": [10, 10],
            "half_inning": ["Top", "Top"], "is_top_inning": [True, True],
            "pitcher_name": ["P", "P"], "batter_name": ["B", "B"],
            "balls": [0, 0], "strikes": [0, 0], "outs": [0, 0],
            "pitch_type": ["FF", "FF"], "pitch_call": ["B", "X"], "play_result": ["", "F8"],
        })
        snap.events = pl.DataFrame({
            "event_seq": [49, 50], "at_bat_index": [9, 10], "inning": [10, 10],
            "is_top_inning": [True, True], "event_type": ["strikeout", "field_out"],
            "event_text": ["KS", "F8"],
            "is_pa": [True, True], "is_ab": [True, True], "batter_id": [1, 1], "batter_name": ["B", "B"],
            "bat_side": ["R", "R"], "pitcher_id": [2, 2], "pitcher_name": ["P", "P"], "pitch_hand": ["R", "R"],
            "outs_before": [0, 0], "outs_after": [1, 2], "r1_before": [None, None], "r2_before": [None, None],
            "r3_before": [None, None], "r1_after": [None, None], "r2_after": [None, None], "r3_after": [None, None],
            "base_state_before": [0, 0], "base_state_after": [0, 0],
            "away_score_before": [0, 0], "home_score_before": [0, 0],
            "away_score_after": [0, 0], "home_score_after": [0, 0], "balls": [0, 0], "strikes": [1, 1],
        })
        pt = snap.play_times().sort("event_seq")
        assert pt["event_time_dt"][0] == datetime(2026, 11, 12, 23, 59, 0)
        assert pt["event_time_dt"][1] == datetime(2026, 11, 13, 0, 0, 30)


class TestPBPClientSafety:
    def test_query_always_read_only(self, monkeypatch):
        from baseballcv.functions.utils.baseball_utils.game_sync.pbp import PBPClient
        seen = {}

        def fake_run(cmd, capture_output, text, timeout):
            seen["cmd"] = cmd
            raise RuntimeError("stop here")

        monkeypatch.setattr("subprocess.run", fake_run)
        client = PBPClient(CFG, cache_dir=str(Path.cwd() / "_pbp_test_cache"))
        with pytest.raises(RuntimeError):
            client._query("SELECT 1")
        cmd = " ".join(seen["cmd"])
        assert "mode=ro" in cmd
        assert "www-data" in cmd
        assert "sudo" in cmd
        # the SQL must be one shell-quoted word for the remote shell: a
        # multi-line SQL previously got executed line-by-line by remote bash
        assert "sqlite3 -json" in cmd


# ---------------------------------------------------------------- calibration
def _median_with_bases(bases):
    med = np.full((480, 854), 100, np.uint8)
    for x, y in bases.values():
        xi, yi = int(round(x)), int(round(y))
        med[yi - 1:yi + 1, xi - 1:xi + 1] = 250   # 2x2 bright bag
    return med


class TestCalibration:
    def test_bright_blob_centroid(self):
        med = _median_with_bases({"second": (516, 190)})
        pos, score = _bright_blob(med, (516, 190), radius=10, contrast_min=5.0)
        assert pos is not None
        assert abs(pos[0] - 516) <= 1.0 and abs(pos[1] - 190) <= 1.0
        assert score >= 5.0

    def test_bright_blob_none_in_flat_area(self):
        med = np.full((480, 854), 100, np.uint8)
        pos, score = _bright_blob(med, (400, 240), radius=10, contrast_min=5.0)
        assert pos is None and score < 5.0

    def test_locate_and_homography_round_trip(self):
        # fisheye-style kite: diagonals intentionally unequal in pixels
        bases = {"home": (487.0, 277.0), "second": (516.0, 190.0),
                 "first": (685.0, 211.0), "third": (342.0, 220.0)}
        med = _median_with_bases({k: (int(round(v[0])), int(round(v[1]))) for k, v in bases.items()})
        locs = locate_bases_from_priors(med, bases, CFG)
        assert set(locs.pixels) == set(bases)
        for name, (px, py) in locs.pixels.items():
            assert abs(px - bases[name][0]) <= 1.5 and abs(py - bases[name][1]) <= 1.5
        hom, notes = build_field_homography(locs)
        fx, fy = hom.to_field(bases["first"])
        assert (fx, fy) == pytest.approx(FIELD_POINTS["first"], abs=1.0)
        sx, sy = hom.to_field(bases["second"])
        assert (sx, sy) == pytest.approx(FIELD_POINTS["second"], abs=1.0)
        diag = fisheye_diagnostics(locs)
        assert diag["kite_ratio"] > 1.5   # 1B-3B px distance >> home-2B px distance

    def test_three_bases_with_known_diagonal(self):
        bases = {"home": (487.0, 277.0), "second": (516.0, 190.0), "first": (685.0, 211.0)}
        med = _median_with_bases({k: (int(v[0]), int(v[1])) for k, v in bases.items()})
        locs = locate_bases_from_priors(med, bases, CFG)
        hom, notes = build_field_homography(locs)
        assert notes.get("mirrored") == "third"
        fx, fy = hom.to_field(bases["home"])
        assert (fx, fy) == pytest.approx((0.0, 0.0), abs=2.0)

    def test_rejects_insufficient_bases(self):
        bases = {"home": (487.0, 277.0), "first": (685.0, 211.0)}
        med = _median_with_bases(bases)
        locs = locate_bases_from_priors(med, bases, CFG)
        with pytest.raises(ValueError):
            build_field_homography(locs)


# ---------------------------------------------------------------- clips
class TestCutClips:
    def test_ffmpeg_args_and_manifest(self, monkeypatch, tmp_path):
        from baseballcv.functions.utils.baseball_utils.game_sync import clips as clips_mod
        seen = []

        def fake_run(cmd, capture_output, text, timeout):
            seen.append(cmd)
            out = tmp_path / Path(cmd[-1]).name
            out.write_bytes(b"")  # pretend the clip exists
            class P:
                returncode = 0
                stderr = ""
            return P()

        monkeypatch.setattr(clips_mod.subprocess, "run", fake_run)
        windows = pl.DataFrame([
            {"event_seq": 1, "inning": 1, "is_top_inning": True, "event_type": "KS",
             "event_text": "KS23", "batter_name": "B", "pitcher_name": "P",
             "outs_before": 0, "base_state_before": 0,
             "video_start_s": 10.0, "video_end_s": 60.0},
        ])
        manifest = clips_mod.cut_clips("game.mp4", windows, str(tmp_path), CFG)
        cmd = seen[0]
        assert cmd[cmd.index("-ss") + 1] == "10.00"
        assert cmd.index("-ss") < cmd.index("-i")          # seek before input
        assert "-c" in cmd and "copy" in cmd
        assert manifest["clip_path"][0] is not None
        assert (tmp_path / "manifest.csv").exists()

    def test_failed_clip_recorded_not_raised(self, monkeypatch, tmp_path):
        from baseballcv.functions.utils.baseball_utils.game_sync import clips as clips_mod

        def fake_run(cmd, capture_output, text, timeout):
            class P:
                returncode = 1
                stderr = "boom"
            return P()

        monkeypatch.setattr(clips_mod.subprocess, "run", fake_run)
        windows = pl.DataFrame([
            {"event_seq": 1, "inning": 1, "is_top_inning": True, "event_type": "KS",
             "event_text": "KS", "batter_name": "B", "pitcher_name": "P",
             "outs_before": 0, "base_state_before": 0,
             "video_start_s": 10.0, "video_end_s": 60.0},
        ])
        manifest = clips_mod.cut_clips("game.mp4", windows, str(tmp_path), CFG)
        assert manifest["clip_path"][0] is None
