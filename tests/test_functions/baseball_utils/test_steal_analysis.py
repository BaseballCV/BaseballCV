import numpy as np
import pandas as pd
import pytest

from baseballcv.functions.utils.baseball_utils.steal_analysis import (
    ClipTracks, PersonObs, StealAnalyzer, StealConfig, aggregate_runners, categorize_play,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.events import (
    arrival_frame, interpolate_gaps, pitcher_first_move, release_frame, runner_first_move, smooth,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import (
    BaselineAxis, scale_from_base_box, scale_from_height, scale_from_two_bases, stable_boxes, best_scale,
)
from baseballcv.functions.utils.baseball_utils.steal_analysis.metrics import classify_first_step
from baseballcv.functions.utils.savant_utils.steal_events import parse_height_in, parse_steal_events

FPS = 30.0
CFG = StealConfig()


# ---------------------------------------------------------------- geometry
class TestGeometry:
    def test_two_base_scale(self):
        est = scale_from_two_bases((0, 0), (450, 0), 90.0, 0.6)
        assert est.px_per_ft == pytest.approx(5.0)
        assert est.method == "two_bases"

    def test_single_base_scale(self):
        est = scale_from_base_box((0, 0, 30, 10), 18.0, 0.3)
        assert est.px_per_ft == pytest.approx(20.0)

    def test_height_scale(self):
        # 6 ft (72 in) runner, nose-to-ankle 0.93 of stature = 5.58 ft -> 279 px => 50 px/ft
        est = scale_from_height(279.0, 72.0, 0.5)
        assert est.px_per_ft == pytest.approx(50.0)

    def test_best_scale_prefers_confidence_and_reports_spread(self):
        a = scale_from_two_bases((0, 0), (450, 0), 90.0, 0.6)
        b = scale_from_base_box((0, 0, 8, 4), 18.0, 0.3)
        best, spread = best_scale([b, a])
        assert best.method == "two_bases"
        assert spread == pytest.approx(6.4, abs=0.5)

    def test_axis_projection(self):
        axis = BaselineAxis((300, 400), second_base=(750, 400), px_per_ft=5.0)
        assert axis.project((300, 400)) == pytest.approx(0.0)
        assert axis.project((360, 410)) == pytest.approx(12.0)
        assert axis.project((270, 400)) == pytest.approx(-6.0)

    def test_stable_boxes_drops_one_off_false_positive(self):
        per_frame = {f: [(100, 100, 130, 110, 0.9)] for f in range(20)}
        per_frame[5].append((500, 300, 530, 310, 0.5))
        boxes = stable_boxes(per_frame, min_frames=8, cluster_frac=0.6)
        assert len(boxes) == 1
        assert boxes[0][4] == 20


# ---------------------------------------------------------------- events
class TestEvents:
    def test_interpolate_gaps_fills_short_only(self):
        x = np.array([0, 1, np.nan, 3, np.nan, np.nan, np.nan, 7.0])
        out = interpolate_gaps(x, max_gap=2)
        assert out[2] == pytest.approx(2.0)
        assert np.isnan(out[4:7]).all()

    def test_smooth_preserves_nan(self):
        out = smooth([1, 2, np.nan, 4, 5], 3)
        assert np.isnan(out[2])

    def test_runner_first_move_recovered_within_two_frames(self):
        t = np.arange(120) / FPS
        move = 40
        pos = np.where(np.arange(120) < move, 12.0, 12.0 + 12.5 * ((np.arange(120) - move) / FPS) ** 2)
        got = runner_first_move(pos, FPS, CFG)
        assert got is not None and abs(got - move) <= 3

    def test_runner_first_move_none_when_static(self):
        assert runner_first_move(np.full(90, 12.0), FPS, CFG) is None

    def test_walking_lead_does_not_trigger_first_move(self):
        t = np.arange(120) / FPS
        pos = 12.0 + 0.8 * np.sin(t * 3)  # slow shuffle, well under 4 ft/s
        assert runner_first_move(pos, FPS, CFG) is None

    def test_pitcher_first_move(self):
        energy = np.full(100, 0.005)
        energy[0] = np.nan
        energy[50:] = 0.08
        got = pitcher_first_move(energy, FPS, CFG)
        assert got == 50

    def test_pitcher_first_move_none_without_data(self):
        assert pitcher_first_move(np.full(50, np.nan), FPS, CFG) is None

    def test_release_frame_needs_consecutive_run(self):
        assert release_frame([3, 9, 10, 11, 12], min_run=3) == 9
        assert release_frame([3, 9, 12], min_run=3) is None
        assert release_frame([1, 2, 3, 20, 21, 22], min_run=3, after=10) == 20

    def test_arrival_frame(self):
        along = np.linspace(0, 100, 101)
        assert arrival_frame(along, 90.0, 1.5) == 89


class TestFirstStep:
    def _ankles(self, back_first: bool):
        n = 20
        back = np.full(n, 12.0)
        front = np.full(n, 14.0)
        if back_first:
            back[3:] = 12.0 + 1.0 * np.arange(1, n - 2)  # back foot sweeps past the front foot
        else:
            front[3:] = 14.0 + 1.0 * np.arange(1, n - 2)  # front foot steps out
        return back, front

    def test_crossover(self):
        b, f = self._ankles(back_first=True)
        assert classify_first_step(b, f, 0, FPS, 0.6) == "crossover"

    def test_open_step(self):
        b, f = self._ankles(back_first=False)
        assert classify_first_step(b, f, 0, FPS, 0.6) == "open_step"

    def test_missing_ankles(self):
        nan = np.full(20, np.nan)
        assert classify_first_step(nan, nan, 0, FPS, 0.4) is None


# ---------------------------------------------------------------- categories
class TestCategorizer:
    def test_boundaries(self):
        base = {"scale_confidence": 0.6, "pre_pitch_speed_ft_s": 0.2, "reaction_time_s": 0.05}
        assert categorize_play({**base, "primary_lead_ft": 10.9}, CFG)["lead_length"] == "short"
        assert categorize_play({**base, "primary_lead_ft": 11.0}, CFG)["lead_length"] == "medium"
        assert categorize_play({**base, "primary_lead_ft": 13.5}, CFG)["lead_length"] == "long"
        assert categorize_play({**base, "primary_lead_ft": 12}, CFG)["lead_style"] == "static"
        assert categorize_play({**base, "primary_lead_ft": 12, "pre_pitch_speed_ft_s": 1.0}, CFG)["lead_style"] == "walking"
        assert categorize_play({**base, "primary_lead_ft": 12, "reaction_time_s": -0.06}, CFG)["jump"] == "early"
        assert categorize_play({**base, "primary_lead_ft": 12, "reaction_time_s": 0.16}, CFG)["jump"] == "late"
        assert categorize_play({**base, "primary_lead_ft": 12}, CFG)["jump"] == "on_time"

    def test_low_scale_confidence_withholds_distance_labels_only(self):
        row = {"scale_confidence": 0.3, "primary_lead_ft": 12.0, "pre_pitch_speed_ft_s": 2.0,
               "reaction_time_s": 0.0, "first_step": "crossover"}
        labels = categorize_play(row, CFG)
        assert labels["lead_length"] is None and labels["lead_style"] is None
        assert labels["jump"] == "on_time"
        assert labels["profile"] == "on_time-crossover"

    def test_aggregate_runners(self):
        df = pd.DataFrame({
            "runner_id": [1, 1, 1, 2],
            "primary_lead_ft": [12.0, 13.0, 14.0, 10.0],
            "reaction_time_s": [0.1, 0.1, 0.2, 0.0],
            "lead_length": ["medium", "medium", "long", "short"],
            "jump": ["on_time", "on_time", "late", "on_time"],
        })
        out = aggregate_runners(df, "runner_id", min_plays=2)
        assert list(out["runner_id"]) == [1]
        r = out.iloc[0]
        assert r["n_plays"] == 3
        assert r["primary_lead_ft_mean"] == pytest.approx(13.0)
        assert r["lead_length_mode"] == "medium"
        assert r["jump_consistency"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- end to end on synthetic tracks
def _person(x, y, still_after=None):
    kpts = np.tile([x, y - 60], (17, 1)).astype(float)
    kpts[5] = (x - 8, y - 55)   # shoulders
    kpts[6] = (x + 8, y - 55)
    kpts[11] = (x - 6, y - 30)  # hips
    kpts[12] = (x + 6, y - 30)
    kpts[15] = (x - 4, y)       # ankles
    kpts[16] = (x + 4, y)
    return PersonObs((x - 12, y - 70, x + 12, y), kpts, np.ones(17))


def _synthetic_steal(n=170, pitcher_move=30, runner_move=38, accel=25.0, lead=12.0, ppf=5.0):
    b1c, b2c = (300.0, 400.0), (300.0 + 90 * ppf, 400.0)
    tr = ClipTracks(fps=FPS, n_frames=n, width=1280, height=720, shots=[(0, n - 1)])
    for f in range(n):
        tr.bases[f] = [(b1c[0] - 4, b1c[1] - 2, b1c[0] + 4, b1c[1] + 2, 0.9),
                       (b2c[0] - 4, b2c[1] - 2, b2c[0] + 4, b2c[1] + 2, 0.9)]
    tr.people[1] = {}
    tr.people[2] = {}
    tr.people[3] = {}
    for f in range(n):
        dt = max(0.0, (f - runner_move) / FPS)
        ft = min(lead + 0.5 * accel * dt ** 2, 95.0)
        tr.people[1][f] = _person(b1c[0] + ft * ppf, 410.0)         # runner
        tr.people[2][f] = _person(b1c[0] - 20.0, 410.0)             # first baseman, stays put
        shift = 2.0 * max(0, f - pitcher_move + 1)
        p = _person(620.0, 380.0)
        p.kpts = p.kpts + np.array([shift, 0.0])
        p.box = (600, 300, 640, 400)
        tr.people[3][f] = p
        tr.pitcher[f] = (600, 300, 640, 400)
    for f in range(60, 70):
        tr.ball[f] = (500.0, 350.0)
    return tr


class TestAnalyzeTracks:
    @pytest.fixture(scope="class")
    def result(self):
        return StealAnalyzer(tracker=object()).analyze_tracks(_synthetic_steal())

    def test_scored(self, result):
        assert result["status"] == "ok"
        assert result["runner_track_id"] == 1
        assert result["ambiguous_runner"] is False

    def test_scale(self, result):
        assert result["scale_method"] == "two_bases"
        assert result["scale_confidence"] == pytest.approx(0.6)
        assert result["scale_px_per_ft"] == pytest.approx(5.0)

    def test_lead_at_pitcher_move(self, result):
        assert result["primary_lead_ft"] == pytest.approx(12.0, abs=0.1)
        assert result["lead_length"] == "medium"

    def test_reaction_time(self, result):
        # runner starts 8 frames after the pitcher (0.27 s); allow a few frames of detector latency
        assert 0.1 <= result["reaction_time_s"] <= 0.45
        assert result["reaction_resolution_s"] == pytest.approx(1 / 30, abs=1e-3)

    def test_run_metrics(self, result):
        assert result["time_to_10ft_s"] is not None and 0.5 < result["time_to_10ft_s"] < 2.5
        assert result["peak_speed_ft_s"] > 10
        assert result["move_to_arrival_s"] is not None
        assert "arrival" in result["events_observed"]

    def test_release_observed_after_pitcher_move(self, result):
        assert result["frame_release"] == 60

    def test_no_steal_when_runner_stays(self):
        tr = _synthetic_steal(accel=0.0)
        assert StealAnalyzer(tracker=object()).analyze_tracks(tr)["status"] == "no_steal_motion"

    def test_no_bases(self):
        tr = ClipTracks(fps=FPS, n_frames=60, width=1280, height=720, shots=[(0, 59)])
        assert StealAnalyzer(tracker=object()).analyze_tracks(tr)["status"] == "no_bases_detected"

    def test_missing_pitcher_leaves_reaction_none_not_guessed(self):
        tr = _synthetic_steal()
        tr.pitcher.clear()
        out = StealAnalyzer(tracker=object()).analyze_tracks(tr)
        assert out["status"] == "ok"
        assert out["reaction_time_s"] is None
        assert out["jump"] is None
        assert "pitcher_first_move" not in out["events_observed"]

    def test_single_base_is_low_confidence_and_withholds_lead_label(self):
        tr = _synthetic_steal()
        for f in tr.bases:
            tr.bases[f] = tr.bases[f][:1]
        out = StealAnalyzer(tracker=object()).analyze_tracks(tr)
        assert out["status"] == "ok"
        assert out["scale_method"] == "single_base"
        assert out["lead_length"] is None


# ---------------------------------------------------------------- statsapi parsing
class TestStealEvents:
    def test_parse_height(self):
        assert parse_height_in("6' 1\"") == 73.0
        assert parse_height_in(None) is None
        assert parse_height_in("n/a") is None

    def test_parse_steal_events(self):
        pbp = {"allPlays": [{
            "about": {"inning": 3, "halfInning": "top"},
            "matchup": {"pitcher": {"id": 9, "fullName": "P One"}, "pitchHand": {"code": "L"}},
            "runners": [{"details": {"runner": {"id": 5, "fullName": "R Five"}}}],
            "playEvents": [
                {"isPitch": True, "details": {"eventType": None}},
                {"details": {"eventType": "stolen_base_2b"}, "actionPlayId": "abc",
                 "player": {"id": 5}, "count": {"balls": 1, "strikes": 1, "outs": 1}},
                {"details": {"eventType": "stolen_base_3b"}, "actionPlayId": "no-3b", "player": {"id": 5}},
                {"details": {"eventType": "caught_stealing_2b"}, "player": {"id": 5}},  # no video id
            ],
        }]}
        rows = parse_steal_events(pbp, 123)
        assert len(rows) == 1
        r = rows[0]
        assert (r["play_id"], r["result"], r["runner_id"], r["runner_name"]) == ("abc", "SB", 5, "R Five")
        assert (r["p_throws"], r["outs"], r["game_pk"]) == ("L", 1, 123)
