"""Unit tests for game_metrics (runner + defender metrics from blob tracks).

All tests run offline: synthetic clips (dark blobs on known field-space
trajectories through the reference homography), synthetic tracks and frozen
frames. The real-feed gate lives in scripts/spike_s6_runner_blobs.py.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from baseballcv.functions.utils.baseball_utils.game_metrics.config import MetricsConfig
from baseballcv.functions.utils.baseball_utils.game_metrics.metrics import (
    FIRST_BASE, baserunning_legs, first_move_frame, runner_play_metrics, to_arrays,
)
from baseballcv.functions.utils.baseball_utils.game_metrics.tracking import (
    BlobTracks, associate_blobs, detect_dark_blobs, extract_blobs,
    median_background, select_runner_track, track_clip, track_field_xy,
)
from baseballcv.functions.utils.baseball_utils.game_metrics.defense import (
    assign_positions, defender_positions_at, fielder_play_metrics, positioning_table,
)
from baseballcv.functions.utils.baseball_utils.game_sync.calibration import FIELD_POINTS
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

CFG = MetricsConfig()
FPS = 30.0

# fisheye-style kite: the reference feed's base pixels (unequal diagonals)
IMG_BASES = {"home": (487.0, 277.0), "second": (516.0, 190.0),
             "first": (685.0, 211.0), "third": (342.0, 220.0)}
HOM = Homography(list(IMG_BASES.values()),
                 [FIELD_POINTS[k] for k in ("home", "second", "first", "third")])


def img_of(pt_ft: Tuple[float, float]) -> Tuple[float, float]:
    return HOM.field_to_image(pt_ft)


# ---------------------------------------------------------------- synthetic clips
def write_synthetic_clip(path, trajectories, n_frames, size=(854, 480),
                         bg_level=90, blob_level=40, seed=0):
    """Encode a clip: flat turf + one DARK blob per trajectory.

    ``trajectories``: list of callables frame -> field ft (x, y) or None
    (None = untracked that frame). Blobs are drawn at the projected pixel.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, size)
    assert writer.isOpened()
    rng = np.random.default_rng(seed)
    for f in range(n_frames):
        frame = np.full((size[1], size[0], 3), bg_level, np.uint8)
        frame[..., :3] = np.clip(
            frame[..., :3].astype(int) + rng.integers(-3, 4, frame[..., :3].shape), 0, 255
        ).astype(np.uint8)
        for traj in trajectories:
            pt = traj(f)
            if pt is None:
                continue
            x, y = img_of(pt)
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(frame, (xi, yi), 2, (blob_level,) * 3, -1)
        writer.write(frame)
    writer.release()


def linear_traj(p0, p1, fps_speed_ft_s, start_frame=0):
    """Callable: moves p0->p1 at fps_speed_ft_s starting at start_frame, then stops."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    total = float(np.hypot(*(p1 - p0)))
    dur = total / fps_speed_ft_s * FPS
    unit = (p1 - p0) / total

    def traj(f):
        if f < start_frame:
            return tuple(p0)
        t = min(f - start_frame, dur) / FPS
        return tuple(p0 + unit * fps_speed_ft_s * t)
    return traj


def stationary(pt):
    return lambda f: tuple(pt)


# ---------------------------------------------------------------- blob extraction
class TestExtractBlobs:
    def test_centroids_within_1px(self):
        bg = np.full((480, 854), 90, np.uint8)
        frame = bg.copy()
        for ft in ((30.0, 10.0), (60.0, 40.0)):
            x, y = img_of(ft)
            frame[int(round(y)) - 2:int(round(y)) + 3,
                  int(round(x)) - 2:int(round(x)) + 3] = 40
        dets = extract_blobs(frame, bg, CFG)
        assert len(dets) == 2
        want = sorted(img_of(ft) for ft in ((30.0, 10.0), (60.0, 40.0)))
        got = sorted((d[0], d[1]) for d in dets)
        for (wx, wy), (gx, gy) in zip(want, got):
            assert abs(wx - gx) <= 1.0 and abs(wy - gy) <= 1.0

    def test_sub_min_area_rejected(self):
        bg = np.full((480, 854), 90, np.uint8)
        frame = bg.copy()
        frame[100:102, 200:202] = 40   # 4 px < blob_min_area_px
        assert extract_blobs(frame, bg, CFG) == []

    def test_flicker_not_detected(self):
        # compression flicker (~0.8 gray levels) stays far below the threshold
        bg = np.full((480, 854), 90, np.uint8)
        frame = bg.copy() + 2
        assert extract_blobs(frame, bg, CFG) == []


# ---------------------------------------------------------------- association
def _dets_from_tracks(specs, n_frames):
    """specs: lists of img pts per frame (None = undetected). Per-frame det lists."""
    out = []
    for f in range(n_frames):
        dets = []
        for spec in specs:
            pt = spec(f) if callable(spec) else (spec[f] if f < len(spec) else None)
            if pt is not None:
                dets.append((pt[0], pt[1], 12))
        out.append(dets)
    return out


class TestAssociate:
    def test_identity_through_head_on_crossing(self):
        # two blobs converging on the 1B line; constant-velocity prediction must
        # keep identities past the crossing point
        a = [(20.0 + f * 1.0, 10.0) for f in range(30)]           # fast +x
        b = [(60.0 - f * 0.5, 10.0) for f in range(30)]           # slower -x
        tracks = associate_blobs(_dets_from_tracks([a, b], 30), FPS, CFG)
        assert len(tracks) == 2
        for frames in tracks.values():
            xs = [frames[f][0] for f in sorted(frames)]
            assert len(xs) == 30
            # each track stays monotonic: never jumps back against its own velocity
            diffs = np.diff(xs)
            assert (np.all(diffs > 0) or np.all(diffs < 0)), diffs

    def test_occlusion_gap_keeps_one_track(self):
        # 5-frame disappearance: prediction bridges it, one track, gap present
        pts = [(30.0 + f * 0.9, 10.0) if not (10 <= f < 15) else None
               for f in range(30)]
        tracks = associate_blobs(_dets_from_tracks([pts], 30), FPS, CFG)
        assert len(tracks) == 1
        frames = next(iter(tracks.values()))
        assert max(frames) - min(frames) + 1 == 30
        assert 10 not in frames and 14 not in frames

    def test_static_decoy_not_merged(self):
        # a decoy parked beside the path: mover stays on y=10, decoy on y=8
        mover = [(30.0 + f * 1.0, 10.0) for f in range(40)]
        decoy = [(40.0, 8.0)] * 40
        tracks = associate_blobs(_dets_from_tracks([mover, decoy], 40), FPS, CFG)
        assert len(tracks) == 2
        ys_seen = sorted({round(next(iter(fr.values()))[1], 1) for fr in tracks.values()})
        assert ys_seen == [8.0, 10.0]
        for fr in tracks.values():
            ys = {round(p[1], 1) for p in fr.values()}
            assert len(ys) == 1   # each track keeps its own lane

    def test_short_tracks_dropped(self):
        pts = [(50.0, 50.0)] * (CFG.track_min_frames - 1)
        assert associate_blobs(_dets_from_tracks([pts], len(pts)), FPS, CFG) == {}


# ---------------------------------------------------------------- runner selection
def _blobtracks_from_field(trajs: Dict[int, list], n_frames) -> BlobTracks:
    """trajs: track_id -> list of field ft (len n_frames; entries may be None)."""
    tracks = {}
    for tid, seq in trajs.items():
        frames = {}
        for f, pt in enumerate(seq):
            if pt is not None:
                x, y = img_of(pt)
                frames[f] = (x, y, 12)
        if len(frames) >= CFG.track_min_frames:
            tracks[tid] = frames
    return BlobTracks(fps=FPS, n_frames=n_frames, tracks=tracks)


def _field_seq(fn, n_frames):
    return [fn(f) for f in range(n_frames)]


class TestSelectRunner:
    def test_selects_batter_runner_among_decoys(self):
        runner = linear_traj((24.0, 3.0), (75.0, 3.0), 25.0)     # bolt toward 1B
        decoy_near = stationary((12.0, 45.0))                    # right zone, no motion
        decoy_out = linear_traj((40.0, 60.0), (45.0, 60.0), 9.0)  # out of start band
        t = _blobtracks_from_field({0: _field_seq(runner, 90),
                                    1: _field_seq(decoy_near, 90),
                                    2: _field_seq(decoy_out, 90)}, 90)
        tid, diag = select_runner_track(t, HOM, CFG)
        assert tid == 0
        assert diag["selected"] == 0

    def test_lead_in_stillness_still_selected(self):
        # runner stands still 1 s (pre-pitch lead-in), then bolts
        runner = linear_traj((24.0, 3.0), (75.0, 3.0), 25.0, start_frame=30)
        t = _blobtracks_from_field({0: _field_seq(runner, 120)}, 120)
        tid, _ = select_runner_track(t, HOM, CFG)
        assert tid == 0

    def test_none_when_no_qualifier(self):
        still = stationary((10.0, 5.0))
        away = linear_traj((24.0, 5.0), (2.0, 5.0), 12.0)   # toward home, not 1B
        t = _blobtracks_from_field({0: _field_seq(still, 60),
                                    1: _field_seq(away, 60)}, 60)
        tid, _ = select_runner_track(t, HOM, CFG)
        assert tid is None


# ---------------------------------------------------------------- end-to-end on a file
class TestTrackClip:
    def test_synthetic_clip_round_trip(self, tmp_path):
        runner = linear_traj((24.0, 3.0), (80.0, 3.0), 25.0)
        decoy = stationary((12.0, 45.0))
        clip = tmp_path / "play.mp4"
        write_synthetic_clip(clip, [runner, decoy], 90)
        tracks = track_clip(str(clip), CFG)
        assert tracks.n_frames == 90
        assert len(tracks.tracks) >= 1
        tid, diag = select_runner_track(tracks, HOM, CFG)
        assert tid is not None
        frames = tracks.tracks[tid]
        # tracked most frames and ends near the trajectory's end in field ft
        assert len(frames) >= 0.8 * 90
        end_ft = HOM.to_field((frames[max(frames)][0], frames[max(frames)][1]))
        assert abs(end_ft[0] - 80.0) <= 4.0 and abs(end_ft[1] - 3.0) <= 4.0

    def test_missing_file(self, tmp_path):
        tracks = track_clip(str(tmp_path / "nope.mp4"), CFG)
        assert tracks.n_frames == 0 and tracks.tracks == {}


# ---------------------------------------------------------------- metrics
class TestRunnerMetrics:
    def test_known_27fts_baseline(self):
        # stand 1 s, then 27 ft/s home->1B along the line
        traj = linear_traj((2.0, 1.0), (92.0, 1.0), 27.0, start_frame=30)
        field = {f: traj(f) for f in range(180)}
        m = runner_play_metrics(field, FPS, 180, CFG, contact_frame=30)
        assert m["status"] == "ok", m["status"]
        # arrival fires when the runner first comes within arrival_margin_ft
        # (3 ft) of the bag, i.e. ~3 ft before the base itself
        assert m["time_to_1b_s"] == pytest.approx((90.0 - 3.0) / 27.0, abs=0.15)
        assert m["peak_speed_ft_s"] == pytest.approx(27.0, abs=1.5)
        for cp in (15, 30, 45, 60, 75):
            assert m[f"speed_at_{cp}_ft"] == pytest.approx(27.0, abs=2.0), cp
        # straight line to the arrival point (3 ft short of the bag) is 85 ft
        assert m["route_efficiency"] == pytest.approx(90.0 / 85.0, abs=0.03)
        assert m["contact_batter_move_frame"] == 30
        assert m["contact_batter_move_s"] == pytest.approx(1.0, abs=0.01)

    def test_first_move_detection_within_3_frames(self):
        traj = linear_traj((2.0, 1.0), (92.0, 1.0), 27.0, start_frame=45)
        field = {f: traj(f) for f in range(180)}
        xs, ys = to_arrays(field, 180)
        fm = first_move_frame(xs, ys, FPS, CFG)
        assert fm is not None and abs(fm - 45) <= 3

    def test_stationary_runner_none(self):
        field = {f: (10.0, 3.0) for f in range(60)}
        m = runner_play_metrics(field, FPS, 60, CFG)
        assert m["status"] == "no_first_move"
        assert m["time_to_1b_s"] is None and m["peak_speed_ft_s"] is None

    def test_empty_track(self):
        m = runner_play_metrics({}, FPS, 0, CFG)
        assert m["status"] == "no_track"

    def test_never_arrives(self):
        # runner thrown out at 60 ft: no arrival, status carries it, partial values kept
        traj = linear_traj((2.0, 1.0), (60.0, 1.0), 25.0)
        field = {f: traj(f) for f in range(200)}
        m = runner_play_metrics(field, FPS, 200, CFG, contact_frame=0)
        assert "no_arrival" in m["status"]
        assert m["time_to_1b_s"] is None
        assert m["peak_speed_ft_s"] is not None
        assert m["min_d_1b_ft"] == pytest.approx(30.0, abs=2.0)


class TestBaserunningLegs:
    def test_triple_three_legs(self):
        # home->1B->2B->3B at ~25 ft/s with brief stops at the bags
        legs_pts = [(2.0, 1.0), FIRST_BASE, (63.6, 63.6), (0.0, 92.0)]
        seq = []
        for i in range(len(legs_pts) - 1):
            seg = linear_traj(legs_pts[i], legs_pts[i + 1], 25.0)
            seq.extend(seg(f) for f in range(int(90 / 25 * FPS) + 15))
        field = {f: pt for f, pt in enumerate(seq)}
        legs = baserunning_legs(field, FPS, CFG)
        assert [l["to_base"] for l in legs] == [1, 2, 3]
        assert [l["from_base"] for l in legs] == [0, 1, 2]
        assert len(legs) == 3
        for leg in legs:
            assert leg["median_speed_ft_s"] == pytest.approx(25.0, abs=3.0)
            # legs between arrivals include the brief stop at the bag
            assert leg["duration_s"] == pytest.approx(90.0 / 25.0, abs=0.6)

    def test_empty(self):
        assert baserunning_legs({}, FPS, CFG) == []


# ---------------------------------------------------------------- defense
class TestDefense:
    def test_seven_defenders_assigned(self):
        priors = {k: CFG.defender_priors_ft[k] for k in ("P", "C", "1B", "2B", "3B", "SS", "LF")}
        pts = [(px + 2.0, py - 1.0) for px, py, _r in priors.values()]
        out = assign_positions(pts, CFG, priors=priors)
        assert all(out[k] is not None for k in priors)
        # each point used once
        assert len({tuple(p) for p in out.values() if p}) == 7

    def test_missing_ss_maps_none(self):
        priors = {k: CFG.defender_priors_ft[k] for k in ("P", "SS")}
        pts = [(42.0, 43.0)]   # P only
        out = assign_positions(pts, CFG, priors=priors)
        assert out["P"] is not None and out["SS"] is None

    def test_two_points_one_slot(self):
        # two detections inside the SS radius: only one is claimed
        priors = {"SS": CFG.defender_priors_ft["SS"]}
        out = assign_positions([(48.0, 58.0), (50.0, 60.0)], CFG, priors=priors)
        assert sum(v is not None for v in out.values()) == 1

    def test_defender_positions_at_synthetic_frame(self):
        # frozen frame with infielders drawn at their priors
        import cv2
        frame = np.full((480, 854), 90, np.uint8)
        slots = ("P", "C", "1B", "2B", "3B", "SS")
        for k in slots:
            px, py, _r = CFG.defender_priors_ft[k]
            x, y = img_of((px, py))
            cv2.circle(frame, (int(round(x)), int(round(y))), 2, 40, -1)
        out = defender_positions_at(frame, HOM, CFG)
        found = [k for k in slots if out[k] is not None]
        assert len(found) >= 5, out
        for k in found:
            px, py, _r = CFG.defender_priors_ft[k]
            got = out[k]
            assert np.hypot(got[0] - px, got[1] - py) <= 4.0

    def test_fielder_play_metrics(self):
        # stand still after contact, then move 10 ft/s away from the start
        traj = linear_traj((48.0, 58.0), (69.2, 79.2), 10.0, start_frame=30)
        field = {f: traj(f) for f in range(150)}
        row = fielder_play_metrics(field, contact_frame=30, fps=FPS, cfg=CFG)
        assert row["status"] == "ok"
        assert row["first_move_s"] == pytest.approx(0.0, abs=0.2)
        assert row["dist_first_3s_ft"] == pytest.approx(30.0, abs=6.0)
        assert row["closing_speed_ft_s"] == pytest.approx(10.0, abs=2.0)

    def test_fielder_no_move(self):
        field = {f: (48.0, 58.0) for f in range(90)}
        row = fielder_play_metrics(field, contact_frame=0, fps=FPS, cfg=CFG)
        assert row["status"] == "no_first_move"

    def test_positioning_table(self):
        rows = [
            {"bat_side": "L", "positions": {"SS": (48.0, 60.0), "1B": None}},
            {"bat_side": "L", "positions": {"SS": (50.0, 62.0)}},
            {"bat_side": "R", "positions": {"SS": (44.0, 56.0)}},
        ]
        out = positioning_table(rows)
        assert len(out) == 2
        l = next(r for r in out if r["bat_side"] == "L")
        assert l["position"] == "SS" and l["n"] == 2
        assert l["mean_x_ft"] == pytest.approx(49.0, abs=0.1)


# ---------------------------------------------------------------- calibration + OCR path
class TestFromMatrixAndOcr:
    def test_from_matrix_round_trip(self):
        hom = Homography.from_matrix(HOM.matrix)
        assert np.allclose(hom.matrix, HOM.matrix)
        assert hom.to_field(img_of((45.0, 30.0))) == pytest.approx((45.0, 30.0), abs=1e-3)

    def test_track_clip_ocr_anchor(self, tmp_path):
        runner = linear_traj((24.0, 3.0), (80.0, 3.0), 25.0)
        clip = tmp_path / "play.mp4"
        write_synthetic_clip(clip, [runner], 60)

        class FakeReader:
            def read_frame(self, frame, video_time_s):
                # OCR is monkeypatched: wall = video time + 5400 (the +90 min offset)
                return type("A", (), {"wall_s": video_time_s + 5400.0, "raw_text": "x"})()

        tracks = track_clip(str(clip), CFG, clock_reader=FakeReader())
        assert tracks.wall0_s == pytest.approx(5400.0, abs=0.1)
        assert tracks.wall1_s == pytest.approx((59 / 30) + 5400.0, abs=0.1)

    def test_track_clip_ocr_failure_leaves_none(self, tmp_path):
        runner = linear_traj((24.0, 3.0), (80.0, 3.0), 25.0)
        clip = tmp_path / "play.mp4"
        write_synthetic_clip(clip, [runner], 60)

        class BadReader:
            def read_frame(self, frame, video_time_s):
                raise RuntimeError("ocr boom")

        tracks = track_clip(str(clip), CFG, clock_reader=BadReader())
        assert tracks.wall0_s is None and tracks.wall1_s is None
        assert tracks.n_frames == 60   # the clip itself is unaffected
