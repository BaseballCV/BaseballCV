import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from baseballcv.model import BaseDetector


class _FakeTensor:
    """Minimal stand-in for a torch tensor holding one box's coordinates."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [_FakeTensor(xyxy)]
        self.conf = float(conf)


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    """Emulates ultralytics conf thresholding and returns fixed boxes for filter testing."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, source, conf=0.5, device="cpu", verbose=False, **kwargs):
        kept = [b for b in self._boxes if b.conf >= conf]
        return [_FakeResults(kept)]


def _make_detector_with_boxes(boxes) -> BaseDetector:
    """Build a BaseDetector without downloading weights, injecting a stub YOLO."""
    detector = BaseDetector.__new__(BaseDetector)
    detector.model = _FakeYOLO(boxes)
    detector.device = "cpu"
    detector.model_path = "stub"
    detector.logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    return detector


class TestBaseDetectorFilter:
    """
    Unit tests for the aspect-ratio mitigation from issue #183, using stubbed
    model output — no weights or network required.

    Measured on broadcast footage (see #183): real bases measure up to ~3.9:1,
    rubber/lines 3.7-6.9:1 at low confidence, legs/uniforms 4.5-4.9:1.
    """

    def test_base_shallow_angle_passes(self):
        # Widest real base measured on broadcast footage (slide angle)
        detector = _make_detector_with_boxes([_FakeBox([100, 100, 493, 200], 0.85)])
        detections = detector.predict("stub.jpg")
        assert len(detections) == 1
        assert detections[0]["conf"] == 0.85
        assert detections[0]["aspect_ratio"] == pytest.approx(3.93, abs=0.01)

    def test_low_conf_rubber_filtered_by_conf(self):
        # Pitching rubber: 59x15 px at conf 0.21 (frame 120, issue #183).
        # Rubber hits are typically low-confidence, so the default conf=0.5 drops them.
        detector = _make_detector_with_boxes([_FakeBox([300, 600, 359, 615], 0.21)])
        assert detector.predict("stub.jpg") == []

    def test_foul_line_thin_box_filtered(self):
        detector = _make_detector_with_boxes([_FakeBox([0, 400, 400, 410], 0.30)])
        assert detector.predict("stub.jpg") == []

    def test_leg_box_filtered(self):
        # Runner's leg during a slide: 167x33 px at conf 0.80 (frame 335, issue #183)
        detector = _make_detector_with_boxes([_FakeBox([600, 400, 767, 433], 0.80)])
        assert detector.predict("stub.jpg") == []

    def test_ar_max_boundary_is_inclusive(self):
        # Exactly 4.2:1 survives (filter is <=)
        detector = _make_detector_with_boxes([_FakeBox([0, 0, 420, 100], 0.9)])
        assert len(detector.predict("stub.jpg")) == 1

    def test_ar_max_can_be_relaxed(self):
        detector = _make_detector_with_boxes([_FakeBox([600, 400, 767, 433], 0.80)])
        assert len(detector.predict("stub.jpg", ar_max=5.5)) == 1

    def test_zero_height_box_filtered(self):
        detector = _make_detector_with_boxes([_FakeBox([10, 10, 100, 10], 0.9)])
        assert detector.predict("stub.jpg") == []

    def test_mixed_boxes_only_base_survives(self):
        boxes = [
            _FakeBox([100, 100, 229, 134], 0.87),   # real base, 3.79:1
            _FakeBox([300, 600, 359, 615], 0.21),   # rubber, low conf
            _FakeBox([600, 400, 767, 433], 0.80),   # leg, thin box
        ]
        detector = _make_detector_with_boxes(boxes)
        detections = detector.predict("stub.jpg")
        assert len(detections) == 1
        assert detections[0]["conf"] == 0.87

    def test_high_conf_rubber_known_residual_limitation(self):
        # Known residual limitation (#183): a mound rubber hit at conf 0.68 with
        # a compact box survives both default filters. Only retraining with
        # labeled negatives fixes this class — here we pin the current behavior.
        detector = _make_detector_with_boxes([_FakeBox([300, 600, 331, 610], 0.68)])
        assert len(detector.predict("stub.jpg")) == 1

    def test_default_thresholds_match_documented_mitigations(self):
        assert BaseDetector.DEFAULT_CONF == 0.5
        assert BaseDetector.DEFAULT_AR_MAX == 4.2


class TestBaseDetectorIntegration:
    """
    Integration tests against the real vault-hosted weights.
    Requires network access (model download).
    """

    @pytest.fixture(scope="class")
    def detector(self):
        return BaseDetector(device="cpu")

    @pytest.mark.network
    def test_model_loads_via_alias(self, detector):
        # default weights are the dataset v2.1 retrain (base + distractor), see #183
        assert detector.model is not None
        assert detector.model.names == {0: "base", 1: "distractor"}

    @pytest.mark.network
    def test_predict_returns_list(self, detector, tmp_path):
        # Synthetic frame: the model is not expected to find a base here —
        # this validates the predict plumbing and return contract.
        frame = np.full((720, 1280, 3), (80, 140, 60), dtype=np.uint8)
        cv2.rectangle(frame, (500, 300), (700, 400), (255, 255, 255), -1)
        path = str(tmp_path / "synthetic.jpg")
        cv2.imwrite(path, frame)
        detections = detector.predict(path)
        assert isinstance(detections, list)
        for det in detections:
            assert set(det) == {"xyxy", "conf", "aspect_ratio"}
            assert 0.0 <= det["conf"] <= 1.0

    @pytest.mark.network
    def test_annotate_image_writes_output(self, detector, tmp_path):
        frame = np.full((720, 1280, 3), (80, 140, 60), dtype=np.uint8)
        path = str(tmp_path / "frame.jpg")
        cv2.imwrite(path, frame)
        output = detector.annotate(path, str(tmp_path / "annotated.jpg"))
        assert os.path.exists(output)
