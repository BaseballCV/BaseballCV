from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

from baseballcv.utilities import BaseballCVLogger


class BaseDetector:
    """Wrapper around the `base_detector` YOLOv11s model for detecting bases in broadcast feeds.

    The underlying model was trained on labeled bases only, so it can false-positive on other
    white field objects — the pitching rubber and foul lines at low confidence, and players'
    legs/uniforms at shallow slide angles (see issue #183). This wrapper applies the two
    documented mitigations by default:

    1. Confidence threshold of 0.5 — rubber/line hits rarely exceed ~0.35 on wide broadcast frames.
    2. Box aspect-ratio filter of 4.2 — bases at broadcast camera angles measure up to ~3.9:1,
       while the false-positive classes start at ~4.5:1.

    Example:
        >>> detector = BaseDetector()
        >>> detections = detector.predict("frame.jpg")
        >>> detector.annotate("clip.mp4", "annotated.mp4")
    """

    DEFAULT_CONF = 0.5
    DEFAULT_AR_MAX = 4.2

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        """
        Initialize the base detector.

        Args:
            model_path (Optional[str], optional): Path to YOLO weights. Defaults to the
                vault-hosted `base_detector` weights resolved through LoadTools.
            device (str, optional): Inference device. Defaults to "cpu".
        """
        from ultralytics import YOLO  # lazy import — heavy dependency

        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)
        if model_path is None:
            from baseballcv.functions import LoadTools

            model_path = LoadTools().load_model("base_detector")
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        self.logger.info(f"BaseDetector initialized from {model_path} on device={device}")

    @staticmethod
    def _keep_box(x1: float, y1: float, x2: float, y2: float, ar_max: float) -> bool:
        """Aspect-ratio filter: drop thin boxes (rubber, foul lines, limbs)."""
        w, h = x2 - x1, y2 - y1
        return h > 0 and w / h <= ar_max

    def predict(
        self,
        source: str,
        conf: float = DEFAULT_CONF,
        ar_max: float = DEFAULT_AR_MAX,
        **pred_kwargs,
    ) -> List[Dict[str, float]]:
        """
        Run filtered detection on a single image.

        Args:
            source (str): Path, URL, or BGR numpy array of an image.
            conf (float, optional): Confidence threshold. Defaults to 0.5.
            ar_max (float, optional): Maximum box aspect ratio (width/height). Defaults to 4.2.
            **pred_kwargs: Extra keyword arguments passed to `ultralytics.YOLO.predict`.

        Returns:
            List[Dict[str, float]]: One dict per surviving detection, with keys
            `xyxy` (x1, y1, x2, y2), `conf`, and `aspect_ratio`.
        """
        # v2.1+ weights carry a "distractor" class that must never be reported as a base.
        names = getattr(self.model, "names", None) or {}
        base_ids = [i for i, n in names.items() if str(n).lower() == "base"]
        if base_ids and len(names) > 1:
            pred_kwargs.setdefault("classes", base_ids)
        result = self.model.predict(source, conf=conf, device=self.device, verbose=False, **pred_kwargs)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if not self._keep_box(x1, y1, x2, y2, ar_max):
                self.logger.debug(f"Filtered thin box at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) — likely not a base")
                continue
            w, h = x2 - x1, y2 - y1
            detections.append({
                "xyxy": (x1, y1, x2, y2),
                "conf": float(box.conf),
                "aspect_ratio": w / h,
            })
        label = source if isinstance(source, str) else "<frame>"  # never dump a frame array into the log
        self.logger.info(f"{len(detections)} base(s) detected in {label} (conf >= {conf}, w/h <= {ar_max})")
        return detections

    def annotate(
        self,
        source: str,
        output_path: Optional[str] = None,
        conf: float = DEFAULT_CONF,
        ar_max: float = DEFAULT_AR_MAX,
        fps: int = 30,
    ) -> str:
        """
        Draw filtered detections on an image or video and save the result.

        Args:
            source (str): Path or URL of an image (.jpg/.png) or video (.mp4/.mov/.avi/.mkv).
            output_path (Optional[str], optional): Output path. Defaults to
                `annotated_<input stem>` next to the input.
            conf (float, optional): Confidence threshold. Defaults to 0.5.
            ar_max (float, optional): Maximum box aspect ratio (width/height). Defaults to 4.2.
            fps (int, optional): Frame rate for video output. Defaults to 30.

        Returns:
            str: Path of the annotated output file.
        """
        import cv2

        is_video = os.path.splitext(source)[1].lower() in {".mp4", ".mov", ".avi", ".mkv"}
        if output_path is None:
            stem = os.path.splitext(os.path.basename(source))[0]
            output_path = f"annotated_{stem}{'.mp4' if is_video else '.jpg'}"

        def _draw(frame, detections):
            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                label = f"base {det['conf']:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 90, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 0), 1, cv2.LINE_AA)
            return frame

        if not is_video:
            detections = self.predict(source, conf=conf, ar_max=ar_max)
            frame = cv2.imread(source)
            cv2.imwrite(output_path, _draw(frame, detections))
            self.logger.info(f"Annotated image saved to {output_path}")
            return output_path

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.predict(frame, conf=conf, ar_max=ar_max)
            frame = _draw(frame, detections)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        cap.release()
        if writer is None:
            raise IOError(f"No frames decoded from video source: {source}")
        writer.release()
        self.logger.info(f"Annotated video saved to {output_path}")
        return output_path
