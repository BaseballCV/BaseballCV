from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class ScaleEstimate:
    """Pixels-per-foot estimate with the method used and a rough 0-1 confidence."""

    px_per_ft: float
    method: str
    confidence: float


def box_center(box: Sequence[float]) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def scale_from_base_box(box: Sequence[float], base_size_in: float, confidence: float) -> ScaleEstimate:
    """Scale from one base's pixel width (widest error band: camera angle inflates/shrinks it)."""
    width_px = float(box[2] - box[0])
    return ScaleEstimate(width_px / (base_size_in / 12.0), "single_base", confidence)


def scale_from_two_bases(c1: Point, c2: Point, spacing_ft: float, confidence: float) -> ScaleEstimate:
    """Scale from the known 1B-2B spacing (average along the baseline; ignores perspective)."""
    dist_px = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
    return ScaleEstimate(dist_px / spacing_ft, "two_bases", confidence)


def scale_from_height(height_px: float, height_in: float, confidence: float,
                      visible_fraction: float = 0.93) -> ScaleEstimate:
    """Scale at the runner's own depth from standing pixel height (nose-to-ankle ~ 93% of stature)."""
    return ScaleEstimate(height_px / (visible_fraction * height_in / 12.0), "runner_height", confidence)


def best_scale(estimates: Sequence[ScaleEstimate]) -> Tuple[Optional[ScaleEstimate], Optional[float]]:
    """Pick the highest-confidence estimate; also return the percent spread against the others.

    The spread (max-min over mean) is the honest signal of how much the geometry can be trusted.
    """
    if not estimates:
        return None, None
    best = max(estimates, key=lambda e: e.confidence)
    if len(estimates) == 1:
        return best, None
    vals = np.array([e.px_per_ft for e in estimates])
    return best, float((vals.max() - vals.min()) / vals.mean() * 100.0)


class Homography:
    """Image -> field-plane mapping from four or more user-supplied correspondences (tier A)."""

    def __init__(self, image_pts: Sequence[Point], field_pts_ft: Sequence[Point]) -> None:
        import cv2

        if len(image_pts) < 4 or len(image_pts) != len(field_pts_ft):
            raise ValueError("Homography needs >= 4 matching image/field point pairs")
        src = np.asarray(image_pts, dtype=np.float32)
        dst = np.asarray(field_pts_ft, dtype=np.float32)
        self.matrix, _ = cv2.findHomography(src, dst, 0)
        if self.matrix is None:
            raise ValueError("Could not compute homography from the supplied points")

    def to_field(self, pt: Point) -> Point:
        v = self.matrix @ np.array([pt[0], pt[1], 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))


class BaselineAxis:
    """Signed distance of image points along the 1B->2B line, in feet, measured from first base.

    Either backed by a homography (field coordinates, feet) or by an image-space axis and a scalar scale.
    """

    def __init__(self, first_base: Point, second_base: Optional[Point] = None,
                 direction: Optional[Point] = None, px_per_ft: Optional[float] = None,
                 homography: Optional[Homography] = None, method: str = "") -> None:
        self.first_base = first_base
        self.homography = homography
        self.px_per_ft = px_per_ft
        if second_base is not None:
            d = np.array([second_base[0] - first_base[0], second_base[1] - first_base[1]], dtype=float)
        elif direction is not None:
            d = np.array(direction, dtype=float)
        else:
            raise ValueError("BaselineAxis needs a second base or a direction")
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Degenerate baseline axis")
        self.unit = d / n
        self.method = method
        if homography is not None:
            f1 = np.array(homography.to_field(first_base))
            if second_base is not None:
                f2 = np.array(homography.to_field(second_base))
            else:  # image-space direction: map a point along it onto the field plane
                probe = (first_base[0] + self.unit[0] * 50.0, first_base[1] + self.unit[1] * 50.0)
                f2 = np.array(homography.to_field(probe))
            fd = f2 - f1
            self._field_unit = fd / np.linalg.norm(fd)
            self._field_origin = f1
        elif px_per_ft is None or px_per_ft <= 0:
            raise ValueError("px_per_ft is required without a homography")

    def project(self, pt: Point) -> float:
        """Feet along the baseline from first base (negative = behind the bag)."""
        if self.homography is not None:
            f = np.array(self.homography.to_field(pt)) - self._field_origin
            return float(f @ self._field_unit)
        rel = np.array([pt[0] - self.first_base[0], pt[1] - self.first_base[1]])
        return float(rel @ self.unit / self.px_per_ft)


def stable_boxes(per_frame: Dict[int, list], min_frames: int, cluster_frac: float) -> list:
    """Collapse per-frame base detections inside one shot into static base boxes.

    Bases do not move within a fixed shot, so detections are greedily clustered by center and each
    cluster is reduced to its median box. Clusters seen in fewer than ``min_frames`` frames are
    dropped (one-off false positives). Returns [(x1, y1, x2, y2, n_frames, mean_conf)] sorted by n_frames.
    """
    clusters: list = []
    for frame in sorted(per_frame):
        for det in per_frame[frame]:
            box, conf = det[:4], det[4]
            cx, cy = box_center(box)
            w = box[2] - box[0]
            for cl in clusters:
                ccx, ccy = box_center(np.median(np.array(cl["boxes"]), axis=0))
                if np.hypot(cx - ccx, cy - ccy) <= cluster_frac * max(w, 1.0):
                    cl["boxes"].append(box)
                    cl["confs"].append(conf)
                    cl["frames"].add(frame)
                    break
            else:
                clusters.append({"boxes": [box], "confs": [conf], "frames": {frame}})
    out = []
    for cl in clusters:
        if len(cl["frames"]) < min_frames:
            continue
        med = np.median(np.array(cl["boxes"]), axis=0)
        out.append((float(med[0]), float(med[1]), float(med[2]), float(med[3]),
                    len(cl["frames"]), float(np.mean(cl["confs"]))))
    return sorted(out, key=lambda b: -b[4])
