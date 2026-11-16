"""Base localization and field homography for the cenital camera.

Spike findings (scripts/spike_s2*.py):
- Bases are 1.5-2 px white squares at 854x480; 2B is a clean bright blob,
  1B sits on the foul line, home plate is inside the batter's-box chalk,
  and 3B can be occluded by a fielder for whole minutes.
- The fisheye makes the image-space diamond a KITE (home-2B diagonal 92 px,
  1B-3B diagonal 344 px): a square prior is wrong and a homography, not a
  similarity, is the right image->field map.
- The on-deck circles (~12 px bright ovals in foul ground) are the main
  false positives; they sit low in the frame, outside the diamond prior box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from baseballcv.utilities import BaseballCVLogger
from baseballcv.functions.utils.baseball_utils.steal_analysis.geometry import Homography

from .config import GameSyncConfig

# Field frame (feet): home=(0,0), first=(90,0), second=(63.64,63.64), third=(0,90)
FIELD_POINTS = {
    "home": (0.0, 0.0),
    "first": (90.0, 0.0),
    "second": (90.0 / np.sqrt(2.0), 90.0 / np.sqrt(2.0)),
    "third": (0.0, 90.0),
}


@dataclass
class BaseLocations:
    pixels: Dict[str, Tuple[float, float]]       # image (x, y) per base
    method: str
    confidence: float
    diagnostics: dict


def median_frame(video_path: str, center_s: float, config: Optional[GameSyncConfig] = None,
                 logger=None) -> np.ndarray:
    """Median gray image over a short window (suppresses moving players)."""
    cfg = config or GameSyncConfig()
    logger = logger or BaseballCVLogger.get_logger("median_frame")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for k in range(cfg.calib_n_frames):
        t = center_s + (k - cfg.calib_n_frames // 2) * (cfg.calib_median_window_s / cfg.calib_n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < cfg.calib_n_frames // 2:
        raise RuntimeError(f"could not decode enough frames around t={center_s}s")
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    logger.info(f"median frame at t={center_s}s from {len(frames)} frames")
    return med


def _bright_blob(med: np.ndarray, center: Tuple[float, float], radius: int,
                 contrast_min: float) -> Tuple[Optional[Tuple[float, float]], float]:
    """Brightness-over-local-mean blob centroid near a prior point."""
    x0, y0 = int(center[0]), int(center[1])
    h, w = med.shape
    if not (radius <= x0 < w - radius and radius <= y0 < h - radius):
        return None, 0.0
    win = med[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1].astype(np.float32)
    resp = win - cv2.blur(win, (7, 7))
    resp[resp < contrast_min] = 0
    if resp.max() < contrast_min:
        return None, 0.0
    ys, xs = np.nonzero(resp > resp.max() * 0.5)
    wgt = resp[ys, xs]
    return ((float((xs * wgt).sum() / wgt.sum()) + x0 - radius,
             float((ys * wgt).sum() / wgt.sum()) + y0 - radius), float(resp.max()))


def locate_bases_from_priors(med: np.ndarray, priors: Dict[str, Tuple[float, float]],
                             config: Optional[GameSyncConfig] = None) -> BaseLocations:
    """Detect each base near a prior pixel position (spike-verified values or manual).

    Priors may come from a venue calibration file or a previous run; the detector
    refines them to sub-pixel bright-blob centroids.
    """
    cfg = config or GameSyncConfig()
    pixels, missing = {}, []
    for name, prior in priors.items():
        pos, score = _bright_blob(med, prior, cfg.search_radius_px, cfg.blob_contrast_min)
        if pos is None:
            missing.append(name)
        else:
            pixels[name] = pos
    confidence = len(pixels) / max(1, len(priors))
    return BaseLocations(pixels=pixels, method="prior+bright_blob",
                         confidence=confidence,
                         diagnostics={"missing": missing})


def build_field_homography(locs: BaseLocations) -> Tuple[Homography, dict]:
    """Homography from detected base pixels to the field frame (feet).

    With all four bases the four correspondences are exact. With three, the
    missing base is mirrored through the midpoint of the known diagonal (an
    approximation under fisheye — the homography absorbs most of it but the
    diagnostics flag the reduced confidence).
    """
    pts = locs.pixels
    needed = ("home", "first", "second", "third")
    have = [n for n in needed if n in pts]
    if len(have) < 3:
        raise ValueError(f"need >= 3 bases for calibration, got {sorted(have)}")
    src = [pts[n] for n in have]
    dst = [FIELD_POINTS[n] for n in have]
    notes = {}
    if len(have) == 3:
        missing = [n for n in needed if n not in have][0]
        diagonals = [{"pair": ("home", "second"), "missing": "third", "opposite": "first"},
                     {"pair": ("first", "third"), "missing": "second", "opposite": "home"}]
        center = None
        for d in diagonals:
            if set(d["pair"]) <= set(have):
                a, b = d["pair"]
                center = (np.array(pts[a]) + np.array(pts[b])) / 2
                break
        if center is None:
            raise ValueError("3-base homography needs a known diagonal (home+2B or 1B+3B)")
        if missing != d["missing"]:
            raise ValueError(f"cannot derive {missing} from the known diagonal")
        # mirror the diagonal opposite of the missing base through the diamond center
        src.append(tuple(2 * center - np.array(pts[d["opposite"]])))
        dst.append(FIELD_POINTS[missing])
        notes["mirrored"] = missing
    hom = Homography(src, dst)
    notes["n_bases"] = len(have)
    return hom, notes


def fisheye_diagnostics(locs: BaseLocations) -> dict:
    """Report image-space diamond asymmetry (the kite effect) — reported, not corrected."""
    pts = locs.pixels
    out = {}
    if {"home", "second"} <= set(pts):
        out["home_2b_px"] = float(np.hypot(pts["second"][0] - pts["home"][0],
                                           pts["second"][1] - pts["home"][1]))
    if {"first", "third"} <= set(pts):
        out["first_3b_px"] = float(np.hypot(pts["third"][0] - pts["first"][0],
                                            pts["third"][1] - pts["first"][1]))
    if "home_2b_px" in out and "first_3b_px" in out:
        # both diagonals are 127.28 ft: their px ratio IS the fisheye distortion measure
        out["kite_ratio"] = out["first_3b_px"] / out["home_2b_px"]
    return out
