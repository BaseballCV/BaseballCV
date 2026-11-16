"""Burned-in timestamp reading and the video-time -> wall-clock map.

Findings from the Magallanes feed (scripts/spike_s1*.py) that shape this module:
- The top-right overlay parses at ~95% with rapidocr on a 4x-upscaled Otsu crop.
- The recorder's clock does NOT drift: within any short window the overlay
  advances at exactly 1 s/s. Globally, however, the file drops ~30 s of frames
  over 3.6 h AND the overlay oscillates by exactly +-3600 s in a band of the
  recording (an NTP/DST toggle on the recorder). A single linear fit is
  therefore wrong: we keep per-anchor reads, fold the +-1 h display state, and
  map video time piecewise-linearly between anchors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Sequence

import cv2
import numpy as np

from baseballcv.utilities import BaseballCVLogger

from .config import GameSyncConfig

_TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)?", re.I)


def parse_timestamp(text: str) -> Optional[int]:
    """Seconds-of-day from an overlay string like '06:33:14 PM' or '08:12:49PM'."""
    m = _TS_RE.search(text or "")
    if not m:
        return None
    hh, mi, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ap = (m.group(4) or "").upper()
    if ap == "PM" and hh != 12:
        hh += 12
    if ap == "AM" and hh == 12:
        hh = 0
    if hh > 23 or mi > 59 or ss > 59:
        return None
    return hh * 3600 + mi * 60 + ss


@dataclass
class ClockAnchor:
    video_time_s: float
    wall_s: float        # folded seconds-of-day
    raw_text: str


class BurnedClockReader:
    """Reads the overlay timestamp from sampled frames and builds the clock map."""

    def __init__(self, config: Optional[GameSyncConfig] = None, logger=None) -> None:
        self.config = config or GameSyncConfig()
        self.logger = logger or BaseballCVLogger.get_logger(self.__class__.__name__)
        self._ocr = None  # lazy: rapidocr is an optional dependency

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "BurnedClockReader needs the optional OCR extra: pip install rapidocr-onnxruntime"
                ) from exc
            self._ocr = RapidOCR()
        return self._ocr

    # ------------------------------------------------------------------
    def _read_one(self, frame: np.ndarray) -> tuple[Optional[str], Optional[int]]:
        h, w = frame.shape[:2]
        x0 = int(w * self.config.ocr_crop_x0_frac)
        y1 = int(h * self.config.ocr_crop_y1_frac)
        crop = frame[0:y1, x0:w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, None, fx=self.config.ocr_upscale, fy=self.config.ocr_upscale,
                         interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.mean() > 127:  # keep text white-on-black
            bw = 255 - bw
        result, _ = self._get_ocr()(bw)
        text = " ".join(r[1] for r in result) if result else ""
        return text.strip(), parse_timestamp(text)

    def read_frame(self, frame: np.ndarray, video_time_s: float) -> Optional[ClockAnchor]:
        text, secs = self._read_one(frame)
        if secs is None:
            return None
        return ClockAnchor(video_time_s=video_time_s, wall_s=float(secs), raw_text=text)

    # ------------------------------------------------------------------
    def build_anchors(self, video_path: str, duration_s: Optional[float] = None) -> list[ClockAnchor]:
        """Sample the whole file every ocr_step_s; retry once at +1 s on failure."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if duration_s is None:
            duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        anchors: list[ClockAnchor] = []
        t = 0.0
        while t < duration_s:
            for offset in (0.0, self.config.ocr_retry_offset_s):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + offset) * fps))
                ok, frame = cap.read()
                anchor = self.read_frame(frame, t) if ok else None
                if anchor is not None:
                    anchors.append(anchor)
                    break
            else:
                self.logger.warning(f"no clock read at video {t:.0f}s")
            t += self.config.ocr_step_s
        cap.release()
        self.logger.info(f"{len(anchors)} clock anchors from {video_path} "
                         f"({100 * len(anchors) / max(1, duration_s / self.config.ocr_step_s):.0f}% of samples)")
        return anchors


@dataclass
class ClockMap:
    """Piecewise-linear video-time -> wall-clock map with the +-1 h state folded."""
    game_date: date
    video_times: np.ndarray      # sorted anchor video times
    wall_times: np.ndarray       # folded wall seconds (absolute, may exceed 24 h)
    base_wall_s: float           # wall at video t=0 (extrapolated)

    @classmethod
    def from_anchors(cls, anchors: Sequence[ClockAnchor], game_date: date,
                     config: Optional[GameSyncConfig] = None) -> "ClockMap":
        cfg = config or GameSyncConfig()
        if len(anchors) < 3:
            raise ValueError("need >= 3 clock anchors")
        vs = np.array([a.video_time_s for a in anchors])
        ws = np.array([a.wall_s for a in anchors])
        # --- fold the recorder's +-1 h display oscillation ---
        # An anchor whose wall time jumps ~ +-clock_fold_s vs the linear trend is
        # in the wrong display state; bring it back.
        # Start from the first anchor, walk forward, compare each delta to the
        # video delta; correct by the nearest multiple of the fold period.
        w = ws.copy()
        # robust reference slope from the median of near-unity deltas
        dv = np.diff(vs)
        dw = np.diff(ws)
        good = np.abs(dw - dv) < cfg.clock_jump_s
        slope = float(np.median(dw[good] / dv[good])) if good.any() else 1.0
        pred = w[0] + slope * (vs - vs[0])
        for i in range(len(w)):
            k = round((pred[i] - w[i]) / cfg.clock_fold_s)
            if k:
                w[i] += k * cfg.clock_fold_s
        # re-fold once more against the corrected trend (handles consecutive flips)
        coef = np.polyfit(vs, w, 1)
        pred = np.polyval(coef, vs)
        for i in range(len(w)):
            k = round((pred[i] - w[i]) / cfg.clock_fold_s)
            if k:
                w[i] += k * cfg.clock_fold_s
        base = float(w[0] - vs[0])  # wall at video t = 0
        return cls(game_date=game_date, video_times=vs, wall_times=w, base_wall_s=base)

    # ------------------------------------------------------------------
    def wall_at(self, video_time_s: float) -> float:
        """Absolute wall seconds (may exceed 24 h into the next day)."""
        return float(np.interp(video_time_s, self.video_times, self.wall_times))

    def video_at(self, wall_s: float) -> float:
        """Inverse map: wall seconds -> video time."""
        order = np.argsort(self.wall_times)
        return float(np.interp(wall_s, self.wall_times[order], self.video_times[order]))

    def datetime_at(self, video_time_s: float) -> datetime:
        base = datetime.combine(self.game_date, datetime.min.time())
        return base + timedelta(seconds=self.wall_at(video_time_s))

    def wall_from_db_time(self, hhmmss: str) -> float:
        """DB HH:MM:SS -> absolute wall seconds on the game date."""
        h, m, s = (int(p) for p in hhmmss.split(":")[:3])
        return h * 3600 + m * 60 + s

    def diagnostics(self) -> dict:
        dv = np.diff(self.video_times)
        dw = np.diff(self.wall_times)
        slope = dw / dv
        return {
            "n_anchors": int(len(self.video_times)),
            "median_slope": float(np.median(slope)),
            "max_seg_dev_s": float(np.max(np.abs(dw - dv))),
            "span_s": float(self.video_times[-1] - self.video_times[0]),
            "base_wall_s": self.base_wall_s,
        }
