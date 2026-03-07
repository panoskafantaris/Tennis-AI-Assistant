"""
Ensemble detector — merges candidates from multiple detection
strategies with Kalman-gated selection.

Key improvements over naive approach:
  - Kalman only initializes after seeing MOTION (not a static dot)
  - Candidates must move between consecutive frames to be accepted
  - If Kalman tracks a static object, it auto-resets
  - Re-acquisition after long gaps uses all candidates
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

from core.base import BaseDetector
from core.background_detector import BackgroundDetector
from core.hybrid import HybridDetector
from tracking.kalman import BallKalmanFilter
from config.settings import KALMAN

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]
_INIT_FRAMES = 2       # need this many moving detections to seed Kalman
_STATIC_THRESH = 10    # pixels — below this movement = static
_STATIC_LIMIT = 8      # static this many frames → Kalman reset


class EnsembleDetector(BaseDetector):
    """
    Multi-strategy ball detector with motion-validated Kalman.
    """

    def __init__(self):
        self._bg_det = BackgroundDetector()
        self._hybrid = HybridDetector()
        self._kalman = BallKalmanFilter()
        self._frame_count = 0

        # Motion validation state
        self._init_buffer: List[Tuple[int, int, float]] = []
        self._last_det: Optional[Tuple[int, int]] = None
        self._static_count = 0

    @property
    def window_size(self) -> int:
        return max(self._bg_det.window_size, self._hybrid.window_size)

    def set_background(self, frames: List[np.ndarray]) -> None:
        self._bg_det.set_background(frames)

    def predict(
        self, frames: List[np.ndarray],
    ) -> Detection:
        self._frame_count += 1

        # Collect raw candidates
        candidates: List[Tuple[int, int, float, str]] = []
        bg_det = self._bg_det.predict(frames)
        if bg_det is not None:
            candidates.append((*bg_det, "bgsub"))
        hybrid_det = self._hybrid.predict(frames)
        if hybrid_det is not None:
            candidates.append((*hybrid_det, "hybrid"))

        # Phase A: Kalman not yet initialized — need motion proof
        if not self._kalman.initialized:
            return self._handle_init_phase(candidates)

        # Phase B: Kalman active — predict + gate + update
        return self._handle_tracking_phase(candidates)

    def _handle_init_phase(
        self, candidates: List[Tuple[int, int, float, str]],
    ) -> Detection:
        """
        Before Kalman is seeded, collect candidates and verify
        they show MOTION (not a static logo dot).
        """
        if not candidates:
            self._init_buffer.clear()
            return None

        # Pick best candidate
        best = max(candidates, key=lambda c: _source_score(c))
        bx, by, score = best[0], best[1], best[2]

        # Check if it moved since last candidate
        if self._last_det is not None:
            dx = abs(bx - self._last_det[0])
            dy = abs(by - self._last_det[1])
            moved = (dx + dy) > _STATIC_THRESH
        else:
            moved = False

        self._last_det = (bx, by)

        if moved:
            self._init_buffer.append((bx, by, score))
        else:
            # Static — don't add, but don't clear either
            # (allow a static frame between moving ones)
            pass

        # Once we have enough moving detections, seed Kalman
        if len(self._init_buffer) >= _INIT_FRAMES:
            sx, sy, sc = self._init_buffer[-1]
            self._kalman.update(sx, sy)
            logger.info(f"Kalman initialized at ({sx},{sy}) "
                        f"after {self._frame_count} frames")
            self._init_buffer.clear()
            self._static_count = 0
            return (sx, sy, sc)

        return None

    def _handle_tracking_phase(
        self, candidates: List[Tuple[int, int, float, str]],
    ) -> Detection:
        """Kalman is active: predict → gate → update → detect static."""
        predicted_pos = self._kalman.predict()

        # Gate candidates against Kalman prediction
        gated = []
        for cx, cy, score, source in candidates:
            if self._kalman.gate(cx, cy):
                gated.append((cx, cy, score, source))

        if gated:
            gated.sort(key=lambda c: -_source_score(c))
            best = gated[0]
            bx, by = self._kalman.update(best[0], best[1])

            # Static tracking detection
            if self._last_det is not None:
                dx = abs(bx - self._last_det[0])
                dy = abs(by - self._last_det[1])
                if (dx + dy) < _STATIC_THRESH:
                    self._static_count += 1
                else:
                    self._static_count = 0

            self._last_det = (bx, by)

            # If Kalman has been tracking a static point, reset
            if self._static_count >= _STATIC_LIMIT:
                logger.info(f"Kalman reset — static at ({bx},{by})")
                self._kalman.reset()
                self._init_buffer.clear()
                self._static_count = 0
                self._last_det = None
                return None

            return (bx, by, best[2])

        # No gated candidates — try re-acquisition after long gap
        if candidates and self._kalman.frames_since_update > 15:
            best = max(candidates, key=lambda c: c[2])
            self._kalman.reset()
            self._init_buffer.clear()
            self._last_det = (best[0], best[1])
            # Don't immediately return — require motion proof again
            self._init_buffer.append((best[0], best[1], best[2]))
            return None

        # Fallback: Kalman prediction
        if predicted_pos is not None:
            conf = self._kalman.prediction_confidence()
            if conf >= KALMAN["min_confidence"]:
                return (predicted_pos[0], predicted_pos[1], conf)

        return None

    def reset(self) -> None:
        self._kalman.reset()
        self._init_buffer.clear()
        self._last_det = None
        self._static_count = 0
        self._frame_count = 0


def _source_score(c: tuple) -> float:
    """Score = priority × raw_score."""
    priority = {"bgsub": 2.0, "hybrid": 1.0}.get(c[3], 0.5)
    return priority * c[2]
