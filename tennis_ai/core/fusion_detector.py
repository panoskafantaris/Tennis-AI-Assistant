"""
Fusion detector — TrackNet V3 as primary, ensemble as fallback.

Strategy per frame:
  1. Run TrackNet V3 (learned ball appearance, ignores players)
  2. If V3 confident → use directly (no blue-surround needed)
  3. If V3 misses → fall back to ensemble (has blue-surround safety)
  4. If both miss for ≤ coast_frames → use Kalman prediction
  5. If gap exceeds coast_frames → report LOST, reset Kalman

This captures the best of both worlds:
  - V3's learned features handle near-player and fast-motion cases
  - Ensemble catches cases V3 misses (occlusion, edge of frame)
  - Kalman fills 3-5 frame micro-gaps for smooth trajectories
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.base import BaseDetector
from config.settings import V3, BGSUB, FUSION

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]


class FusionDetector(BaseDetector):
    """TrackNet V3 primary + ensemble fallback + Kalman coast."""

    def __init__(self, v3_weights: Path = None):
        from core.tracknet_v3 import TrackNetV3Detector
        from core.ensemble_detector import EnsembleDetector
        from tracking.kalman import BallKalmanFilter

        self._v3 = TrackNetV3Detector(v3_weights)
        self._ensemble = EnsembleDetector()
        self._kalman = BallKalmanFilter()

        self._coast_max = FUSION["coast_frames"]
        self._v3_min_conf = FUSION["v3_min_confidence"]
        self._coast_min_conf = FUSION["coast_min_confidence"]

        self._frames_since_det = 0
        self._frame_count = 0
        self._v3_hits = 0
        self._ens_hits = 0
        self._coast_hits = 0

    @property
    def window_size(self) -> int:
        return max(self._v3.window_size, self._ensemble.window_size)

    def set_background(self, frames: List[np.ndarray]) -> None:
        """Set background for both V3 and ensemble."""
        self._v3.set_background(frames)
        self._ensemble.set_background(frames)

    def set_court_mask(self, mask: np.ndarray) -> None:
        """Pass court mask to ensemble for spatial filtering."""
        self._ensemble.set_court_mask(mask)

    def predict(self, frames: List[np.ndarray]) -> Detection:
        self._frame_count += 1

        # --- Tier 1: TrackNet V3 (primary) ---
        v3_det = self._try_v3(frames)
        if v3_det is not None:
            self._accept(v3_det, "v3")
            return v3_det

        # --- Tier 1b: Ensemble fallback ---
        ens_det = self._ensemble.predict(frames)
        if ens_det is not None:
            self._accept(ens_det, "ensemble")
            return ens_det

        # --- Tier 2: Kalman coast-through ---
        self._frames_since_det += 1
        if (self._kalman.initialized
                and self._frames_since_det <= self._coast_max):
            pred = self._kalman.predict()
            conf = self._kalman.prediction_confidence()
            if pred is not None and conf >= self._coast_min_conf:
                self._coast_hits += 1
                return (pred[0], pred[1], conf)

        # --- Lost: gap too long, reset ---
        if self._frames_since_det > self._coast_max:
            self._kalman.reset()
            self._ensemble.reset()
        return None

    def _try_v3(self, frames: List[np.ndarray]) -> Detection:
        """Run V3 and filter by minimum confidence."""
        if len(frames) < self._v3.window_size:
            return None
        det = self._v3.predict(frames)
        if det is not None and det[2] >= self._v3_min_conf:
            return det
        return None

    def _accept(self, det: Detection, source: str) -> None:
        """Update Kalman with accepted detection."""
        x, y, conf = det
        self._kalman.update(x, y)
        self._frames_since_det = 0
        if source == "v3":
            self._v3_hits += 1
        else:
            self._ens_hits += 1

    def reset(self) -> None:
        """Full reset — call on scene cuts."""
        self._kalman.reset()
        self._ensemble.reset()
        self._frames_since_det = 0

    def log_stats(self) -> None:
        """Log detection source breakdown."""
        total = self._v3_hits + self._ens_hits + self._coast_hits
        if total == 0:
            return
        logger.info(
            f"Fusion stats: V3={self._v3_hits} "
            f"({self._v3_hits/total*100:.0f}%) "
            f"Ensemble={self._ens_hits} "
            f"({self._ens_hits/total*100:.0f}%) "
            f"Coast={self._coast_hits} "
            f"({self._coast_hits/total*100:.0f}%)"
        )