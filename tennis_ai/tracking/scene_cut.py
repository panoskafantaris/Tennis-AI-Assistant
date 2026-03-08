"""
Scene cut detector — detects hard cuts in highlight/broadcast video.

Uses histogram correlation between consecutive frames.
A sudden large change in the frame histogram signals a scene transition.

On detection, the pipeline should:
  - Rebuild background model from new scene frames
  - Reset Kalman filter, stationarity blacklist, and all tracking state
  - Re-calibrate court zone polygon
"""
import logging
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from config.settings import SCENE_CUT

logger = logging.getLogger(__name__)


class SceneCutDetector:
    """Detect hard scene cuts via histogram + pixel difference."""

    def __init__(self):
        self._hist_thresh = SCENE_CUT["hist_threshold"]
        self._pixel_thresh = SCENE_CUT["pixel_threshold"]
        self._cooldown = SCENE_CUT["cooldown_frames"]
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._frames_since_cut = 0
        self._cut_count = 0

    def check(self, frame: np.ndarray) -> bool:
        """
        Check if current frame is a scene cut.
        Returns True on the first frame of a new scene.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))

        hist = cv2.calcHist([small], [0], None, [64], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)

        is_cut = False
        self._frames_since_cut += 1

        if self._prev_hist is not None and self._prev_gray is not None:
            # Skip if in cooldown
            if self._frames_since_cut < self._cooldown:
                self._prev_gray = small
                self._prev_hist = hist
                return False

            # Method 1: histogram L1 distance
            hist_diff = float(np.sum(np.abs(hist - self._prev_hist)))

            # Method 2: pixel-level MSE
            pixel_diff = float(np.mean((small.astype(float) - self._prev_gray.astype(float)) ** 2))

            if hist_diff > self._hist_thresh or pixel_diff > self._pixel_thresh:
                is_cut = True
                self._frames_since_cut = 0
                self._cut_count += 1
                logger.info(
                    f"Scene cut #{self._cut_count} detected "
                    f"(hist={hist_diff:.3f}, pixel={pixel_diff:.1f})"
                )

        self._prev_gray = small
        self._prev_hist = hist
        return is_cut

    @property
    def total_cuts(self) -> int:
        return self._cut_count

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_hist = None
        self._frames_since_cut = 0