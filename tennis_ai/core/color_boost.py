"""
Color boost — scores ball candidates by HSV color match.

On blue hard courts (Qatar Open, Australian Open, US Open), the
yellow-green ball has extreme color contrast against the court.
This module provides a fast color-matching score that can rescue
candidates missed by motion-only detection.

Also provides a standalone color detector as a fallback when
both background subtraction and motion detection fail.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import COLOR_BOOST

logger = logging.getLogger(__name__)

Candidate = Tuple[int, int, float]


class ColorBallDetector:
    """
    Detect ball candidates using HSV color alone.

    Useful as a fallback when motion-based methods fail,
    e.g., during slow rallies or after scene cuts.
    """

    def __init__(self):
        self._hsv_lo = np.array(COLOR_BOOST["hsv_lower"], dtype=np.uint8)
        self._hsv_hi = np.array(COLOR_BOOST["hsv_upper"], dtype=np.uint8)
        self._min_area = COLOR_BOOST["min_area"]
        self._max_area = COLOR_BOOST["max_area"]
        self._min_circ = COLOR_BOOST["min_circularity"]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray, k: int = 5) -> List[Candidate]:
        """Find up to k ball candidates by color matching."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lo, self._hsv_hi)

        # Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: List[Candidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self._min_area <= area <= self._max_area):
                continue

            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area / (perim ** 2)) if perim > 1 else 0
            if circ < self._min_circ:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Score based on color purity and circularity
            roi = hsv[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
            if roi.size == 0:
                continue
            sat = float(roi[:, :, 1].mean()) / 255.0
            score = circ * 0.5 + sat * 0.3 + (1.0 - area / self._max_area) * 0.2
            candidates.append((cx, cy, float(score)))

        candidates.sort(key=lambda c: -c[2])
        return candidates[:k]


def color_score_at(frame: np.ndarray, x: int, y: int, r: int = 5) -> float:
    """
    Score how well a point matches the ball color.
    Returns 0.0-1.0 where 1.0 = perfect ball color match.
    """
    h, w = frame.shape[:2]
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    x1, x2 = max(0, x - r), min(w, x + r + 1)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo = np.array(COLOR_BOOST["hsv_lower"], dtype=np.uint8)
    hi = np.array(COLOR_BOOST["hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    return float(mask.sum() / 255) / max(mask.size, 1)