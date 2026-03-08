"""
Player mask — detects players as large foreground blobs.

Players are the dominant moving objects on court. By detecting them
as large contours in the background diff, we can reject small ball
candidates that fall inside player bounding boxes.

This eliminates the most common false positive: the detector locking
onto a player's shirt, arm, racket, or shoes instead of the ball.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import PLAYER_MASK

logger = logging.getLogger(__name__)


class PlayerMask:
    """Detect players and provide exclusion zones for ball detection."""

    def __init__(self):
        self._zones: List[Tuple[int, int, int, int]] = []  # (cx, cy, half_w, half_h)
        self._margin = PLAYER_MASK["margin_px"]
        self._min_area = PLAYER_MASK["min_player_area"]

    def update(self, bg_diff: np.ndarray) -> None:
        """
        Find large foreground blobs (players) in background diff.
        bg_diff: grayscale abs(frame - background)
        """
        thresh = PLAYER_MASK.get("bg_thresh", 35)
        _, binary = cv2.threshold(bg_diff, thresh, 255, cv2.THRESH_BINARY)

        # Moderate dilation to merge nearby fragments into single player blob
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        self._zones.clear()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            # Expanded half-dimensions with margin
            hw = w // 2 + self._margin
            hh = h // 2 + self._margin
            self._zones.append((cx, cy, hw, hh))

    def is_near_player(self, x: int, y: int) -> bool:
        """Check if (x, y) falls inside any player exclusion zone."""
        for cx, cy, hw, hh in self._zones:
            if abs(x - cx) < hw and abs(y - cy) < hh:
                return True
        return False

    def distance_to_nearest(self, x: int, y: int) -> float:
        """Distance from (x, y) to nearest player zone edge. 0 if inside."""
        if not self._zones:
            return 999.0
        min_dist = 999.0
        for cx, cy, hw, hh in self._zones:
            dx = max(0, abs(x - cx) - hw)
            dy = max(0, abs(y - cy) - hh)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            min_dist = min(min_dist, dist)
        return min_dist

    @property
    def zones(self) -> List[Tuple[int, int, int, int]]:
        return self._zones

    def reset(self) -> None:
        self._zones.clear()