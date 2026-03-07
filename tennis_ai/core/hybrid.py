"""
Hybrid ball detector — motion + color + shape analysis.
Works without pre-trained weights.

Pipeline per 3-frame window:
  1. Frame differencing -> isolate moving pixels
  2. HSV color filter   -> keep yellow-green ball range
  3. Combine masks (OR logic with scoring, not strict AND)
  4. Contour analysis   -> filter by area + circularity
  5. Pick best candidate
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base import BaseDetector
from config.settings import HYBRID


class HybridDetector(BaseDetector):
    """Motion + color + circularity ball detector. No GPU needed."""

    _HSV_LO = np.array(HYBRID["hsv_lower"], dtype=np.uint8)
    _HSV_HI = np.array(HYBRID["hsv_upper"], dtype=np.uint8)

    @property
    def window_size(self) -> int:
        return 3

    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        if len(frames) < 3:
            return None

        f1, f2, f3 = frames[-3], frames[-2], frames[-1]
        h, w = f3.shape[:2]

        min_area = max(3, int(np.pi * (h * HYBRID["min_radius_frac"]) ** 2))
        max_area = int(np.pi * (h * HYBRID["max_radius_frac"]) ** 2)

        color_m  = self._color_mask(f3)
        motion_m = self._motion_mask(f1, f2, f3)

        # Use OR logic: accept pixels that match color OR motion
        # Then score candidates by how many signals they match
        combined = cv2.bitwise_or(color_m, motion_m)

        # Smaller kernel — preserves tiny ball blobs
        ksize = HYBRID.get("morph_kernel", 3)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ksize, ksize),
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        best: Optional[Tuple[int, int, float]] = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            circ = self._circularity(cnt)
            if circ < HYBRID["min_circularity"]:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Score: circularity + bonus if both color AND motion match
            has_color = color_m[cy, cx] > 0
            has_motion = motion_m[cy, cx] > 0
            bonus = 0.3 if (has_color and has_motion) else 0.0
            score = circ + bonus

            if best is None or score > best[2]:
                best = (cx, cy, float(score))
        return best

    @classmethod
    def _color_mask(cls, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, cls._HSV_LO, cls._HSV_HI)

    @staticmethod
    def _motion_mask(
        f1: np.ndarray, f2: np.ndarray, f3: np.ndarray,
    ) -> np.ndarray:
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2, f3)]
        d1 = cv2.absdiff(gray[0], gray[1])
        d2 = cv2.absdiff(gray[1], gray[2])
        diff = cv2.max(d1, d2)
        _, motion = cv2.threshold(
            diff, HYBRID["diff_thresh"], 255, cv2.THRESH_BINARY,
        )
        return motion

    @staticmethod
    def _circularity(contour) -> float:
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, True)
        if area < 1 or perim < 1:
            return 0.0
        return 4 * np.pi * area / (perim * perim)
