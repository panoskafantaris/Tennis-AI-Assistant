"""
Court zone filter — restricts detections to the playing surface.

Uses white court sidelines to build a perspective-correct trapezoid:
narrow at the far end (top), wide at the near end (bottom).
This excludes spectator stands, umpire chairs, and ball baskets
even though they're adjacent to the court.

Detection: find longest diagonal white lines (sidelines), then
compute the bounding trapezoid from their slopes.
"""
import logging
from typing import Optional

import cv2
import numpy as np

from config.settings import COURT_ZONE

logger = logging.getLogger(__name__)

Detection = Optional[tuple]


class CourtZoneFilter:
    """Reject detections outside the court playing surface."""

    def __init__(self):
        self._mask: Optional[np.ndarray] = None
        self._polygon: Optional[np.ndarray] = None

    def calibrate(self, frame: np.ndarray) -> None:
        """Auto-detect court trapezoid from sidelines."""
        h, w = frame.shape[:2]
        poly = self._detect_sidelines(frame)
        if poly is None:
            poly = self._manual_polygon(h, w)
            logger.info("Court zone: using manual polygon")
        else:
            logger.info("Court zone: auto-detected from sidelines")

        self._polygon = poly
        self._mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [poly], 255)
        pct = cv2.countNonZero(self._mask) / (h * w) * 100
        logger.info(f"Court zone: {pct:.1f}% of frame")

    def __call__(self, det: Detection, frame_h: int, frame_w: int) -> Detection:
        if det is None:
            return None
        x, y, conf = det
        if x < 0 or x >= frame_w or y < 0 or y >= frame_h:
            return None
        if self._mask is not None and self._mask[y, x] == 0:
            return None
        return det

    def _detect_sidelines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Find left and right sidelines to build court trapezoid."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        # Restrict to plausible court region
        mask = np.zeros_like(white)
        mask[int(h * 0.15):int(h * 0.75), int(w * 0.05):int(w * 0.95)] = 255
        white = cv2.bitwise_and(white, mask)

        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=15,
        )
        if lines is None or len(lines) < 6:
            return None

        # Find diagonal lines (sidelines have angle 50-85 degrees)
        left_lines = []   # leaning left (/)
        right_lines = []  # leaning right (\)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 == y2:
                continue
            angle = np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1)))
            if not (10 < angle < 50):
                continue
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length < 50:
                continue

            # Ensure y1 < y2 (top to bottom)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            mid_x = (x1 + x2) / 2
            if mid_x < w / 2:
                left_lines.append((x1, y1, x2, y2, length))
            else:
                right_lines.append((x1, y1, x2, y2, length))

        if not left_lines or not right_lines:
            return None

        # Pick longest left and right sideline
        left = max(left_lines, key=lambda l: l[4])
        right = max(right_lines, key=lambda l: l[4])

        # Extrapolate sidelines to find court corners
        # Left sideline goes from top-left to bottom-left
        lx1, ly1, lx2, ly2 = left[:4]
        # Right sideline goes from top-right to bottom-right
        rx1, ry1, rx2, ry2 = right[:4]

        # Find Y range from horizontal lines (baselines)
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5 and abs(x2 - x1) > 80:
                h_lines.append((min(y1, y2), max(y1, y2)))

        if h_lines:
            y_top = min(y for y, _ in h_lines)
            y_bot = max(y for _, y in h_lines)
        else:
            y_top = min(ly1, ry1)
            y_bot = max(ly2, ry2)

        # Ensure y_bot covers at least 68% of frame (near baseline)
        y_bot = max(y_bot, int(h * 0.68))

        # Extrapolate sidelines to y_top and y_bot
        def interp_x(x1, y1, x2, y2, target_y):
            if y2 == y1:
                return (x1 + x2) // 2
            return int(x1 + (x2 - x1) * (target_y - y1) / (y2 - y1))

        margin = COURT_ZONE.get("margin_px", 40)
        m_top = COURT_ZONE.get("margin_top_px", 50)

        tl_x = interp_x(lx1, ly1, lx2, ly2, y_top) - margin
        tr_x = interp_x(rx1, ry1, rx2, ry2, y_top) + margin
        bl_x = interp_x(lx1, ly1, lx2, ly2, y_bot) - margin * 2
        br_x = interp_x(rx1, ry1, rx2, ry2, y_bot) + margin * 2

        poly = np.array([
            [max(0, tl_x), max(0, y_top - m_top)],
            [min(w - 1, tr_x), max(0, y_top - m_top)],
            [min(w - 1, br_x), min(h - 1, y_bot + margin)],
            [max(0, bl_x), min(h - 1, y_bot + margin)],
        ], dtype=np.int32)

        return poly

    def _manual_polygon(self, h: int, w: int) -> np.ndarray:
        """Fallback: configurable manual polygon."""
        cfg = COURT_ZONE["manual_polygon"]
        return np.array([
            [int(w * cfg["top_left"][0]), int(h * cfg["top_left"][1])],
            [int(w * cfg["top_right"][0]), int(h * cfg["top_right"][1])],
            [int(w * cfg["bottom_right"][0]), int(h * cfg["bottom_right"][1])],
            [int(w * cfg["bottom_left"][0]), int(h * cfg["bottom_left"][1])],
        ], dtype=np.int32)
