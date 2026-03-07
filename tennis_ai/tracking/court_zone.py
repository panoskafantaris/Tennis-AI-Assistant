"""
Court zone filter — restricts detections to the playing area.

Two modes:
  1. Auto-detect: find white court lines → compute bounding polygon
  2. Manual fallback: use configured trapezoid from settings

The playing area is a perspective-warped trapezoid in broadcast view:
wider at the bottom (near camera), narrower at the top (far baseline).
A margin is added above for serves/lobs that go high.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import COURT_ZONE

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]


class CourtZoneFilter:
    """Reject detections outside the court playing area polygon."""

    def __init__(self):
        self._polygon: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._frame_h = 0
        self._frame_w = 0

    def calibrate(self, frame: np.ndarray) -> None:
        """
        Auto-detect court boundaries from a frame.
        Falls back to manual config if detection fails.
        """
        h, w = frame.shape[:2]
        self._frame_h, self._frame_w = h, w

        poly = self._detect_court_polygon(frame)
        if poly is not None:
            self._polygon = poly
            logger.info("Court zone: auto-detected from court lines")
        else:
            self._polygon = self._manual_polygon(h, w)
            logger.info("Court zone: using configured polygon")

        # Pre-compute binary mask for fast lookup
        self._mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [self._polygon], 255)

        area_pct = cv2.countNonZero(self._mask) / (h * w) * 100
        logger.info(f"Court zone covers {area_pct:.1f}% of frame")

    def __call__(
        self, det: Detection, frame_h: int, frame_w: int,
    ) -> Detection:
        if det is None:
            return None
        x, y, conf = det

        # Bounds check
        if x < 0 or x >= frame_w or y < 0 or y >= frame_h:
            return None

        # If mask exists, use fast pixel lookup
        if self._mask is not None:
            if self._mask[y, x] > 0:
                return det
            return None

        # Fallback: point-in-polygon test
        if self._polygon is not None:
            if cv2.pointPolygonTest(
                self._polygon, (float(x), float(y)), False,
            ) >= 0:
                return det
            return None

        return det

    def _detect_court_polygon(
        self, frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Auto-detect court from white lines in the frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find bright white pixels (court lines)
        _, white = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)

        # Focus on center court region — exclude edges/banners
        mask = np.zeros_like(white)
        mask[int(h * 0.15):int(h * 0.75), int(w * 0.05):int(w * 0.95)] = 255
        white = cv2.bitwise_and(white, mask)

        # Detect line segments
        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 80,
            minLineLength=80, maxLineGap=20,
        )
        if lines is None or len(lines) < 4:
            return None

        # Collect all line endpoints
        pts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pts.extend([(x1, y1), (x2, y2)])

        pts = np.array(pts)
        y_min = pts[:, 1].min()
        y_max = pts[:, 1].max()
        x_min = pts[:, 0].min()
        x_max = pts[:, 0].max()

        # Build trapezoid with margin for ball trajectory
        margin_top = int(h * COURT_ZONE["margin_top"])
        margin_side = int(w * COURT_ZONE["margin_side"])

        polygon = np.array([
            [x_min - margin_side, y_min - margin_top],
            [x_max + margin_side, y_min - margin_top],
            [x_max + margin_side * 2, y_max + int(h * 0.02)],
            [x_min - margin_side * 2, y_max + int(h * 0.02)],
        ], dtype=np.int32)

        # Clamp to frame bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
        return polygon

    def _manual_polygon(
        self, h: int, w: int,
    ) -> np.ndarray:
        """Fallback polygon from config percentages."""
        cfg = COURT_ZONE["manual_polygon"]
        return np.array([
            [int(w * cfg["top_left"][0]),     int(h * cfg["top_left"][1])],
            [int(w * cfg["top_right"][0]),    int(h * cfg["top_right"][1])],
            [int(w * cfg["bottom_right"][0]), int(h * cfg["bottom_right"][1])],
            [int(w * cfg["bottom_left"][0]),  int(h * cfg["bottom_left"][1])],
        ], dtype=np.int32)
