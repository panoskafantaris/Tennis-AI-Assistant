"""
Detection filters — reject false positives before they reach the tracker.

ROIFilter      : reject detections in scoreboard/banner zones.
VelocityFilter : reject physically impossible jumps between frames.
"""
from collections import deque
from typing import Deque, Optional, Tuple

from config.settings import ROI, VELOCITY

Detection = Optional[Tuple[int, int, float]]


class ROIFilter:
    """Reject detections outside the playable court region."""

    def __init__(self):
        self._top    = ROI["top"]
        self._bottom = ROI["bottom"]
        self._left   = ROI["left"]
        self._right  = ROI["right"]

    def __call__(
        self, det: Detection, frame_h: int, frame_w: int,
    ) -> Detection:
        if det is None:
            return None
        x, y, conf = det
        y_min = int(frame_h * self._top)
        y_max = int(frame_h * (1.0 - self._bottom))
        x_min = int(frame_w * self._left)
        x_max = int(frame_w * (1.0 - self._right))
        if y_min <= y <= y_max and x_min <= x <= x_max:
            return det
        return None


class VelocityFilter:
    """
    Reject detections that teleport too far from predicted position.

    A serve tops ~260 km/h => ~192 px/frame at 30fps on 1920px.
    We use a generous max_px_per_frame for safety margin.
    """

    def __init__(self):
        self._max_px   = VELOCITY["max_px_per_frame"]
        self._min_det  = VELOCITY["min_detections"]
        self._history: Deque[Tuple[int, int]] = deque(
            maxlen=VELOCITY["history_len"],
        )
        self._gap = 0

    def __call__(self, det: Detection) -> Detection:
        if det is None:
            self._gap += 1
            if self._gap > 15:
                self._history.clear()
            return None

        x, y, conf = det

        if len(self._history) < self._min_det:
            self._history.append((x, y))
            self._gap = 0
            return det

        # Linear extrapolation from last two points
        px, py = self._history[-1]
        if len(self._history) >= 2:
            ppx, ppy = self._history[-2]
            pred_x = px + (px - ppx)
            pred_y = py + (py - ppy)
        else:
            pred_x, pred_y = px, py

        dist = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5
        max_dist = self._max_px * max(1, self._gap + 1)

        if dist <= max_dist:
            self._history.append((x, y))
            self._gap = 0
            return det

        self._gap += 1
        return None

    def reset(self) -> None:
        self._history.clear()
        self._gap = 0
