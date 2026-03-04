"""
Velocity-based temporal consistency filter.

Rejects detections that teleport too far from the previous
known position in a single frame. A tennis ball can move fast
but not instantaneously across the court.

Also provides: when multiple candidates exist, prefer the one
closest to the predicted position (linear extrapolation).
"""
from typing import Optional, Tuple, Deque
from collections import deque

from config.settings import VELOCITY_FILTER


class VelocityFilter:
    """
    Tracks ball velocity and rejects physically impossible jumps.

    A serve tops out ~260 km/h ≈ 72 m/s. At 30fps on a 1920px wide
    frame (~24m court), max movement per frame ≈ 72/30 * (1920/24)
    ≈ 192 px/frame. We use a generous multiplier for safety.
    """

    def __init__(self):
        self._max_px_per_frame = VELOCITY_FILTER["max_px_per_frame"]
        self._min_detections   = VELOCITY_FILTER["min_detections_to_activate"]
        self._history: Deque[Tuple[int, int]] = deque(
            maxlen=VELOCITY_FILTER["history_len"]
        )
        self._frames_since_detection = 0

    def filter(
        self,
        detection: Optional[Tuple[int, int, float]],
    ) -> Optional[Tuple[int, int, float]]:
        """
        Accept or reject a detection based on distance from
        predicted position. Returns detection or None.
        """
        if detection is None:
            self._frames_since_detection += 1
            # Reset after long gap — ball may reappear anywhere
            if self._frames_since_detection > 15:
                self._history.clear()
            return None

        x, y, conf = detection

        # Not enough history to filter — accept and record
        if len(self._history) < self._min_detections:
            self._history.append((x, y))
            self._frames_since_detection = 0
            return detection

        # Predict next position via linear extrapolation
        prev_x, prev_y = self._history[-1]
        if len(self._history) >= 2:
            pp_x, pp_y = self._history[-2]
            vx = prev_x - pp_x
            vy = prev_y - pp_y
            pred_x = prev_x + vx
            pred_y = prev_y + vy
        else:
            pred_x, pred_y = prev_x, prev_y

        # Distance from prediction
        dist = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5

        # Scale max distance by frames since last detection
        # (ball moves further if we missed frames)
        gap = max(1, self._frames_since_detection + 1)
        max_dist = self._max_px_per_frame * gap

        if dist <= max_dist:
            self._history.append((x, y))
            self._frames_since_detection = 0
            return detection

        # Reject — too far from expected position
        self._frames_since_detection += 1
        return None

    def reset(self) -> None:
        self._history.clear()
        self._frames_since_detection = 0