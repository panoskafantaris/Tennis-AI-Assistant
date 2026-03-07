"""
Stationarity filter — rejects detections that stay in the same position.

A tennis ball is always moving during play. If a detection sits at the
same coordinates for several frames, it's a static false positive
(logo dot, camera artifact, scoreboard element).

Also rejects candidates that appear at known static FP locations
accumulated during the run.
"""
import logging
from collections import deque
from typing import Deque, Optional, Set, Tuple

from config.settings import STATIONARITY

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]


class StationarityFilter:
    """Reject detections that don't move between frames."""

    def __init__(self):
        self._history: Deque[Optional[Tuple[int, int]]] = deque(
            maxlen=STATIONARITY["window"],
        )
        self._blacklist: Set[Tuple[int, int]] = set()
        self._static_count = 0

    def __call__(self, det: Detection) -> Detection:
        if det is None:
            self._history.append(None)
            self._static_count = 0
            return None

        x, y, conf = det

        # Check against blacklisted static positions
        for bx, by in self._blacklist:
            if abs(x - bx) < STATIONARITY["radius"] and \
               abs(y - by) < STATIONARITY["radius"]:
                return None

        # Check if position is static (same as recent history)
        is_static = self._check_static(x, y)
        self._history.append((x, y))

        if is_static:
            self._static_count += 1
            if self._static_count >= STATIONARITY["max_static_frames"]:
                # Blacklist this position permanently
                self._blacklist.add(
                    (x // STATIONARITY["radius"] * STATIONARITY["radius"],
                     y // STATIONARITY["radius"] * STATIONARITY["radius"]),
                )
                logger.info(
                    f"Blacklisted static FP at ({x},{y})"
                )
                self._static_count = 0
                return None
        else:
            self._static_count = 0

        return det

    def _check_static(self, x: int, y: int) -> bool:
        """Check if position is within radius of ALL recent positions."""
        if len(self._history) < STATIONARITY["min_history"]:
            return False

        radius = STATIONARITY["radius"]
        for pos in self._history:
            if pos is None:
                return False  # gap means it's not stationary
            px, py = pos
            if abs(x - px) > radius or abs(y - py) > radius:
                return False
        return True

    def reset(self) -> None:
        self._history.clear()
        self._static_count = 0
        # Keep blacklist — static objects don't move between resets
