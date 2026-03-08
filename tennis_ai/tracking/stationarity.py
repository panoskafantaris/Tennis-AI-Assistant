"""
Stationarity filter — rejects detections that stay in the same position.

A tennis ball is always moving during play. If a detection sits at the
same coordinates for several frames, it's a static false positive
(logo dot, camera artifact, scoreboard element).

Improvements:
  - Blacklist entries have a TTL (expire after N frames)
  - Full reset method for scene cuts
  - Less aggressive blacklisting thresholds
"""
import logging
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

from config.settings import STATIONARITY

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]


class StationarityFilter:
    """Reject detections that don't move between frames."""

    def __init__(self):
        self._history: Deque[Optional[Tuple[int, int]]] = deque(
            maxlen=STATIONARITY["window"],
        )
        # Blacklist with TTL: grid_pos -> frames_remaining
        self._blacklist: Dict[Tuple[int, int], int] = {}
        self._static_count = 0
        self._frame_count = 0
        self._ttl = STATIONARITY.get("blacklist_ttl", 300)

    def __call__(self, det: Detection) -> Detection:
        self._frame_count += 1

        # Decay blacklist TTL every frame
        if self._frame_count % 10 == 0:
            self._decay_blacklist()

        if det is None:
            self._history.append(None)
            self._static_count = 0
            return None

        x, y, conf = det
        radius = STATIONARITY["radius"]

        # Check against blacklisted static positions
        grid_key = (x // radius * radius, y // radius * radius)
        for bkey, ttl in self._blacklist.items():
            if ttl <= 0:
                continue
            bx, by = bkey
            if abs(x - bx) < radius and abs(y - by) < radius:
                return None

        # Check if position is static
        is_static = self._check_static(x, y)
        self._history.append((x, y))

        if is_static:
            self._static_count += 1
            if self._static_count >= STATIONARITY["max_static_frames"]:
                self._blacklist[grid_key] = self._ttl
                logger.info(f"Blacklisted static FP at ({x},{y}) TTL={self._ttl}")
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
                return False
            px, py = pos
            if abs(x - px) > radius or abs(y - py) > radius:
                return False
        return True

    def _decay_blacklist(self) -> None:
        """Reduce TTL of all blacklist entries, remove expired."""
        expired = [k for k, v in self._blacklist.items() if v <= 0]
        for k in expired:
            del self._blacklist[k]
        for k in self._blacklist:
            self._blacklist[k] -= 10

    def reset(self) -> None:
        """Full reset — call on scene cuts."""
        self._history.clear()
        self._static_count = 0

    def reset_full(self) -> None:
        """Reset everything including blacklist — call on scene cuts."""
        self._history.clear()
        self._static_count = 0
        self._blacklist.clear()
        logger.info("Stationarity filter fully reset (blacklist cleared)")