"""Ball tracker — manages detection state and trajectory trail."""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from config.settings import TRACKER


@dataclass
class BallState:
    """Snapshot of ball state at a given frame."""
    detected   : bool  = False
    x          : Optional[int] = None
    y          : Optional[int] = None
    confidence : float = 0.0
    missing    : int   = 0
    trail      : list  = field(default_factory=list)

    @property
    def position(self) -> Optional[Tuple[int, int]]:
        return (self.x, self.y) if self.detected else None


class BallTracker:
    """
    Stateful tracker updated once per frame.
    On detection: append to trail, reset missing counter.
    On miss: increment missing; if > threshold -> clear trail.
    """

    def __init__(self):
        self._trail: Deque[Tuple[int, int]] = deque(
            maxlen=TRACKER["trail_length"],
        )
        self._missing = 0
        self._state = BallState()

    def update(
        self, detection: Optional[Tuple[int, int, float]],
    ) -> BallState:
        """detection: (x, y, confidence) or None."""
        if detection is not None:
            x, y, conf = detection
            if conf >= TRACKER["min_confidence"]:
                self._trail.append((x, y))
                self._missing = 0
                self._state = BallState(
                    detected=True, x=x, y=y,
                    confidence=conf, trail=list(self._trail),
                )
                return self._state

        self._missing += 1
        if self._missing > TRACKER["max_missing_frames"]:
            self._trail.clear()

        self._state = BallState(
            missing=self._missing, trail=list(self._trail),
        )
        return self._state

    def reset(self) -> None:
        self._trail.clear()
        self._missing = 0
        self._state = BallState()
