"""Abstract detector interface — all backends implement this."""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseDetector(ABC):
    """
    Uniform interface for ball detection.
    Every detector takes a list of BGR frames and returns
    (x, y, confidence) in original frame coordinates, or None.
    """

    @property
    @abstractmethod
    def window_size(self) -> int:
        """Number of consecutive frames required for prediction."""

    @abstractmethod
    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        """
        frames: list of BGR numpy arrays (oldest -> newest).
        Returns (x, y, confidence) or None.
        """
