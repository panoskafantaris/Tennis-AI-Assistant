"""Rolling frame buffer — feeds N-frame windows to detectors."""
from collections import deque
from typing import List, Optional

import numpy as np


class FrameBuffer:
    """
    Fixed-size FIFO buffer.
    push(frame) each frame; get_window() returns the N-frame list.
    """

    def __init__(self, size: int = 3):
        self._size = size
        self._buffer: deque = deque(maxlen=size)

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def ready(self) -> bool:
        return len(self._buffer) == self._size

    def get_window(self) -> Optional[List[np.ndarray]]:
        if not self.ready():
            return None
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
