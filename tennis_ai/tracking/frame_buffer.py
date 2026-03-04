"""
Frame buffer — maintains a rolling window of N frames for TrackNet.

TrackNet requires exactly 3 consecutive frames as input.
This buffer fills up gradually and is always ready to provide the latest window.
"""
from collections import deque
from typing import List, Optional

import numpy as np

from config.settings import TRACKNET


class FrameBuffer:
    """
    Fixed-size FIFO buffer.
    Call `push(frame)` each frame; call `get_window()` to get the 3-frame list.
    """

    def __init__(self, size: int = None):
        self._size   = size or TRACKNET["input_frames"]
        self._buffer : deque = deque(maxlen=self._size)

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def ready(self) -> bool:
        """True once the buffer has collected enough frames."""
        return len(self._buffer) == self._size

    def get_window(self) -> Optional[List[np.ndarray]]:
        """Return list of frames (oldest → newest), or None if not ready."""
        if not self.ready():
            return None
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()