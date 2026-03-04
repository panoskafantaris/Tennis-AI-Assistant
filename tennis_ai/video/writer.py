"""Video writer — saves annotated frames to MP4."""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """Context manager wrapping cv2.VideoWriter."""

    def __init__(self, path: Path, fps: float, size: tuple):
        self.path = Path(path)
        self.fps  = fps
        self.size = size  # (width, height)
        self._writer: cv2.VideoWriter = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, self.fps, self.size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {self.path}")
        logger.info(f"Writing output -> {self.path}")
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer:
            h, w = frame.shape[:2]
            if (w, h) != self.size:
                frame = cv2.resize(frame, self.size)
            self._writer.write(frame)

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
            logger.info(f"Video saved: {self.path}")
