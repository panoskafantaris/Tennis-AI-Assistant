"""
Video writer — saves annotated frames to an MP4 file.
Optional: only write if an output path is provided.
"""
import cv2
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter.

    Usage:
        with VideoWriter("output/result.mp4", fps=30, size=(1280, 720)) as vw:
            for frame in frames:
                vw.write(frame)
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple,   # (width, height)
    ):
        self.path  = Path(path)
        self.fps   = fps
        self.size  = size
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, self.fps, self.size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter at {self.path}")
        logger.info(f"💾 Writing output → {self.path}")
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer:
            # Resize frame to match writer size if needed
            h, w = frame.shape[:2]
            if (w, h) != self.size:
                frame = cv2.resize(frame, self.size)
            self._writer.write(frame)

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
            logger.info(f"✅ Video saved: {self.path}")