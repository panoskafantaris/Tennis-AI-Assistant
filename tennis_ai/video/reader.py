"""Video source abstraction — local files and YouTube URLs via yt-dlp."""
import logging
import subprocess
from pathlib import Path
from typing import Generator, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_youtube(url: str) -> str:
    """Use yt-dlp (open-source, Unlicense) to get a direct stream URL."""
    logger.info("Resolving YouTube stream via yt-dlp...")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]/best[ext=mp4]/best",
         "-g", url],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip().splitlines()[0]


class VideoReader:
    """Context manager yielding BGR frames from any video source."""

    def __init__(self, source: Union[str, Path]):
        self.source = str(source)
        self._cap: cv2.VideoCapture = None
        self.fps = 30.0
        self.width = self.height = self.total_frames = 0

    def __enter__(self):
        src = self.source
        if src.startswith(("http://", "https://", "www.")):
            src = _resolve_youtube(src)

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {src}")

        self.fps          = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width        = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Opened: {self.width}x{self.height} @ {self.fps:.1f} fps "
            f"({self.total_frames} frames)"
        )
        return self

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()
