"""
Video source abstraction.

Supports:
  - Local video files
  - YouTube URLs (via yt-dlp, open-source)

Yields BGR numpy frames one at a time.
The rest of the pipeline never needs to know the source type.
"""
import logging
import subprocess
import cv2
from pathlib import Path
from typing import Generator, Union

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_youtube_url(url: str) -> str:
    """
    Use yt-dlp (open-source, free) to get a direct stream URL.
    yt-dlp: https://github.com/yt-dlp/yt-dlp (Unlicense)
    """
    logger.info("🎥 Resolving YouTube stream via yt-dlp …")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]/best[ext=mp4]/best",
         "-g", url],
        capture_output=True, text=True, check=True,
    )
    stream_url = result.stdout.strip().splitlines()[0]
    logger.info(f"✅ Stream URL resolved.")
    return stream_url


class VideoReader:
    """
    Context manager that wraps a video source.

    Usage:
        with VideoReader("path/to/video.mp4") as vr:
            for frame in vr:
                process(frame)
    """

    def __init__(self, source: Union[str, Path]):
        self.source_input = str(source)
        self._cap: cv2.VideoCapture = None
        self.fps    = 30.0
        self.width  = 0
        self.height = 0
        self.total_frames = 0

    def __enter__(self):
        source = self.source_input
        if source.startswith(("http://", "https://", "www.")):
            source = _resolve_youtube_url(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.fps          = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width        = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"📹 Opened: {self.source_input}\n"
            f"   {self.width}×{self.height} @ {self.fps:.1f} fps  "
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