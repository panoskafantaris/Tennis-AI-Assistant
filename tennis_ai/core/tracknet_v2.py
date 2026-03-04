"""
TrackNet V2 detector — full pipeline from frames to ball coordinates.

Pre-processing  : BGR->RGB, resize, float32 [0,255]
Forward pass    : [B, 256, H, W] logits
Post-processing : argmax -> background subtraction -> blob filter

Source: yastrebksv/TrackNet (MIT License)
"""
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from core.base import BaseDetector
from core.tracknet_v2_model import TrackNetV2Model
from core.weight_adapter import adapt_state_dict
from config.settings import V2, V2_POST, TRACKNET_V2_WEIGHTS, DEVICE, USE_FP16
from utils.device import get_device, to_fp16_if_available

logger = logging.getLogger(__name__)
_GDRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"


def _download_weights(dest: Path) -> None:
    """Auto-download weights via gdown (Google Drive)."""
    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
        )
        import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={_GDRIVE_ID}"
    logger.info(f"Downloading TrackNet V2 weights -> {dest}")
    gdown.download(url, str(dest), quiet=False)
    if not dest.exists():
        raise RuntimeError(f"Download failed. Manual: {url}")


class TrackNetV2Detector(BaseDetector):
    """TrackNet V2 ball detector (3-frame input, 256-class heatmap)."""

    def __init__(self, weights: Path = None):
        self._device = get_device(DEVICE)
        self._H, self._W = V2["input_height"], V2["input_width"]
        self._bg_level: Optional[float] = None

        weights = weights or TRACKNET_V2_WEIGHTS
        self._model = self._load(weights)

    @property
    def window_size(self) -> int:
        return V2["input_frames"]

    def _load(self, path: Path) -> torch.nn.Module:
        model = TrackNetV2Model(input_frames=V2["input_frames"])
        if not path.exists():
            _download_weights(path)
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        adapted = adapt_state_dict(model, raw)
        model.load_state_dict(adapted, strict=False)
        model = to_fp16_if_available(model.to(self._device), USE_FP16)
        model.eval()
        logger.info(f"TrackNet V2 loaded from {path}")
        return model

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Stack 3 BGR frames -> [1, 9, H, W] float32."""
        channels = []
        for bgr in frames:
            resized = cv2.resize(bgr, (self._W, self._H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)
            channels.append(t)
        return torch.cat(channels, dim=0).unsqueeze(0).to(self._device)

    def _postprocess(
        self, heatmap: np.ndarray, orig_h: int, orig_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """Background-subtracted blob detection."""
        hist, edges = np.histogram(heatmap.ravel(), bins=256, range=(0, 256))
        bg = float(edges[hist.argmax()])
        self._bg_level = bg if self._bg_level is None else (
            0.9 * self._bg_level + 0.1 * bg
        )
        residual = np.clip(heatmap - self._bg_level, 0, None)
        r_max = residual.max()
        if r_max < 5.0:
            return None

        norm = (residual / r_max * 255).astype(np.uint8)
        thresh = int(V2_POST["heatmap_thresh"] * 255)
        _, binary = cv2.threshold(norm, thresh, 255, cv2.THRESH_BINARY)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        best_peak, best_cx, best_cy = -1.0, None, None

        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (V2_POST["min_blob_area"] <= area <= V2_POST["max_blob_area"]):
                continue
            peak = float(residual[labels == i].max())
            if peak > best_peak:
                best_peak = peak
                best_cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
                best_cy = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)

        if best_cx is None:
            return None

        x = int(best_cx * orig_w / self._W)
        y = int(best_cy * orig_h / self._H)
        return x, y, float(min(best_peak / r_max, 1.0))

    @torch.no_grad()
    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        if len(frames) < self.window_size:
            return None
        orig_h, orig_w = frames[-1].shape[:2]
        tensor = self._preprocess(frames)
        logits = self._model(tensor)[0]
        heatmap = logits.argmax(dim=0).cpu().numpy().astype(np.float32)
        return self._postprocess(heatmap, orig_h, orig_w)
