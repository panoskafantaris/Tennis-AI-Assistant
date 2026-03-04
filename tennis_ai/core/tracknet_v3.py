"""
TrackNet V3 detector — 8-frame input with background concatenation.

Input  : [B, 27, H, W] — 8 RGB frames + 1 background = 27 channels
Output : [B, 8, H, W]  — per-frame heatmaps

Source: qaz812345/TrackNetV3 (MIT License)
Setup: python scripts/setup_v3.py
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from core.base import BaseDetector
from config.settings import V3, V3_POST, TRACKNET_V3_DIR, TRACKNET_V3_REPO, DEVICE
from utils.device import get_device

logger = logging.getLogger(__name__)


class TrackNetV3Detector(BaseDetector):
    """TrackNet V3 ball detector (8-frame + background input)."""

    def __init__(self, weights: Path = None):
        self._device = get_device(DEVICE)
        self._H, self._W = V3["input_height"], V3["input_width"]
        self._bg_tensor: Optional[torch.Tensor] = None

        self._model, self._params = self._load(weights)

    @property
    def window_size(self) -> int:
        return self._params.get("seq_len", V3["input_frames"])

    def _load(self, weights: Path = None) -> tuple:
        """Import from cloned repo and load checkpoint."""
        repo = str(TRACKNET_V3_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        if not TRACKNET_V3_REPO.exists():
            raise FileNotFoundError(
                f"TrackNetV3 repo not found at {TRACKNET_V3_REPO}\n"
                "Run: python scripts/setup_v3.py"
            )
        from model import TrackNet  # from cloned repo

        path = weights or (TRACKNET_V3_DIR / "TrackNet_best.pt")
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}\nRun: python scripts/setup_v3.py"
            )

        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        state = ckpt["model"]
        params = ckpt.get("param_dict", {})

        first_key = next(k for k in state if "weight" in k)
        last_key = [k for k in state if "weight" in k][-1]
        in_dim = state[first_key].shape[1]
        out_dim = state[last_key].shape[0]

        model = TrackNet(in_dim=in_dim, out_dim=out_dim)
        model.load_state_dict(state, strict=True)
        model = model.to(self._device).eval()
        logger.info(f"TrackNet V3 loaded (seq_len={params.get('seq_len', 8)})")
        return model, params

    def set_background(self, frames: List[np.ndarray]) -> None:
        """Estimate background via median of sampled frames."""
        resized = [
            cv2.resize(f, (self._W, self._H)).astype(np.float32)
            for f in frames
        ]
        bg = np.median(np.stack(resized), axis=0).astype(np.uint8)
        self._bg_tensor = self._to_tensor(bg)

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        r = cv2.resize(bgr, (self._W, self._H))
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(
            rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        tensors = [self._to_tensor(f) for f in frames]
        if self._bg_tensor is None:
            self._bg_tensor = self._to_tensor(frames[0])
        tensors.append(self._bg_tensor)
        return torch.cat(tensors, dim=0).unsqueeze(0).to(self._device)

    def _postprocess(
        self, heatmap: np.ndarray, orig_h: int, orig_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """Compact blob detection — real ball = tiny sharp spike."""
        if float(heatmap.max()) < V3_POST["peak_thresh"]:
            return None

        norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        _, binary = cv2.threshold(
            norm, V3_POST["binary_thresh"], 255, cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        best_score, best_result = -1.0, None
        for c in contours:
            area = cv2.contourArea(c)
            if not (V3_POST["min_area"] <= area <= V3_POST["max_area"]):
                continue
            mask = np.zeros_like(norm)
            cv2.drawContours(mask, [c], -1, 255, -1)
            peak = float(heatmap[mask > 0].max())
            if peak < V3_POST["peak_thresh"]:
                continue
            sharpness = peak / max(area, 1)
            if sharpness < V3_POST["min_sharpness"]:
                continue
            score = peak * (1.0 + sharpness)
            if score > best_score:
                best_score = score
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_result = (
                    int(cx * orig_w / self._W),
                    int(cy * orig_h / self._H),
                    float(peak),
                )
        return best_result

    @torch.no_grad()
    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        if len(frames) < self.window_size:
            return None
        orig_h, orig_w = frames[-1].shape[:2]
        inp = self._preprocess(frames[-self.window_size:])
        out = self._model(inp)
        heatmap = out[0, -1].cpu().numpy() if out.dim() == 4 else out[0].cpu().numpy()
        return self._postprocess(heatmap, orig_h, orig_w)
