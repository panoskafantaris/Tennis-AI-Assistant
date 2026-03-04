"""
TrackNetV3 inference engine.

Key difference from V2: takes 8 consecutive frames + 1 background.
  Input:  [B, 27, H, W] float32 [0,1] — 8 RGB frames + 1 bg = 27ch
  Output: [B, 8, H, W] — per-frame heatmaps (one per input frame)

Post-processing: threshold → contour → filter by compactness.
Real ball = tiny sharp spike (5-25 px area, peak > 0.5).
Court marks / shadows = diffuse blobs (50+ px, lower peak density).
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List

from utils.device import get_device
from config.settings import DEVICE

_H = 288
_W = 512

# ── Post-processing thresholds ─────────────────────────────────────────────
_PEAK_THRESH     = 0.45   # minimum raw heatmap peak to consider
_BINARY_THRESH   = 128    # threshold for binary mask (0-255 scale)
_MIN_AREA        = 1      # minimum blob area in model pixels
_MAX_AREA        = 40     # maximum blob area (ball is ~5-25px at 288×512)
_MIN_SHARPNESS   = 0.015  # peak / area — higher = more compact


class TrackNetV3Inference:
    """
    Call `predict(frames)` with a list of 8 BGR frames.
    Returns (x, y, confidence) or None.
    """

    def __init__(self, model: torch.nn.Module, param_dict: dict):
        self.model = model
        self.device = get_device(DEVICE)
        self.seq_len = param_dict.get("seq_len", 8)
        self.bg_mode = param_dict.get("bg_mode", "concat")
        self.H = _H
        self.W = _W
        self._bg_tensor: Optional[torch.Tensor] = None

    def set_background(self, frames: List[np.ndarray]) -> None:
        """Estimate background via median of sampled frames."""
        resized = [
            cv2.resize(f, (self.W, self.H)).astype(np.float32)
            for f in frames
        ]
        bg = np.median(np.stack(resized), axis=0).astype(np.uint8)
        self._bg_tensor = self._to_tensor(bg)

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → [3, H, W] float32 [0, 1]."""
        r = cv2.resize(bgr, (self.W, self.H))
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(
            rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Build [1, 27, H, W] tensor: 8 frames + 1 background."""
        tensors = [self._to_tensor(f) for f in frames]
        if self._bg_tensor is None:
            self._bg_tensor = self._to_tensor(frames[0])
        tensors.append(self._bg_tensor)
        return torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)

    def _postprocess(
        self, heatmap: np.ndarray, orig_h: int, orig_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Single-frame heatmap [H, W] float → (x, y, confidence).

        Strategy: real ball produces a tiny, sharp spike (5-25 px).
        Court marks and shadows produce diffuse blobs (50+ px).
        We filter by: peak value, blob area, and sharpness (peak/area).
        """
        peak_val = float(heatmap.max())
        if peak_val < _PEAK_THRESH:
            return None

        # Normalise to 0-255
        norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        _, binary = cv2.threshold(norm, _BINARY_THRESH, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Score each candidate by compactness (sharpness = peak / area)
        best_score = -1.0
        best_result = None

        for c in contours:
            area = cv2.contourArea(c)
            if not (_MIN_AREA <= area <= _MAX_AREA):
                continue

            # Peak value within this blob
            mask = np.zeros_like(norm)
            cv2.drawContours(mask, [c], -1, 255, -1)
            blob_peak = float(heatmap[mask > 0].max())

            if blob_peak < _PEAK_THRESH:
                continue

            # Sharpness: compact bright spike scores higher
            sharpness = blob_peak / max(area, 1)
            if sharpness < _MIN_SHARPNESS:
                continue

            # Combined score: peak weighted by compactness
            score = blob_peak * (1.0 + sharpness)

            if score > best_score:
                best_score = score
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x = int(cx * orig_w / self.W)
                y = int(cy * orig_h / self.H)
                best_result = (x, y, float(blob_peak))

        return best_result

    @torch.no_grad()
    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        """
        frames: list of 8 BGR numpy arrays (oldest → newest).
        Returns (x, y, confidence) for the LAST frame, or None.
        """
        if len(frames) < self.seq_len:
            return None

        orig_h, orig_w = frames[-1].shape[:2]
        inp = self._preprocess(frames[-self.seq_len:])

        out = self.model(inp)  # [B, seq_len, H, W]

        # Take prediction for the last frame
        if out.dim() == 4:
            heatmap = out[0, -1].cpu().numpy()
        elif out.dim() == 3:
            heatmap = out[0].cpu().numpy()
        else:
            heatmap = out[0, -1].cpu().numpy()

        return self._postprocess(heatmap, orig_h, orig_w)