"""
TrackNet inference engine.

Pre-processing  : BGR->RGB, resize, float32 [0,255]
Forward pass    : model outputs [B, 256, H, W] logits
Post-processing : argmax -> background subtraction -> blob filter

The model predicts class ~140 for background and peaks near 233
for ball pixels. Subtracting the background mode isolates the
ball signal cleanly.
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple

from config.settings import TRACKNET, POSTPROCESS, DEVICE
from utils.device import get_device


class TrackNetInference:
    """
    Call `predict(frames)` with a list of 3 BGR frames.
    Returns (x, y, confidence) or None.
    """

    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.device = get_device(DEVICE)
        self.H      = TRACKNET["input_height"]
        self.W      = TRACKNET["input_width"]
        self.color  = TRACKNET["color_mode"]
        self._bg_level = None  # estimated once, reused

    # ── Preprocessing ─────────────────────────────────────────────

    def _preprocess(self, frames: list) -> torch.Tensor:
        """Stack 3 BGR frames -> [1, 9, H, W] float32 in [0,255]."""
        channels = []
        for bgr in frames:
            resized = cv2.resize(bgr, (self.W, self.H))
            if self.color == "rgb":
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                img = resized
            t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            channels.append(t)
        return torch.cat(channels, dim=0).unsqueeze(0).to(self.device)

    # ── Post-processing ───────────────────────────────────────────

    def _find_ball(
        self,
        heatmap: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Background-subtracted blob detection.

        1. Compute background mode (most common argmax value)
        2. Subtract it -> residual (ball peaks above zero)
        3. Threshold at fraction of max residual
        4. Find small blobs -> pick highest peak
        """
        # Estimate background level from histogram mode
        hist, edges = np.histogram(heatmap.ravel(), bins=256, range=(0, 256))
        bg_level = float(edges[hist.argmax()])

        # Update running estimate (smooth across frames)
        if self._bg_level is None:
            self._bg_level = bg_level
        else:
            self._bg_level = 0.9 * self._bg_level + 0.1 * bg_level

        # Residual heatmap
        residual = np.clip(heatmap - self._bg_level, 0, None)
        r_max = residual.max()

        if r_max < 5.0:
            return None  # no ball signal

        # Threshold at configured fraction of max
        thresh_frac = POSTPROCESS["heatmap_thresh"]
        norm_res = (residual / r_max * 255).astype(np.uint8)
        thresh_val = int(thresh_frac * 255)
        _, binary = cv2.threshold(
            norm_res, thresh_val, 255, cv2.THRESH_BINARY
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return None

        min_area = POSTPROCESS["min_blob_area"]
        max_area = POSTPROCESS["max_blob_area"]

        best_peak = -1.0
        best_cx, best_cy = None, None

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            mask = (labels == i)
            peak = float(residual[mask].max())
            if peak > best_peak:
                best_peak = peak
                cx = int(
                    stats[i, cv2.CC_STAT_LEFT]
                    + stats[i, cv2.CC_STAT_WIDTH] / 2
                )
                cy = int(
                    stats[i, cv2.CC_STAT_TOP]
                    + stats[i, cv2.CC_STAT_HEIGHT] / 2
                )
                best_cx, best_cy = cx, cy

        if best_cx is None:
            return None

        confidence = min(best_peak / r_max, 1.0)
        x = int(best_cx * orig_w / self.W)
        y = int(best_cy * orig_h / self.H)
        return x, y, float(confidence)

    # ── Main predict ──────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        frames: list,
    ) -> Optional[Tuple[int, int, float]]:
        """
        frames : list of 3 BGR numpy arrays (most recent last)
        returns: (x, y, confidence) in original frame coords, or None
        """
        if len(frames) < TRACKNET["input_frames"]:
            return None

        orig_h, orig_w = frames[-1].shape[:2]
        tensor = self._preprocess(frames)
        logits = self.model(tensor)[0]  # [256, H, W]

        heatmap = logits.argmax(dim=0).cpu().numpy().astype(np.float32)
        return self._find_ball(heatmap, orig_h, orig_w)