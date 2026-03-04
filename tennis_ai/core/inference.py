"""
TrackNet inference engine.

Pre/post-processing + forward pass.
Post-processing uses a margin score (non-bg logit minus bg logit) and
blob-size filtering to isolate the small, compact tennis ball peak.
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple

from config.settings import TRACKNET, DEVICE
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
        self.thresh = TRACKNET["heatmap_thresh"]

    def _preprocess(self, frames: list) -> torch.Tensor:
        """Stack 3 BGR frames → normalised 9-channel tensor [1, 9, H, W]."""
        channels = []
        for bgr in frames:
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (self.W, self.H)),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32) / 255.0
            channels.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.cat(channels, dim=0).unsqueeze(0).to(self.device)

    def _find_ball_peak(
        self,
        score_map: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find the most ball-like peak in score_map.

        Strategy:
          1. Threshold at top-N% of values to get candidate regions.
          2. Find connected components and filter by area (ball is small).
          3. Pick the component whose peak value is highest.

        This avoids large blobs (crowd, players) dominating over the tiny ball.
        """
        # Normalise to 0-255 uint8 for OpenCV
        s_min, s_max = score_map.min(), score_map.max()
        if s_max - s_min < 1e-6:
            return None

        norm = ((score_map - s_min) / (s_max - s_min) * 255).astype(np.uint8)

        # Threshold: keep top 5% of pixels as candidates
        thresh_val = int(255 * (1 - self.thresh))
        _, binary  = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            return None

        # Image area — ball should occupy 0.01% to 2% of frame
        total_px   = self.H * self.W
        min_area   = max(1,  int(total_px * 0.0001))
        max_area   = int(total_px * 0.02)

        best_score = -1.0
        best_cx, best_cy = None, None

        for i in range(1, num_labels):          # skip background label 0
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue

            # Score = peak value of the score_map within this component
            mask       = (labels == i)
            peak_score = float(score_map[mask].max())
            if peak_score > best_score:
                best_score = peak_score
                cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  / 2)
                cy = int(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] / 2)
                best_cx, best_cy = cx, cy

        if best_cx is None:
            return None

        confidence = (best_score - s_min) / (s_max - s_min)
        x = int(best_cx * orig_w / self.W)
        y = int(best_cy * orig_h / self.H)
        return x, y, float(confidence)

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
        tensor         = self._preprocess(frames)

        out = self.model(tensor)[0].float()   # [256, H, W]

        # Margin score: best non-bg logit minus background logit.
        # Where ball is present the non-bg class should pull ahead of bg.
        # Even if bg wins globally, the ball pixel has the smallest gap.
        non_bg_max = out[1:].max(dim=0).values   # [H, W]
        bg_logit   = out[0]                       # [H, W]
        score_map  = (non_bg_max - bg_logit).cpu().numpy()   # [H, W]

        return self._find_ball_peak(score_map, orig_h, orig_w)