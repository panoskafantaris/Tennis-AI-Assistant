"""
Background Subtraction detector — finds the ball as a bright foreground
object against a learned static background.

Pipeline:
  1. Build background via median of sampled frames
  2. Per-frame: abs diff against background → threshold → contours
  3. Score candidates by: diff intensity × circularity × brightness
  4. Return best candidate above score threshold

This detector dramatically outperforms HSV-only on broadcast footage
where the ball color shifts due to court surface, compression, and lighting.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base import BaseDetector
from config.settings import BGSUB

logger = logging.getLogger(__name__)


class BackgroundDetector(BaseDetector):
    """Ball detection via background subtraction + candidate scoring."""

    def __init__(self):
        self._bg: Optional[np.ndarray] = None
        self._bg_gray: Optional[np.ndarray] = None
        self._frame_h = 0
        self._frame_w = 0

    @property
    def window_size(self) -> int:
        return 3  # needs 3 frames: bg-sub + frame diff for motion

    def set_background(self, frames: List[np.ndarray]) -> None:
        """Build background from median of sampled frames."""
        if not frames:
            return
        stack = np.stack(
            [f.astype(np.float32) for f in frames], axis=0,
        )
        self._bg = np.median(stack, axis=0).astype(np.uint8)
        self._bg_gray = cv2.cvtColor(self._bg, cv2.COLOR_BGR2GRAY)
        self._frame_h, self._frame_w = self._bg.shape[:2]
        logger.info(f"Background built from {len(frames)} frames")

    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        if len(frames) < self.window_size or self._bg is None:
            return None

        current = frames[-1]
        prev = frames[-2]
        h, w = current.shape[:2]

        # Compute background difference
        bg_diff = cv2.absdiff(current, self._bg)
        bg_gray = cv2.cvtColor(bg_diff, cv2.COLOR_BGR2GRAY)

        # Also compute frame-to-frame diff for motion confirmation
        motion = cv2.absdiff(
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        )

        # Threshold foreground
        _, fg_mask = cv2.threshold(
            bg_gray, BGSUB["diff_thresh"], 255, cv2.THRESH_BINARY,
        )

        # Light cleanup — small kernel to preserve tiny blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Score each candidate
        hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
        candidates = []
        for cnt in contours:
            cand = self._score_candidate(
                cnt, bg_gray, motion, hsv, h, w,
            )
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        # Return highest scoring candidate
        candidates.sort(key=lambda c: -c[2])
        return candidates[0]

    def _score_candidate(
        self,
        cnt,
        bg_diff_gray: np.ndarray,
        motion: np.ndarray,
        hsv: np.ndarray,
        frame_h: int,
        frame_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """Score a contour as a ball candidate. Returns (x, y, score)."""
        area = cv2.contourArea(cnt)
        if not (BGSUB["min_area"] <= area <= BGSUB["max_area"]):
            return None

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Bounds check
        if cx < 2 or cx >= frame_w - 2 or cy < 2 or cy >= frame_h - 2:
            return None

        # Background diff intensity at centroid
        diff_val = float(bg_diff_gray[cy, cx])
        if diff_val < BGSUB["min_diff_score"]:
            return None

        # HSV check — ball is bright, not deeply saturated
        h_val, s_val, v_val = hsv[cy, cx]
        if v_val < BGSUB["min_brightness"]:
            return None
        if s_val > BGSUB["max_saturation"]:
            return None

        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 1 else 0

        # Motion confirmation — strongest discriminator
        # Real ball has motion > 20, static logos have motion ≈ 0
        motion_val = float(motion[cy, cx])
        motion_score = min(motion_val / 50.0, 1.0)

        # Size penalty — ball is 5-80px; players/text > 100px
        large_thresh = BGSUB.get("large_area_thresh", 100)
        if area > large_thresh:
            size_score = 0.3
        elif area < 5:
            size_score = 0.5
        else:
            size_score = 1.0

        # Combined score: motion dominates
        score = (
            (diff_val / 150.0) * 0.20    # diff contribution
            + motion_score * 0.40          # motion is king
            + circ * 0.20                  # roundness
            + size_score * 0.20            # size appropriateness
        )
        return (cx, cy, float(score))
