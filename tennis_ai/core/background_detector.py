"""
Background Subtraction detector — finds the ball as a SMALL moving
foreground object via TopHat + frame motion.

Key insight: the ball is 3-15px diameter. Morphological TopHat removes
objects larger than the kernel (players, ball baskets, scoreboard).
Frame-to-frame motion rejects anything static.

Pipeline:
  1. Background diff → TopHat (removes large objects)
  2. Frame motion mask (removes static objects)
  3. Combine: TopHat ∩ motion ∩ court zone
  4. Contour analysis: area, circularity, brightness, motion scoring
  5. Return top-K candidates sorted by score
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base import BaseDetector
from config.settings import BGSUB

logger = logging.getLogger(__name__)

Candidate = Tuple[int, int, float]  # (x, y, score)


class BackgroundDetector(BaseDetector):
    """Ball detection via TopHat + motion filtering."""

    def __init__(self):
        self._bg: Optional[np.ndarray] = None
        self._tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (21, 21),
        )

    @property
    def window_size(self) -> int:
        return 3

    def set_background(self, frames: List[np.ndarray]) -> None:
        if not frames:
            return
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        self._bg = np.median(stack, axis=0).astype(np.uint8)
        logger.info(f"Background built from {len(frames)} frames")

    def predict(
        self, frames: List[np.ndarray],
    ) -> Optional[Candidate]:
        """Return best candidate, or None."""
        candidates = self.predict_topk(frames, k=1)
        return candidates[0] if candidates else None

    def predict_topk(
        self, frames: List[np.ndarray], k: int = 5,
    ) -> List[Candidate]:
        """Return up to k candidates sorted by score (best first)."""
        if len(frames) < self.window_size or self._bg is None:
            return []

        current = frames[-1]
        prev = frames[-2]
        h, w = current.shape[:2]

        # 1. Background diff → TopHat (kills large objects)
        bg_diff = cv2.cvtColor(
            cv2.absdiff(current, self._bg), cv2.COLOR_BGR2GRAY,
        )
        tophat = cv2.morphologyEx(bg_diff, cv2.MORPH_TOPHAT, self._tophat_kernel)

        # 2. Frame-to-frame motion (kills static objects)
        motion = cv2.absdiff(
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        )

        # 3. Threshold and combine
        _, th_bin = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
        _, mo_bin = cv2.threshold(motion, 6, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(th_bin, mo_bin)

        # Light cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # 4. Score candidates
        hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
        scored: List[Candidate] = []

        for cnt in contours:
            cand = self._score_candidate(
                cnt, tophat, motion, hsv, h, w,
            )
            if cand is not None:
                scored.append(cand)

        scored.sort(key=lambda c: -c[2])
        return scored[:k]

    def _score_candidate(
        self, cnt, tophat: np.ndarray, motion: np.ndarray,
        hsv: np.ndarray, frame_h: int, frame_w: int,
    ) -> Optional[Candidate]:
        area = cv2.contourArea(cnt)
        if not (BGSUB["min_area"] <= area <= BGSUB["max_area"]):
            return None

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if cx < 2 or cx >= frame_w - 2 or cy < 2 or cy >= frame_h - 2:
            return None

        # TopHat value — how "small and bright" is this spot
        tv = float(tophat[cy, cx])
        if tv < 8:
            return None

        # Frame motion — how much did this pixel move
        mv = float(motion[cy, cx])
        if mv < 4:
            return None

        # Brightness check
        v_val = int(hsv[cy, cx, 2])
        if v_val < BGSUB["min_brightness"]:
            return None

        # Circularity
        perim = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / (perim ** 2)) if perim > 1 else 0

        # Size: ball-typical (3-60px) gets full credit; larger penalized
        large_thresh = BGSUB.get("large_area_thresh", 100)
        size_score = 1.0 if area <= large_thresh else 0.3

        # Combined score: motion dominates
        score = (
            min(tv / 50.0, 1.5) * 0.30
            + min(mv / 50.0, 1.5) * 0.35
            + circ * 0.20
            + size_score * 0.15
        )
        return (cx, cy, float(score))
