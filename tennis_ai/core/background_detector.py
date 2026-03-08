"""
Background Subtraction detector — finds the ball as a SMALL moving
foreground object via TopHat + frame motion.

Player exclusion: detects large foreground blobs (players) and rejects
small candidates inside their bounding boxes unless the candidate has
strong ball-color match (ball near player during a shot is allowed).
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base import BaseDetector
from config.settings import BGSUB, COLOR_BOOST, PLAYER_MASK
from tracking.player_mask import PlayerMask

logger = logging.getLogger(__name__)

Candidate = Tuple[int, int, float]


class BackgroundDetector(BaseDetector):
    """Ball detection via TopHat + motion + player exclusion."""

    def __init__(self):
        self._bg: Optional[np.ndarray] = None
        self._tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (21, 21),
        )
        self._ball_hsv_lo = np.array(COLOR_BOOST["hsv_lower"], dtype=np.uint8)
        self._ball_hsv_hi = np.array(COLOR_BOOST["hsv_upper"], dtype=np.uint8)
        self._player_mask = PlayerMask()
        self._color_gate = PLAYER_MASK.get("color_gate", 0.03)

    @property
    def window_size(self) -> int:
        return 3

    def set_background(self, frames: List[np.ndarray]) -> None:
        if not frames:
            return
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        self._bg = np.median(stack, axis=0).astype(np.uint8)
        logger.info(f"Background built from {len(frames)} frames")

    def predict(self, frames: List[np.ndarray]) -> Optional[Candidate]:
        candidates = self.predict_topk(frames, k=1)
        return candidates[0] if candidates else None

    def predict_topk(
        self, frames: List[np.ndarray], k: int = 5,
    ) -> List[Candidate]:
        if len(frames) < self.window_size or self._bg is None:
            return []

        current = frames[-1]
        prev = frames[-2]
        h, w = current.shape[:2]

        # 1. Background diff
        bg_diff = cv2.cvtColor(
            cv2.absdiff(current, self._bg), cv2.COLOR_BGR2GRAY,
        )

        # 2. Update player zones from background diff
        if PLAYER_MASK["enabled"]:
            self._player_mask.update(bg_diff)

        # 3. TopHat (kills large objects, keeps small)
        tophat = cv2.morphologyEx(bg_diff, cv2.MORPH_TOPHAT, self._tophat_kernel)

        # 4. Frame-to-frame motion
        motion = cv2.absdiff(
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        )

        # 5. Threshold and combine (AND logic)
        _, th_bin = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
        _, mo_bin = cv2.threshold(motion, 6, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(th_bin, mo_bin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # 6. Score candidates with player exclusion
        hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
        scored: List[Candidate] = []

        for cnt in contours:
            cand = self._score_candidate(cnt, tophat, motion, hsv, h, w)
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

        tv = float(tophat[cy, cx])
        mv = float(motion[cy, cx])
        if tv < 8 or mv < 4:
            return None

        v_val = int(hsv[cy, cx, 2])
        if v_val < BGSUB["min_brightness"]:
            return None

        # Circularity
        perim = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / (perim ** 2)) if perim > 1 else 0

        # Color match
        color_score = self._color_match(hsv, cx, cy)

        # Player exclusion: reject candidates on player body
        # UNLESS they have strong ball color (ball near player during shot)
        if PLAYER_MASK["enabled"] and self._player_mask.is_near_player(cx, cy):
            if color_score < self._color_gate:
                return None  # On player, no ball color → reject

        # Size scoring
        large_thresh = BGSUB.get("large_area_thresh", 100)
        size_score = 1.0 if area <= large_thresh else 0.3

        score = (
            min(tv / 50.0, 1.5) * 0.20
            + min(mv / 50.0, 1.5) * 0.25
            + circ * 0.15
            + size_score * 0.10
            + color_score * 0.30
        )
        return (cx, cy, float(score))

    def _color_match(self, hsv: np.ndarray, cx: int, cy: int) -> float:
        h, w = hsv.shape[:2]
        r = 4
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        mask = cv2.inRange(roi, self._ball_hsv_lo, self._ball_hsv_hi)
        return float(mask.sum() / 255) / max(mask.size, 1)