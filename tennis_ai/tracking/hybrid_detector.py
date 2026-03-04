"""
HybridBallDetector — motion + color + shape based tennis ball detection.

Works without pre-trained weights.

Pipeline per frame-triple:
  1. Frame differencing  → isolate moving pixels (OR of two diffs, low threshold)
  2. HSV color filter    → keep yellow-green ball color range
  3. Combine motion + color mask
  4. Morphological clean → remove noise
  5. Contour analysis    → filter by area + circularity
  6. Pick best candidate → (x, y, confidence)
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List


# ── Tuneable constants ────────────────────────────────────────────────────────

# Tennis ball HSV range (yellow-green broadcast lighting)
# Confirmed from pixel sampling: ball HSV ≈ H=34, S=170, V=255
_HSV_LOWER = np.array([20,  80,  80],  dtype=np.uint8)
_HSV_UPPER = np.array([45, 255, 255],  dtype=np.uint8)

# Ball size bounds (fraction of frame height)
_MIN_RADIUS_FRAC = 0.003
_MAX_RADIUS_FRAC = 0.03

# Minimum circularity (1.0 = perfect circle)
_MIN_CIRCULARITY = 0.40

# Motion diff threshold — lowered to 10 to catch slow/stationary ball frames
_DIFF_THRESH = 10


def _color_mask(frame: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)


def _motion_mask(f1: np.ndarray, f2: np.ndarray, f3: np.ndarray) -> np.ndarray:
    """
    OR of two consecutive frame diffs — catches motion in either interval.
    Ball may barely move in one diff pair but clearly move in the other.
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2, f3)]
    d1   = cv2.absdiff(gray[0], gray[1])
    d2   = cv2.absdiff(gray[1], gray[2])
    # OR: pixel is moving if it moved in EITHER diff pair
    diff = cv2.max(d1, d2)
    _, motion = cv2.threshold(diff, _DIFF_THRESH, 255, cv2.THRESH_BINARY)
    return motion


def _circularity(contour) -> float:
    area  = cv2.contourArea(contour)
    if area < 1: return 0.0
    perim = cv2.arcLength(contour, True)
    if perim < 1: return 0.0
    return 4 * np.pi * area / (perim * perim)


class HybridBallDetector:
    """
    Frame-differencing + color + circularity ball detector.
    Call predict(frames) with a list of 3 BGR frames.
    Returns (x, y, confidence) or None.
    """

    def predict(
        self,
        frames: List[np.ndarray],
    ) -> Optional[Tuple[int, int, float]]:
        if len(frames) < 3:
            return None

        f1, f2, f3 = frames[-3], frames[-2], frames[-1]
        h, w       = f3.shape[:2]

        min_area = np.pi * max(2, int(h * _MIN_RADIUS_FRAC)) ** 2
        max_area = np.pi * int(h * _MAX_RADIUS_FRAC) ** 2

        # ── Masks ─────────────────────────────────────────────────────────
        color_m  = _color_mask(f3)
        motion_m = _motion_mask(f1, f2, f3)
        combined = cv2.bitwise_and(color_m, motion_m)

        # Morphological cleanup
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

        # ── Contour filtering ─────────────────────────────────────────────
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best: Optional[Tuple[int, int, float]] = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            circ = _circularity(cnt)
            if circ < _MIN_CIRCULARITY:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if best is None or circ > best[2]:
                best = (cx, cy, float(circ))

        return best