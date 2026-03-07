"""
Trajectory interpolator — fills detection gaps with smooth curves.

Uses polynomial fitting between anchor detections to create
physically plausible ball paths through gap regions.

Two modes:
  1. Linear interpolation for short gaps (1-3 frames)
  2. Quadratic fit for longer gaps (uses gravity-like curvature)
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config.settings import INTERP

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """Single point in the trajectory."""
    frame_idx   : int
    x           : int
    y           : int
    confidence  : float
    is_detected : bool   # True = real detection, False = interpolated
    is_predicted: bool   # True = Kalman prediction, False = other source


def interpolate_trajectory(
    points: List[TrackPoint],
) -> List[TrackPoint]:
    """
    Fill gaps in a trajectory with interpolated positions.

    Args:
        points: sparse list indexed by frame, may contain None gaps.
    Returns:
        dense list with gaps filled by interpolation.
    """
    if not points:
        return points

    result = list(points)
    max_gap = INTERP["max_gap"]
    min_anchors = INTERP["min_anchors"]

    # Find detected points (anchors)
    anchors = [
        (i, p) for i, p in enumerate(result)
        if p is not None and p.is_detected
    ]
    if len(anchors) < 2:
        return result

    # Fill gaps between consecutive detected segments
    for a_idx in range(len(anchors) - 1):
        idx_a, pt_a = anchors[a_idx]
        idx_b, pt_b = anchors[a_idx + 1]
        gap_len = idx_b - idx_a - 1

        if gap_len <= 0 or gap_len > max_gap:
            continue

        # Gather anchor context (points before and after gap)
        before = _gather_anchors(anchors, a_idx, min_anchors, side="before")
        after = _gather_anchors(anchors, a_idx + 1, min_anchors, side="after")

        # Interpolate
        filled = _fill_gap(before, after, idx_a, idx_b, gap_len)
        for frame_idx, x, y, conf in filled:
            if 0 <= frame_idx < len(result):
                result[frame_idx] = TrackPoint(
                    frame_idx=frame_idx, x=x, y=y,
                    confidence=conf,
                    is_detected=False,
                    is_predicted=False,
                )

    return result


def _gather_anchors(
    anchors: list, center_idx: int, count: int, side: str,
) -> List[Tuple[int, int, int]]:
    """Gather up to `count` anchor points before/after center."""
    result = []
    if side == "before":
        start = max(0, center_idx - count + 1)
        for i in range(start, center_idx + 1):
            idx, pt = anchors[i]
            result.append((pt.frame_idx, pt.x, pt.y))
    else:
        end = min(len(anchors), center_idx + count)
        for i in range(center_idx, end):
            idx, pt = anchors[i]
            result.append((pt.frame_idx, pt.x, pt.y))
    return result


def _fill_gap(
    before: List[Tuple[int, int, int]],
    after: List[Tuple[int, int, int]],
    idx_a: int,
    idx_b: int,
    gap_len: int,
) -> List[Tuple[int, int, int, float]]:
    """
    Interpolate positions for frames between idx_a and idx_b.
    Returns list of (frame_idx, x, y, confidence).
    """
    all_pts = before + after
    frames = np.array([p[0] for p in all_pts], dtype=np.float64)
    xs = np.array([p[1] for p in all_pts], dtype=np.float64)
    ys = np.array([p[2] for p in all_pts], dtype=np.float64)

    # Choose interpolation order based on available points and gap
    n_pts = len(all_pts)
    if n_pts <= 2 or gap_len <= 3:
        order = 1  # linear
    else:
        order = min(2, n_pts - 1)  # quadratic max

    # Fit polynomial for x(t) and y(t)
    try:
        px = np.polyfit(frames, xs, order)
        py = np.polyfit(frames, ys, order)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to linear
        px = np.polyfit(frames, xs, 1)
        py = np.polyfit(frames, ys, 1)

    # Generate interpolated points
    decay = INTERP["confidence_decay"]
    results = []
    for fidx in range(idx_a + 1, idx_b):
        t = float(fidx)
        ix = int(round(np.polyval(px, t)))
        iy = int(round(np.polyval(py, t)))

        # Confidence decays toward the center of the gap
        dist_to_edge = min(fidx - idx_a, idx_b - fidx)
        conf = decay ** (gap_len // 2 - dist_to_edge + 1)
        results.append((fidx, ix, iy, float(conf)))

    return results


def smooth_trajectory(
    points: List[Optional[TrackPoint]],
) -> List[Optional[TrackPoint]]:
    """Apply moving average smoothing to reduce jitter."""
    window = INTERP["smoothing_window"]
    if window < 3:
        return points

    result = list(points)
    n = len(result)
    half = window // 2

    for i in range(n):
        if result[i] is None:
            continue
        # Gather neighbors
        xs, ys = [], []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if result[j] is not None:
                xs.append(result[j].x)
                ys.append(result[j].y)

        if len(xs) >= 3:
            smoothed_x = int(round(np.mean(xs)))
            smoothed_y = int(round(np.mean(ys)))
            result[i] = TrackPoint(
                frame_idx=result[i].frame_idx,
                x=smoothed_x, y=smoothed_y,
                confidence=result[i].confidence,
                is_detected=result[i].is_detected,
                is_predicted=result[i].is_predicted,
            )

    return result
