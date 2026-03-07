"""
Two-pass pipeline — detect → interpolate → render.

Pass 1: Run detector on all frames with court zone + stationarity filters.
Pass 2: Fill gaps with trajectory interpolation + smoothing.
Pass 3: Render annotated video with full coverage.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from config.settings import OUTPUT_DIR, VIZ, COURT_ZONE
from core.base import BaseDetector
from tracking import BallTracker, FrameBuffer, ROIFilter
from tracking.court_zone import CourtZoneFilter
from tracking.stationarity import StationarityFilter
from tracking.interpolator import (
    TrackPoint, interpolate_trajectory, smooth_trajectory,
)
from utils import draw_ball, draw_trail, draw_hud
from video import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def run_two_pass(
    source: str,
    detector: BaseDetector,
    save_path: str = None,
    max_frames: int = None,
) -> None:
    """Two-pass pipeline: detect → interpolate → render."""
    logger.info("=== Pass 1: Detection ===")
    buf = FrameBuffer(size=detector.window_size)
    court = CourtZoneFilter()
    static = StationarityFilter()
    roi = ROIFilter()

    raw_frames = []
    raw_points: list = []
    frame_idx = 0
    court_calibrated = False

    with VideoReader(source) as vr:
        fps = vr.fps
        size = (vr.width, vr.height)
        for frame in vr:
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Calibrate court zone on first frame
            if not court_calibrated and COURT_ZONE["enabled"]:
                court.calibrate(frame)
                court_calibrated = True

            raw_frames.append(frame)
            buf.push(frame)
            det = detector.predict(buf.get_window()) if buf.ready() else None
            h, w = frame.shape[:2]

            # Filter chain: court zone → stationarity
            if court_calibrated:
                det = court(det, h, w)
            else:
                det = roi(det, h, w)
            det = static(det)

            if det is not None:
                raw_points.append(TrackPoint(
                    frame_idx=frame_idx - 1, x=det[0], y=det[1],
                    confidence=det[2], is_detected=True, is_predicted=False,
                ))
            else:
                raw_points.append(None)

    detected = sum(1 for p in raw_points if p is not None)
    total = len(raw_points)
    logger.info(f"Pass 1: {detected}/{total} detected "
                f"({detected / max(total, 1) * 100:.1f}%)")

    # Pass 2: interpolate and smooth
    logger.info("=== Pass 2: Interpolation ===")
    filled = interpolate_trajectory(raw_points)
    filled = smooth_trajectory(filled)

    covered = sum(1 for p in filled if p is not None)
    logger.info(f"Pass 2: {covered}/{total} covered "
                f"({covered / max(total, 1) * 100:.1f}%)")

    # Pass 3: render
    logger.info("=== Rendering ===")
    out_path = Path(save_path) if save_path else OUTPUT_DIR / "tracked.mp4"
    tracker = BallTracker()

    with VideoWriter(out_path, fps, size) as vw:
        for i, frame in enumerate(raw_frames):
            pt = filled[i] if i < len(filled) else None
            det = (pt.x, pt.y, pt.confidence) if pt else None
            state = tracker.update(det)

            viz = frame.copy()
            if state.trail:
                viz = draw_trail(viz, state.trail)
            if state.detected:
                is_interp = pt is not None and not pt.is_detected
                viz = draw_ball(
                    viz, state.x, state.y, state.confidence,
                    predicted=is_interp,
                )
            viz = draw_hud(viz, i + 1, fps, state.detected, state.position)
            vw.write(viz)

    cv2.destroyAllWindows()
    logger.info(f"Done. {len(raw_frames)} frames -> {out_path}")
