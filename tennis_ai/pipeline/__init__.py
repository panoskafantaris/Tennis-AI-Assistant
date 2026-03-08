"""
Two-pass pipeline — detect → interpolate → render.

Pass 1: Detect with scene-cut-aware background + court zone + stationarity.
Pass 2: Fill gaps with trajectory interpolation + smoothing.
Pass 3: Render annotated video.

Improvements:
  - Scene cut detection resets all state per scene
  - Background model rebuilt per scene
  - Wider interpolation gaps for better coverage
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from config.settings import OUTPUT_DIR, VIZ, COURT_ZONE, SCENE_CUT
from core.base import BaseDetector
from tracking import BallTracker, FrameBuffer, ROIFilter
from tracking.court_zone import CourtZoneFilter
from tracking.stationarity import StationarityFilter
from tracking.scene_cut import SceneCutDetector
from tracking.interpolator import (
    TrackPoint, interpolate_trajectory, smooth_trajectory,
)
from utils import draw_ball, draw_trail, draw_hud
from video import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def _rebuild_background(detector, frames: list, count: int = 30) -> None:
    """Rebuild background model from recent scene frames."""
    if not hasattr(detector, "set_background"):
        return
    if len(frames) < 5:
        return
    # Sample evenly from available frames
    n = min(count, len(frames))
    indices = np.linspace(0, len(frames) - 1, n).astype(int)
    bg_frames = [frames[i] for i in indices]
    detector.set_background(bg_frames)
    logger.info(f"Background rebuilt from {n} scene frames")


def _reset_pipeline(detector, court, static, buf) -> None:
    """Reset all pipeline state after a scene cut."""
    if hasattr(detector, "reset"):
        detector.reset()
    static.reset_full()
    buf.clear()
    logger.info("Pipeline state reset for new scene")


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
    scene_cut = SceneCutDetector()

    raw_frames = []
    raw_points: list = []
    frame_idx = 0
    court_calibrated = False

    # Per-scene frame accumulator for background rebuild
    scene_frames: list = []
    bg_rebuild_count = SCENE_CUT.get("bg_rebuild_frames", 20)

    with VideoReader(source) as vr:
        fps = vr.fps
        size = (vr.width, vr.height)
        for frame in vr:
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Scene cut detection
            if SCENE_CUT["enabled"] and scene_cut.check(frame):
                logger.info(f"Scene cut at frame {frame_idx} — resetting")
                _reset_pipeline(detector, court, static, buf)
                court_calibrated = False
                scene_frames.clear()

            # Accumulate scene frames for background
            scene_frames.append(frame)

            # Rebuild background after accumulating enough scene frames
            if len(scene_frames) == bg_rebuild_count:
                _rebuild_background(detector, scene_frames)

            # Calibrate court zone on first frame of each scene
            if not court_calibrated and COURT_ZONE["enabled"]:
                court.calibrate(frame)
                court_calibrated = True
                # Share court mask with ensemble detector for pre-filtering
                if hasattr(detector, 'set_court_mask') and court._mask is not None:
                    detector.set_court_mask(court._mask)

            raw_frames.append(frame)
            buf.push(frame)
            det = detector.predict(buf.get_window()) if buf.ready() else None
            h, w = frame.shape[:2]

            # Filter chain
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
    if scene_cut.total_cuts > 0:
        logger.info(f"Scene cuts detected: {scene_cut.total_cuts}")

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