"""
Tennis AI — Ball Tracking Pipeline.

Usage:
    python main.py --source video.mp4                          # ensemble (default)
    python main.py --source video.mp4 --detector hybrid        # hybrid only
    python main.py --source video.mp4 --detector tracknetv3    # TrackNetV3
    python main.py --source video.mp4 --two-pass               # full interpolation
    python main.py --source "https://youtube.com/..." --save out.mp4
    python main.py --source video.mp4 --max-frames 200
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config.settings import OUTPUT_DIR, VIZ, V3, BGSUB, COURT_ZONE, SCENE_CUT
from core.base import BaseDetector
from tracking import BallTracker, FrameBuffer, ROIFilter
from tracking.court_zone import CourtZoneFilter
from tracking.stationarity import StationarityFilter
from tracking.scene_cut import SceneCutDetector
from utils import draw_ball, draw_trail, draw_hud
from video import VideoReader, VideoWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tennis AI — Ball Tracker")
    p.add_argument("--source", required=True)
    p.add_argument("--detector", default="ensemble",
                   choices=["ensemble", "hybrid", "tracknetv2", "tracknetv3"])
    p.add_argument("--weights", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--two-pass", action="store_true",
                   help="Enable two-pass interpolation for gap filling")
    return p.parse_args()


def sample_background(source: str, n: int = 50, max_frame: int = None) -> list:
    """Sample N frames from the FIRST scene for background model."""
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 500
    if max_frame:
        total = min(total, max_frame)

    # Sample from first 30% of video to avoid cross-scene contamination
    sample_end = min(total, int(total * 0.3))
    sample_end = max(sample_end, min(n * 2, total))

    frames = []
    for idx in np.linspace(0, sample_end - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    logger.info(f"Background: sampled {len(frames)} frames (first {sample_end})")
    return frames


def build_detector(args, source: str) -> BaseDetector:
    """Factory for detection backend."""
    weights = Path(args.weights) if args.weights else None

    if args.detector == "ensemble":
        from core.ensemble_detector import EnsembleDetector
        logger.info("Using EnsembleDetector (bg-sub + hybrid + color + Kalman)")
        det = EnsembleDetector()
        bg = sample_background(source, BGSUB["bg_samples"], args.max_frames)
        det.set_background(bg)
        return det

    if args.detector == "hybrid":
        from core.hybrid import HybridDetector
        logger.info("Using HybridDetector")
        return HybridDetector()

    if args.detector == "tracknetv3":
        from core.tracknet_v3 import TrackNetV3Detector
        det = TrackNetV3Detector(weights)
        bg = sample_background(source, V3["bg_samples"])
        det.set_background(bg)
        return det

    from core.tracknet_v2 import TrackNetV2Detector
    return TrackNetV2Detector(weights)


def run_single_pass(args: argparse.Namespace) -> None:
    """Single-pass pipeline with scene-cut awareness."""
    detector = build_detector(args, args.source)
    tracker = BallTracker()
    buf = FrameBuffer(size=detector.window_size)
    court = CourtZoneFilter()
    static = StationarityFilter()
    roi = ROIFilter()
    scene_cut = SceneCutDetector()

    frame_idx = 0
    fps_timer = time.time()
    fps_disp = 0.0
    court_calibrated = False
    scene_frames: list = []

    with VideoReader(args.source) as vr:
        out_path = Path(args.save) if args.save else OUTPUT_DIR / "tracked.mp4"
        with VideoWriter(out_path, vr.fps, (vr.width, vr.height)) as vw:
            for frame in vr:
                frame_idx += 1
                if args.max_frames and frame_idx > args.max_frames:
                    break

                # Scene cut detection → reset everything
                if SCENE_CUT["enabled"] and scene_cut.check(frame):
                    logger.info(f"Scene cut at frame {frame_idx}")
                    if hasattr(detector, "reset"):
                        detector.reset()
                    static.reset_full()
                    buf.clear()
                    tracker.reset()
                    court_calibrated = False
                    scene_frames.clear()

                scene_frames.append(frame)
                bg_n = SCENE_CUT.get("bg_rebuild_frames", 20)
                if len(scene_frames) == bg_n and hasattr(detector, "set_background"):
                    detector.set_background(scene_frames)

                if not court_calibrated and COURT_ZONE["enabled"]:
                    court.calibrate(frame)
                    court_calibrated = True
                    if hasattr(detector, 'set_court_mask') and court._mask is not None:
                        detector.set_court_mask(court._mask)

                now = time.time()
                fps_disp = 0.9 * fps_disp + 0.1 / max(now - fps_timer, 1e-6)
                fps_timer = now

                buf.push(frame)
                detection = (
                    detector.predict(buf.get_window()) if buf.ready() else None
                )

                h, w = frame.shape[:2]
                if court_calibrated:
                    detection = court(detection, h, w)
                else:
                    detection = roi(detection, h, w)
                detection = static(detection)

                state = tracker.update(detection)

                viz = frame.copy()
                if state.trail:
                    viz = draw_trail(viz, state.trail)
                if state.detected:
                    viz = draw_ball(viz, state.x, state.y, state.confidence)
                viz = draw_hud(viz, frame_idx, fps_disp,
                               state.detected, state.position)
                vw.write(viz)

                if not args.no_display:
                    scale = VIZ["display_width"] / viz.shape[1]
                    small = cv2.resize(viz, None, fx=scale, fy=scale)
                    cv2.imshow("Tennis AI (q=quit)", small)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    cv2.destroyAllWindows()
    logger.info(f"Done. {frame_idx} frames -> {out_path}")


def run(args: argparse.Namespace) -> None:
    if args.two_pass:
        from pipeline import run_two_pass
        detector = build_detector(args, args.source)
        run_two_pass(args.source, detector, args.save, args.max_frames)
    else:
        run_single_pass(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(0)