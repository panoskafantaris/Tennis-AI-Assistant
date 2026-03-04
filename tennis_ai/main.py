"""
Tennis AI — Ball Tracking Pipeline.

Usage:
    python main.py --source video.mp4                          # hybrid (default)
    python main.py --source video.mp4 --detector tracknetv3    # TrackNetV3
    python main.py --source video.mp4 --detector tracknetv2    # TrackNetV2
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

from config.settings import OUTPUT_DIR, VIZ, V3
from core.base import BaseDetector
from tracking import BallTracker, FrameBuffer, ROIFilter, VelocityFilter
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
    p.add_argument("--source", required=True,
                   help="Video file or YouTube URL")
    p.add_argument("--detector", default="hybrid",
                   choices=["hybrid", "tracknetv2", "tracknetv3"])
    p.add_argument("--weights", default=None,
                   help="Custom weights path (optional)")
    p.add_argument("--save", default=None,
                   help="Output video path")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def build_detector(args, source: str) -> BaseDetector:
    """Factory for the chosen detection backend."""
    weights = Path(args.weights) if args.weights else None

    if args.detector == "hybrid":
        from core.hybrid import HybridDetector
        logger.info("Using HybridDetector (no weights)")
        return HybridDetector()

    if args.detector == "tracknetv3":
        from core.tracknet_v3 import TrackNetV3Detector
        logger.info("Loading TrackNetV3...")
        det = TrackNetV3Detector(weights)
        bg = _sample_background(source, V3["bg_samples"])
        det.set_background(bg)
        return det

    # tracknetv2
    from core.tracknet_v2 import TrackNetV2Detector
    logger.info("Loading TrackNetV2...")
    return TrackNetV2Detector(weights)


def _sample_background(source: str, n: int = 50) -> list:
    """Sample N frames evenly across video for V3 background."""
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 500
    frames = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    logger.info(f"Background: sampled {len(frames)} frames")
    return frames


def run(args: argparse.Namespace) -> None:
    detector = build_detector(args, args.source)
    tracker  = BallTracker()
    buf      = FrameBuffer(size=detector.window_size)
    roi      = ROIFilter()
    vel      = VelocityFilter()

    frame_idx = 0
    fps_timer = time.time()
    fps_disp  = 0.0

    with VideoReader(args.source) as vr:
        out_path = Path(args.save) if args.save else OUTPUT_DIR / "tracked.mp4"
        with VideoWriter(out_path, vr.fps, (vr.width, vr.height)) as vw:
            for frame in vr:
                frame_idx += 1
                if args.max_frames and frame_idx > args.max_frames:
                    break

                now = time.time()
                fps_disp = 0.9 * fps_disp + 0.1 / max(now - fps_timer, 1e-6)
                fps_timer = now

                buf.push(frame)
                detection = (
                    detector.predict(buf.get_window())
                    if buf.ready() else None
                )

                h, w = frame.shape[:2]
                detection = roi(detection, h, w)
                detection = vel(detection)
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


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(0)
