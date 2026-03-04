"""
Tennis AI — Ball Tracking Pipeline

Usage:
    python main.py --source video.mp4                          # hybrid detector (default)
    python main.py --source video.mp4 --detector tracknet     # TrackNet (needs good weights)
    python main.py --source "https://youtube.com/..." --save output/result.mp4
    python main.py --source video.mp4 --max-frames 200
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from config.settings import OUTPUT_DIR, VIDEO, TRACKNET_WEIGHTS
from tracking import BallTracker, FrameBuffer, HybridBallDetector
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
    p.add_argument("--source",   required=True,
                   help="Video file path OR YouTube URL")
    p.add_argument("--detector", default="hybrid",
                   choices=["hybrid", "tracknet"],
                   help="Detection backend (default: hybrid)")
    p.add_argument("--weights",  default=str(TRACKNET_WEIGHTS),
                   help="Path to TrackNet weights (only used with --detector tracknet)")
    p.add_argument("--save",     default=None,
                   help="Save annotated video to this path")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview window")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process only first N frames")
    return p.parse_args()


def build_engine(args):
    """Instantiate the chosen detection backend."""
    if args.detector == "hybrid":
        logger.info("🎾 Using HybridBallDetector (motion + color + shape)")
        return HybridBallDetector()

    # TrackNet path
    from core import load_model, TrackNetInference
    logger.info("🔧 Loading TrackNet model …")
    model = load_model(Path(args.weights))
    return TrackNetInference(model)


def run_pipeline(args: argparse.Namespace) -> None:
    engine  = build_engine(args)
    tracker = BallTracker()
    buf     = FrameBuffer()

    frame_idx = 0
    fps_timer = time.time()
    fps_disp  = 0.0

    with VideoReader(args.source) as vr:
        out_path = Path(args.save) if args.save else OUTPUT_DIR / "tracked_output.mp4"
        out_size = (vr.width, vr.height)

        with VideoWriter(out_path, fps=vr.fps, size=out_size) as vw:
            for frame in vr:
                frame_idx += 1
                if args.max_frames and frame_idx > args.max_frames:
                    logger.info(f"Stopped at frame {frame_idx - 1}.")
                    break

                now       = time.time()
                fps_disp  = 0.9 * fps_disp + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
                fps_timer = now

                buf.push(frame)
                detection = engine.predict(buf.get_window()) if buf.ready() else None
                state     = tracker.update(detection)

                viz = frame.copy()
                if state.trail:
                    viz = draw_trail(viz, state.trail)
                if state.detected:
                    viz = draw_ball(viz, state.x, state.y, state.confidence)
                viz = draw_hud(viz, frame_idx, fps_disp, state.detected, state.position)

                vw.write(viz)

                if not args.no_display:
                    if VIDEO["resize_output"]:
                        scale = VIDEO["display_width"] / viz.shape[1]
                        viz   = cv2.resize(viz, None, fx=scale, fy=scale)
                    cv2.imshow("Tennis AI — Ball Tracker (q to quit)", viz)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    cv2.destroyAllWindows()
    logger.info(f"✅ Done. {frame_idx} frames. Output → {out_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        sys.exit(0)