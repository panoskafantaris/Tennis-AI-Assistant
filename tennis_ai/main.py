"""
Tennis AI — Ball Tracking Pipeline

Usage:
    python main.py --source video.mp4                            # hybrid (default)
    python main.py --source video.mp4 --detector tracknetv3     # TrackNetV3
    python main.py --source video.mp4 --detector tracknet       # TrackNetV2
    python main.py --source "https://youtube.com/..." --save output/result.mp4
    python main.py --source video.mp4 --max-frames 200
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    OUTPUT_DIR, VIDEO, TRACKNET_WEIGHTS, TRACKNETV3,
)
from tracking import BallTracker, FrameBuffer, HybridBallDetector
from tracking.roi_filter import ROIFilter
from tracking.velocity_filter import VelocityFilter
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
                   help="Video file path OR YouTube URL")
    p.add_argument("--detector", default="hybrid",
                   choices=["hybrid", "tracknet", "tracknetv3"],
                   help="Detection backend (default: hybrid)")
    p.add_argument("--weights", default=None,
                   help="Path to weights file (auto-detected if omitted)")
    p.add_argument("--save", default=None,
                   help="Save annotated video to this path")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview window")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process only first N frames")
    return p.parse_args()


def estimate_background(source: str, n_samples: int = 50):
    """Sample n frames evenly across the video for background."""
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 500  # fallback

    indices = np.linspace(0, total - 1, n_samples).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    logger.info(f"  Background: sampled {len(frames)} frames")
    return frames


def build_engine(args, source_path: str):
    """Instantiate the chosen detection backend."""
    if args.detector == "hybrid":
        logger.info("🎾 Using HybridBallDetector")
        return HybridBallDetector(), 3

    if args.detector == "tracknetv3":
        from core.tracknetv3_loader import load_tracknetv3
        from core.tracknetv3_inference import TrackNetV3Inference

        logger.info("🔧 Loading TrackNetV3 …")
        model, param_dict = load_tracknetv3()
        engine = TrackNetV3Inference(model, param_dict)

        bg_frames = estimate_background(
            source_path, TRACKNETV3["bg_samples"]
        )
        engine.set_background(bg_frames)
        seq_len = param_dict.get("seq_len", 8)
        return engine, seq_len

    # TrackNetV2
    from core.model_loader import load_model
    from core.inference import TrackNetInference
    logger.info("🔧 Loading TrackNetV2 …")
    w = Path(args.weights) if args.weights else TRACKNET_WEIGHTS
    model = load_model(w)
    return TrackNetInference(model), 3


def resolve_source(source: str) -> str:
    """Return direct path/URL for VideoReader."""
    return source


def run_pipeline(args: argparse.Namespace) -> None:
    source = resolve_source(args.source)
    engine, buf_size = build_engine(args, source)
    tracker = BallTracker()
    buf = FrameBuffer(size=buf_size)
    roi = ROIFilter()
    vel = VelocityFilter()

    frame_idx = 0
    fps_timer = time.time()
    fps_disp = 0.0

    with VideoReader(source) as vr:
        out_path = Path(args.save) if args.save else (
            OUTPUT_DIR / "tracked_output.mp4"
        )
        with VideoWriter(out_path, fps=vr.fps,
                         size=(vr.width, vr.height)) as vw:
            for frame in vr:
                frame_idx += 1
                if args.max_frames and frame_idx > args.max_frames:
                    logger.info(f"Stopped at frame {frame_idx - 1}.")
                    break

                now = time.time()
                fps_disp = 0.9 * fps_disp + 0.1 / max(
                    now - fps_timer, 1e-6
                )
                fps_timer = now

                buf.push(frame)
                detection = (
                    engine.predict(buf.get_window())
                    if buf.ready() else None
                )
                # Reject detections on scoreboard / banners
                h, w = frame.shape[:2]
                detection = roi.filter(detection, h, w)
                # Reject physically impossible jumps
                detection = vel.filter(detection)
                state = tracker.update(detection)

                viz = frame.copy()
                if state.trail:
                    viz = draw_trail(viz, state.trail)
                if state.detected:
                    viz = draw_ball(
                        viz, state.x, state.y, state.confidence
                    )
                viz = draw_hud(
                    viz, frame_idx, fps_disp,
                    state.detected, state.position,
                )
                vw.write(viz)

                if not args.no_display:
                    if VIDEO["resize_output"]:
                        s = VIDEO["display_width"] / viz.shape[1]
                        viz = cv2.resize(viz, None, fx=s, fy=s)
                    cv2.imshow("Tennis AI (q=quit)", viz)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    cv2.destroyAllWindows()
    logger.info(f"✅ Done. {frame_idx} frames → {out_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        sys.exit(0)