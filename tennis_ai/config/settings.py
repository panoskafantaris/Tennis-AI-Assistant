"""
Central configuration for Tennis AI.
All hyperparameters, paths, and constants live here.
To adapt for a new sport/model, only this file needs changes.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
OUTPUT_DIR  = ROOT_DIR / "output"

TRACKNET_WEIGHTS = WEIGHTS_DIR / "tracknet_v2.pt"

# ── TrackNet Model ────────────────────────────────────────────────────────────
TRACKNET = {
    "input_frames"   : 3,       # consecutive frames fed as input
    "input_height"   : 360,     # must match training resolution
    "input_width"    : 640,     # must match training resolution
    "heatmap_thresh" : 0.95,    # keep top (1-thresh)*100 % pixels as ball candidates
    "sigma"          : 5,       # gaussian sigma used during training
}

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE = "cuda"   # force GPU; falls back to cpu if unavailable (see device.py)
# FP16 inference — DISABLED for TrackNet.
# TrackNet uses BatchNorm with FP32 running stats; casting to FP16 causes
# numerical instability and collapses all predictions to class 0.
# The model is ~50MB so FP32 comfortably fits in 6GB VRAM.
USE_FP16 = False

# ── Video Processing ──────────────────────────────────────────────────────────
VIDEO = {
    "target_fps"     : 30,
    "resize_output"  : True,    # resize displayed output to fit screen
    "display_width"  : 1280,
}

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER = {
    "max_missing_frames" : 5,   # frames before trajectory resets
    "trail_length"       : 30,  # how many past positions to draw
    "min_confidence"     : 0.3,
}

# ── Visualization ─────────────────────────────────────────────────────────────
VIZ = {
    "ball_color"    : (0, 255, 255),   # BGR cyan
    "trail_color"   : (0, 165, 255),   # BGR orange
    "text_color"    : (255, 255, 255),
    "ball_radius"   : 6,
    "trail_fade"    : True,            # older trail points become transparent
}