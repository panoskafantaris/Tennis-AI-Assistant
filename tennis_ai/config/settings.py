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

# ── TrackNet V2 Model ─────────────────────────────────────────────────────────
TRACKNET = {
    "input_frames"   : 3,
    "input_height"   : 360,
    "input_width"    : 640,
    "heatmap_thresh" : 0.95,
    "sigma"          : 5,
}

# ── TrackNet V3 Model ─────────────────────────────────────────────────────────
# From checkpoint param_dict: seq_len=8, bg_mode='concat'
# Input: (seq_len + 1) * 3 = 27 channels at 288×512
TRACKNETV3 = {
    "input_frames"   : 8,       # must match weights (seq_len=8)
    "input_height"   : 288,     # training resolution
    "input_width"    : 512,     # training resolution
    "bg_mode"        : "concat",
    "bg_samples"     : 50,      # frames sampled for background estimation
}

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
USE_FP16 = False    # TrackNet uses BatchNorm — keep FP32

# ── Video Processing ──────────────────────────────────────────────────────────
VIDEO = {
    "target_fps"     : 30,
    "resize_output"  : True,
    "display_width"  : 1280,
}

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER = {
    "max_missing_frames" : 5,
    "trail_length"       : 30,
    "min_confidence"     : 0.3,
}

# ── Velocity Filter (reject impossible jumps) ─────────────────────────────────
VELOCITY_FILTER = {
    "max_px_per_frame"          : 250,  # max pixels ball can move in 1 frame
    "min_detections_to_activate": 3,    # need N points before filtering
    "history_len"               : 10,   # rolling window of positions
}

# ── ROI Filter (exclude scoreboard / banners) ─────────────────────────────────
ROI = {
    "exclude_top_pct"    : 0.03,   # top 3%
    "exclude_bottom_pct" : 0.18,   # bottom 18% (scoreboard)
    "exclude_left_pct"   : 0.02,   # left 2%
    "exclude_right_pct"  : 0.02,   # right 2%
}

# ── Visualization ─────────────────────────────────────────────────────────────
VIZ = {
    "ball_color"    : (0, 255, 255),
    "trail_color"   : (0, 165, 255),
    "text_color"    : (255, 255, 255),
    "ball_radius"   : 6,
    "trail_fade"    : True,
}