"""
Central configuration — single source of truth.
To adapt for a new sport or model, only this file needs changes.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
OUTPUT_DIR  = ROOT_DIR / "output"

TRACKNET_V2_WEIGHTS = WEIGHTS_DIR / "tracknet_v2.pt"
TRACKNET_V3_DIR     = WEIGHTS_DIR / "tracknetv3"
TRACKNET_V3_REPO    = ROOT_DIR / "tracknetv3_repo"

# ── Device ────────────────────────────────────────────────────────
DEVICE   = "cuda"
USE_FP16 = False  # TrackNet uses BatchNorm — keep FP32

# ── TrackNet V2 ───────────────────────────────────────────────────
V2 = {
    "input_frames" : 3,
    "input_height" : 360,
    "input_width"  : 640,
    "color_mode"   : "rgb",
}

V2_POST = {
    "heatmap_thresh" : 0.5,
    "min_blob_area"  : 1,
    "max_blob_area"  : 200,
}

# ── TrackNet V3 ───────────────────────────────────────────────────
V3 = {
    "input_frames" : 8,
    "input_height" : 288,
    "input_width"  : 512,
    "bg_mode"      : "concat",
    "bg_samples"   : 50,
}

V3_POST = {
    "peak_thresh"   : 0.45,
    "binary_thresh" : 128,
    "min_area"      : 1,
    "max_area"      : 40,
    "min_sharpness" : 0.015,
}

# ── Hybrid detector (no weights needed) ──────────────────────────
HYBRID = {
    "hsv_lower"      : (20, 80, 80),
    "hsv_upper"      : (45, 255, 255),
    "min_radius_frac": 0.003,
    "max_radius_frac": 0.03,
    "min_circularity": 0.40,
    "diff_thresh"    : 10,
}

# ── Tracker ───────────────────────────────────────────────────────
TRACKER = {
    "max_missing_frames" : 5,
    "trail_length"       : 30,
    "min_confidence"     : 0.3,
}

# ── Velocity filter ──────────────────────────────────────────────
VELOCITY = {
    "max_px_per_frame" : 250,
    "min_detections"   : 3,
    "history_len"      : 10,
}

# ── ROI filter (exclude scoreboard/banners) ──────────────────────
ROI = {
    "top"    : 0.03,
    "bottom" : 0.18,
    "left"   : 0.02,
    "right"  : 0.02,
}

# ── Visualization ────────────────────────────────────────────────
VIZ = {
    "ball_color"    : (0, 255, 255),
    "trail_color"   : (0, 165, 255),
    "text_color"    : (255, 255, 255),
    "ball_radius"   : 6,
    "trail_fade"    : True,
    "display_width" : 1280,
}
