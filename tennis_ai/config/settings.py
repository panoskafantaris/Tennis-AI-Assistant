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
    "hsv_lower"      : (15, 40, 140),   # widened: H down to 15, S down to 40
    "hsv_upper"      : (80, 255, 255),   # widened: H up to 80 for blue courts
    "min_radius_frac": 0.002,            # smaller min radius
    "max_radius_frac": 0.03,
    "min_circularity": 0.30,             # relaxed circularity
    "diff_thresh"    : 8,                # more sensitive motion
    "morph_kernel"   : 3,                # smaller kernel preserves small blobs
}

# ── Background Subtraction detector ──────────────────────────────
BGSUB = {
    "bg_samples"     : 40,        # frames to sample for background median
    "diff_thresh"    : 12,        # foreground threshold (lowered for sensitivity)
    "min_area"       : 3,         # minimum contour area (pixels)
    "max_area"       : 200,       # maximum contour area (pixels)
    "min_brightness" : 130,       # ball is bright (V channel) — relaxed
    "max_saturation" : 200,       # ball is not deeply saturated — relaxed
    "min_diff_score" : 20,        # minimum diff value at centroid — relaxed
    "candidate_limit": 10,        # max candidates per frame to score
    "large_area_thresh": 100,     # above this → penalize (probably player)
}

# ── Kalman filter ─────────────────────────────────────────────────
KALMAN = {
    "process_noise"   : 50.0,     # how much we trust the physics model
    "measurement_noise": 10.0,    # how noisy detections are (pixels)
    "gate_distance"   : 200,      # max pixels from prediction to accept
    "init_covariance" : 500.0,    # initial uncertainty
    "min_confidence"  : 0.3,      # below this, use prediction only
}

# ── Interpolation ────────────────────────────────────────────────
INTERP = {
    "max_gap"         : 15,       # max frames to interpolate across
    "min_anchors"     : 2,        # min detections on each side of gap
    "smoothing_window": 5,        # moving average window for smoothing
    "confidence_decay": 0.85,     # per-frame confidence decay in gaps
}

# ── Tracker ───────────────────────────────────────────────────────
TRACKER = {
    "max_missing_frames" : 8,     # increased from 5 for interpolation
    "trail_length"       : 30,
    "min_confidence"     : 0.15,  # lowered to accept Kalman predictions
}

# ── Velocity filter ──────────────────────────────────────────────
VELOCITY = {
    "max_px_per_frame" : 250,
    "min_detections"   : 3,
    "history_len"      : 10,
}

# ── ROI filter (basic rectangle — used as fallback) ──────────────
ROI = {
    "top"    : 0.15,      # 15% — clears all banners and logos
    "bottom" : 0.18,
    "left"   : 0.02,
    "right"  : 0.02,
}

# ── Court zone filter (polygon — primary spatial filter) ─────────
COURT_ZONE = {
    "enabled"       : True,
    "margin_top"    : 0.04,     # extra margin above topmost court line
    "margin_side"   : 0.03,     # extra margin beyond sidelines
    "manual_polygon": {         # fallback if auto-detection fails
        "top_left"     : (0.10, 0.14),   # (x_frac, y_frac) of frame
        "top_right"    : (0.80, 0.14),
        "bottom_right" : (0.85, 0.68),
        "bottom_left"  : (0.05, 0.68),
    },
}

# ── Stationarity filter (reject static false positives) ──────────
STATIONARITY = {
    "window"           : 8,     # frames to check for stationarity
    "min_history"      : 4,     # need this many frames to decide
    "max_static_frames": 6,     # static this long → blacklist
    "radius"           : 15,    # pixels — positions within this are "same"
}

# ── Visualization ────────────────────────────────────────────────
VIZ = {
    "ball_color"      : (0, 255, 255),
    "trail_color"     : (0, 165, 255),
    "predicted_color" : (255, 100, 100),   # blue-ish for predicted positions
    "text_color"      : (255, 255, 255),
    "ball_radius"     : 6,
    "trail_fade"      : True,
    "display_width"   : 1280,
}
