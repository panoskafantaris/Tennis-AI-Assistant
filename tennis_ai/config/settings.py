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
    "hsv_lower"      : (15, 40, 140),
    "hsv_upper"      : (80, 255, 255),
    "min_radius_frac": 0.002,
    "max_radius_frac": 0.03,
    "min_circularity": 0.30,
    "diff_thresh"    : 8,
    "morph_kernel"   : 3,
}

# ── Background Subtraction detector ──────────────────────────────
BGSUB = {
    "bg_samples"     : 40,
    "diff_thresh"    : 12,
    "min_area"       : 3,
    "max_area"       : 200,
    "min_brightness" : 130,
    "max_saturation" : 200,
    "min_diff_score" : 20,
    "candidate_limit": 10,
    "large_area_thresh": 100,
}

# ── Color-based ball detection (blue court enhancement) ──────────
COLOR_BOOST = {
    "enabled"         : True,
    "hsv_lower"       : (20, 80, 140),   # yellow-green ball
    "hsv_upper"       : (55, 255, 255),
    "min_area"        : 3,
    "max_area"        : 150,
    "min_circularity" : 0.25,
    "weight"          : 0.35,   # blend weight for color score in ensemble
}

# ── Player exclusion mask ─────────────────────────────────────────
PLAYER_MASK = {
    "enabled"         : True,
    "min_player_area" : 4000,   # minimum contour area to be a "player"
    "margin_px"       : 25,     # expand player bbox by this many pixels
    "color_gate"      : 0.03,   # min ball color match to survive near-player
    "bg_thresh"       : 35,     # threshold for player detection in bg_diff
}

# ── Scene cut detection ──────────────────────────────────────────
SCENE_CUT = {
    "enabled"         : True,
    "hist_threshold"  : 0.30,   # L1 histogram distance for hard cut
    "pixel_threshold" : 800.0,  # MSE threshold for pixel difference
    "cooldown_frames" : 5,      # min frames between detected cuts
    "bg_rebuild_frames": 20,    # frames to collect for new background
}

# ── Kalman filter ─────────────────────────────────────────────────
KALMAN = {
    "process_noise"   : 50.0,
    "measurement_noise": 10.0,
    "gate_distance"   : 250,    # WIDENED from 200 — allow larger search area
    "init_covariance" : 500.0,
    "min_confidence"  : 0.3,
}

# ── Ensemble detector tuning ─────────────────────────────────────
ENSEMBLE = {
    "init_window"      : 5,     # Original working value (was 5)
    "init_min_disp"    : 25,    # Original working value (was 25)
    "static_limit"     : 10,    # Original working value (was 10)
    "top_k"            : 5,     # Original working value (was 5)
    "reacquire_gap"    : 10,    # Slightly faster re-acquisition (was 12)
    "blue_surround_min": 0.55,  # Min blue ratio in ring around detection
}

# ── Interpolation ────────────────────────────────────────────────
INTERP = {
    "max_gap"         : 20,     # INCREASED from 15 — bridge longer gaps
    "min_anchors"     : 2,
    "smoothing_window": 5,
    "confidence_decay": 0.85,
}

# ── Tracker ───────────────────────────────────────────────────────
TRACKER = {
    "max_missing_frames" : 12,  # INCREASED from 8 — longer trail persistence
    "trail_length"       : 30,
    "min_confidence"     : 0.10,  # LOWERED from 0.15 — accept weaker detections
}

# ── Velocity filter ──────────────────────────────────────────────
VELOCITY = {
    "max_px_per_frame" : 300,   # INCREASED from 250 — allow faster movements
    "min_detections"   : 2,     # REDUCED from 3 — faster startup
    "history_len"      : 10,
}

# ── ROI filter (basic rectangle — used as fallback) ──────────────
ROI = {
    "top"    : 0.12,    # REDUCED from 0.15 — allow detections closer to top
    "bottom" : 0.15,    # REDUCED from 0.18
    "left"   : 0.02,
    "right"  : 0.02,
}

# ── Court zone filter (polygon — primary spatial filter) ─────────
COURT_ZONE = {
    "enabled"        : True,
    "margin_px"      : 50,      # INCREASED from 40
    "margin_top_px"  : 70,      # INCREASED from 50 — more room for lobs/serves
    "manual_polygon" : {
        "top_left"     : (0.14, 0.14),  # WIDENED from (0.16, 0.17)
        "top_right"    : (0.74, 0.14),  # WIDENED from (0.72, 0.17)
        "bottom_right" : (0.85, 0.75),  # WIDENED from (0.80, 0.70)
        "bottom_left"  : (0.05, 0.75),  # WIDENED from (0.08, 0.70)
    },
}

# ── Stationarity filter (reject static false positives) ──────────
STATIONARITY = {
    "window"           : 10,    # INCREASED from 8 — more patience
    "min_history"      : 5,     # INCREASED from 4
    "max_static_frames": 8,     # INCREASED from 6 — less aggressive
    "radius"           : 15,
    "blacklist_ttl"    : 300,   # NEW: blacklist entries expire after N frames
}

# ── Visualization ────────────────────────────────────────────────
VIZ = {
    "ball_color"      : (0, 255, 255),
    "trail_color"     : (0, 165, 255),
    "predicted_color" : (255, 100, 100),
    "text_color"      : (255, 255, 255),
    "ball_radius"     : 6,
    "trail_fade"      : True,
    "display_width"   : 1280,
}