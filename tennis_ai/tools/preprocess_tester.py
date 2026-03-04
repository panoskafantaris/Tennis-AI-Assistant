"""
Preprocessing brute-force tester — finds the correct input config.

Tries every combination of:
  - Resolution : 288×512, 360×640, 320×640, 288×640
  - Color      : RGB, BGR
  - Normalise  : [0,1], [0,255], ImageNet mean/std

For each combo, runs a forward pass and reports the margin statistics.
The correct config will show POSITIVE margins at the ball location.

Run: python -m tools.preprocess_tester --source tennis_match_2.mp4
"""
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tracknet_model import TrackNetV2
from config.settings import TRACKNET_WEIGHTS
from utils.device import get_device


# ── Configs to test ───────────────────────────────────────────────
RESOLUTIONS = [
    (288, 512),   # original TrackNet paper resolution
    (360, 640),   # common broadcast aspect ratio
    (320, 640),   # 320 is divisible by 16
    (288, 640),   # hybrid
]

COLOR_MODES = ["rgb", "bgr"]

NORM_MODES = ["zero_one", "zero_255", "imagenet"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(device):
    model = TrackNetV2(input_frames=3)
    state = torch.load(
        str(TRACKNET_WEIGHTS), map_location="cpu", weights_only=False
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess(frames, H, W, color, norm, device):
    """Build input tensor with the given config."""
    channels = []
    for bgr in frames:
        resized = cv2.resize(bgr, (W, H))

        if color == "rgb":
            img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            img = resized  # keep BGR

        img = img.astype(np.float32)

        if norm == "zero_one":
            img /= 255.0
        elif norm == "zero_255":
            pass  # keep as [0, 255]
        elif norm == "imagenet":
            img /= 255.0
            img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        t = torch.from_numpy(img).permute(2, 0, 1)
        channels.append(t)

    return torch.cat(channels, dim=0).unsqueeze(0).to(device)


def compute_margin(model, tensor):
    """Forward pass → margin map stats."""
    with torch.no_grad():
        out = model(tensor)[0].float()

    bg = out[0].cpu().numpy()
    non_bg = out[1:].max(dim=0).values.cpu().numpy()
    margin = non_bg - bg

    pos_count = int((margin > 0).sum())
    return {
        "min": float(margin.min()),
        "max": float(margin.max()),
        "mean": float(margin.mean()),
        "pos_pixels": pos_count,
        "pos_pct": pos_count / margin.size * 100,
        "out_shape": tuple(out.shape),
        "margin": margin,
    }


def run(source):
    device = get_device("cuda")
    model = load_model(device)

    cap = cv2.VideoCapture(source)
    frames = []
    for _ in range(5):
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()

    if len(frames) < 3:
        print("❌ Need at least 3 frames from the video.")
        return

    print(f"Frame size: {frames[0].shape[1]}×{frames[0].shape[0]}")
    print(f"Testing {len(RESOLUTIONS)} resolutions × "
          f"{len(COLOR_MODES)} colors × {len(NORM_MODES)} norms "
          f"= {len(RESOLUTIONS) * len(COLOR_MODES) * len(NORM_MODES)} combos\n")

    results = []

    for H, W in RESOLUTIONS:
        for color in COLOR_MODES:
            for norm in NORM_MODES:
                tag = f"{H}×{W} {color:3s} {norm:10s}"
                try:
                    tensor = preprocess(frames[:3], H, W, color, norm, device)
                    stats = compute_margin(model, tensor)
                    results.append((tag, stats))

                    marker = "✅" if stats["pos_pixels"] > 10 else "  "
                    print(
                        f"{marker} {tag}  "
                        f"margin=[{stats['min']:+.1f}, {stats['max']:+.1f}]  "
                        f"pos={stats['pos_pixels']:6d} ({stats['pos_pct']:.2f}%)  "
                        f"out={stats['out_shape']}"
                    )
                except Exception as e:
                    print(f"   {tag}  ❌ Error: {e}")

    # ── Rank by best margin ───────────────────────────────────────
    results.sort(key=lambda r: r[1]["max"], reverse=True)

    print("\n" + "=" * 65)
    print("  TOP 5 CONFIGS (by peak margin)")
    print("=" * 65)
    for tag, stats in results[:5]:
        print(
            f"  {tag}  peak={stats['max']:+.2f}  "
            f"pos_px={stats['pos_pixels']}  "
            f"shape={stats['out_shape']}"
        )

    # ── Save best overlay ─────────────────────────────────────────
    best_tag, best_stats = results[0]
    margin = best_stats["margin"]
    H_out, W_out = margin.shape

    print(f"\n🏆 Best config: {best_tag}")
    print(f"   Peak margin: {best_stats['max']:.2f}")
    print(f"   Positive pixels: {best_stats['pos_pixels']}")

    if best_stats["pos_pixels"] > 0:
        orig_h, orig_w = frames[0].shape[:2]
        pos_margin = np.clip(margin, 0, None)
        norm_m = cv2.normalize(
            pos_margin, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_m, cv2.COLORMAP_JET)
        heatmap_full = cv2.resize(heatmap, (orig_w, orig_h))

        overlay = cv2.addWeighted(frames[2], 0.55, heatmap_full, 0.45, 0)

        flat_idx = margin.argmax()
        py, px = divmod(int(flat_idx), W_out)
        ox = int(px * orig_w / W_out)
        oy = int(py * orig_h / H_out)
        cv2.circle(overlay, (ox, oy), 14, (0, 255, 255), 3)

        cv2.imwrite("output/best_overlay.png", overlay)
        print(f"   Saved → output/best_overlay.png")
    else:
        print("   ⚠️  No positive margins in any config.")
        print("   The weights may not be trained for this input type.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    run(p.parse_args().source)