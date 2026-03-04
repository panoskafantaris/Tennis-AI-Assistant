"""
Diagnostic v7 — argmax with background subtraction.

The model predicts class ~140 for background and ~233 for the ball.
We subtract the background level (mode) to isolate the ball signal.

Run: python diagnose.py --source tennis_match_2.mp4
"""
import sys
import os
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from core.model_loader import load_model
from config.settings import TRACKNET

os.makedirs("output", exist_ok=True)


def preprocess(frames, H, W, device):
    """BGR->RGB, resize, float32 [0,255]."""
    channels = []
    for bgr in frames:
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)
        channels.append(t)
    return torch.cat(channels, dim=0).unsqueeze(0).to(device)


def run(source):
    model = load_model()
    device = next(model.parameters()).device
    H = TRACKNET["input_height"]
    W = TRACKNET["input_width"]

    cap, frames = cv2.VideoCapture(source), []
    for _ in range(5):
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()

    orig_h, orig_w = frames[0].shape[:2]
    tensor = preprocess(frames[:3], H, W, device)

    with torch.no_grad():
        logits = model(tensor)[0]  # [256, H, W]

    # Argmax heatmap: raw class predictions per pixel
    heatmap = logits.argmax(dim=0).cpu().numpy().astype(np.float32)

    # ---- Background subtraction ----
    # The mode (most common value) IS the background level
    hist, bin_edges = np.histogram(heatmap.ravel(), bins=256, range=(0, 256))
    bg_level = float(bin_edges[hist.argmax()])

    # Residual: subtract background, clip negatives
    residual = np.clip(heatmap - bg_level, 0, None)

    print(f"Raw heatmap : min={heatmap.min():.0f}  max={heatmap.max():.0f}  "
          f"mean={heatmap.mean():.1f}")
    print(f"Background  : mode={bg_level:.0f}")
    print(f"Residual    : min={residual.min():.1f}  max={residual.max():.1f}  "
          f"mean={residual.mean():.2f}")

    r_max = residual.max()
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7]:
        count = int((residual > frac * r_max).sum())
        print(f"  Residual > {frac:.0%} of max ({frac * r_max:.1f}): "
              f"{count} pixels")

    # ---- Find ball peak in residual ----
    flat_idx = residual.argmax()
    py, px = divmod(int(flat_idx), W)
    peak_val = residual[py, px]
    ox = int(px * orig_w / W)
    oy = int(py * orig_h / H)
    print(f"\nResidual peak: {peak_val:.1f}  at ({px},{py}) -> orig ({ox},{oy})")

    # ---- Save residual heatmap ----
    if r_max > 0:
        norm_res = (residual / r_max * 255).astype(np.uint8)
    else:
        norm_res = np.zeros_like(residual, dtype=np.uint8)

    cv2.imwrite("output/residual_heatmap.png",
                cv2.applyColorMap(norm_res, cv2.COLORMAP_JET))

    # ---- Overlay residual on frame ----
    frame_vis = frames[2].copy()
    res_full = cv2.resize(norm_res, (orig_w, orig_h))
    heat_color = cv2.applyColorMap(res_full, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_vis, 0.6, heat_color, 0.4, 0)
    cv2.circle(overlay, (ox, oy), 14, (0, 255, 255), 3)
    cv2.putText(overlay, f"Ball ({ox},{oy}) res={peak_val:.0f}",
                (ox + 18, oy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    cv2.imwrite("output/residual_overlay.png", overlay)

    # ---- Blob detection at multiple thresholds ----
    print(f"\n{'='*55}")
    print(f"  Blob analysis at multiple thresholds")
    print(f"{'='*55}")

    for thresh_frac in [0.3, 0.5, 0.7]:
        thresh_val = int(thresh_frac * 255)
        _, binary = cv2.threshold(norm_res, thresh_val, 255, cv2.THRESH_BINARY)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        blobs = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx = int(stats[i, cv2.CC_STAT_LEFT]
                     + stats[i, cv2.CC_STAT_WIDTH] / 2)
            cy = int(stats[i, cv2.CC_STAT_TOP]
                     + stats[i, cv2.CC_STAT_HEIGHT] / 2)
            peak = float(residual[labels == i].max())
            blobs.append((area, cx, cy, peak))

        blobs.sort(key=lambda b: b[3], reverse=True)
        print(f"\nThreshold {thresh_frac:.0%} of max ({thresh_frac * r_max:.1f}):"
              f"  {len(blobs)} blobs")
        for area, cx, cy, pk in blobs[:8]:
            ocx = int(cx * orig_w / W)
            ocy = int(cy * orig_h / H)
            tag = ""
            if area <= 50:
                tag = " <-- SMALL"
            if area <= 200 and pk > 0.7 * r_max:
                tag = " <-- BALL?"
            print(f"  area={area:5d}  peak={pk:5.1f}  "
                  f"({cx},{cy}) -> orig ({ocx},{ocy}){tag}")

        if thresh_frac == 0.5:
            cv2.imwrite("output/residual_binary.png", binary)

    # ---- Also check known ball location from debug ----
    # Ball was at ~(1127, 341) in 1920x1080 = ~(375, 114) in 640x360
    bx, by = int(1127 * W / orig_w), int(341 * H / orig_h)
    if 0 <= bx < W and 0 <= by < H:
        ball_raw = heatmap[by, bx]
        ball_res = residual[by, bx]
        print(f"\nAt known ball location ({bx},{by}):")
        print(f"  Raw heatmap: {ball_raw:.0f}  Residual: {ball_res:.1f}")

    print(f"\nSaved -> output/residual_heatmap.png, "
          f"residual_overlay.png, residual_binary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    run(p.parse_args().source)