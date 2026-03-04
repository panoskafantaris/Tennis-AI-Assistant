"""
Diagnostic for TrackNetV3 — verify ball detection on first frames.
Run: python diagnose_v3.py --source tennis_match_2.mp4

Reads 60 frames: first 50 for background estimation, frames 3-10 for
inference (8-frame window). Saves heatmap overlay to output/.
"""
import sys, os, argparse, logging
import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
os.makedirs("output", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

from config.settings import TRACKNETV3, DEVICE
from core.tracknetv3_loader import load_tracknetv3
from core.tracknetv3_inference import TrackNetV3Inference
from utils.device import get_device


def run(source):
    device = get_device(DEVICE)
    model, param_dict = load_tracknetv3()
    engine = TrackNetV3Inference(model, param_dict)

    seq_len = param_dict.get("seq_len", 8)
    bg_count = TRACKNETV3["bg_samples"]

    # Read enough frames
    cap = cv2.VideoCapture(source)
    all_frames = []
    need = max(bg_count, seq_len + 10)
    for _ in range(need):
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    cap.release()
    print(f"Read {len(all_frames)} frames, size: "
          f"{all_frames[0].shape[1]}x{all_frames[0].shape[0]}")

    # Background estimation
    bg_indices = np.linspace(
        0, len(all_frames) - 1, min(bg_count, len(all_frames))
    ).astype(int)
    bg_frames = [all_frames[i] for i in bg_indices]
    engine.set_background(bg_frames)
    print(f"Background estimated from {len(bg_frames)} frames")

    # Test inference on several windows
    for start in range(0, min(len(all_frames) - seq_len, 5)):
        window = all_frames[start:start + seq_len]
        result = engine.predict(window)

        # Get raw heatmap for visualization
        inp = engine._preprocess(window)
        with torch.no_grad():
            out = model(inp)

        if out.dim() == 4:
            hm = out[0, -1].cpu().numpy()
        else:
            hm = out[0].cpu().numpy()

        print(f"\nWindow [{start}:{start+seq_len}]")
        print(f"  Output shape: {out.shape}")
        print(f"  Heatmap: min={hm.min():.4f}  max={hm.max():.4f}"
              f"  mean={hm.mean():.4f}")
        print(f"  Pixels > 0.5: {(hm > 0.5).sum()}")
        print(f"  Detection: {result}")

        if start == 0:
            _save_visuals(hm, window[-1], result)

    print("\nDone! Check output/v3_*.png")


def _save_visuals(hm, frame, detection):
    """Save heatmap, overlay, and binary mask."""
    H, W = hm.shape[:2]
    oh, ow = frame.shape[:2]

    # Normalise heatmap
    hm_clip = np.clip(hm, 0, None)
    if hm_clip.max() > 0:
        norm = (hm_clip / hm_clip.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(hm, dtype=np.uint8)

    jet = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite("output/v3_heatmap.png", jet)

    # Overlay on frame
    hm_big = cv2.resize(norm, (ow, oh))
    jet_big = cv2.applyColorMap(hm_big, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.55, jet_big, 0.45, 0)

    if detection:
        x, y, conf = detection
        cv2.circle(overlay, (x, y), 12, (0, 255, 255), 3)
        cv2.putText(overlay, f"({x},{y}) c={conf:.2f}",
                    (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    cv2.imwrite("output/v3_overlay.png", overlay)

    # Binary
    _, binary = cv2.threshold(norm, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output/v3_binary.png", binary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    run(p.parse_args().source)