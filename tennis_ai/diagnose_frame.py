"""
Diagnose a specific frame range — shows all heatmap candidates.
Run: python diagnose_frame.py --source tennis_match_2.mp4 --frame 1122

Seeks to the frame, runs V3 inference, shows ALL candidate
blobs in the heatmap (not just the top one).
"""
import sys, os, argparse, logging
import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
os.makedirs("output", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

from config.settings import TRACKNETV3
from core.tracknetv3_loader import load_tracknetv3
from core.tracknetv3_inference import TrackNetV3Inference
from utils.device import get_device


def run(source, target_frame):
    model, param_dict = load_tracknetv3()
    engine = TrackNetV3Inference(model, param_dict)
    seq_len = param_dict.get("seq_len", 8)

    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample background from across the video
    bg_frames = []
    for idx in np.linspace(0, total - 1, 50).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            bg_frames.append(f)
    engine.set_background(bg_frames)

    # Seek to target and read 8-frame window
    start = max(0, target_frame - seq_len + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    window = []
    for _ in range(seq_len):
        ret, f = cap.read()
        if ret:
            window.append(f)
    cap.release()

    if len(window) < seq_len:
        print(f"Only got {len(window)} frames, need {seq_len}")
        return

    oh, ow = window[-1].shape[:2]
    print(f"Frame {target_frame}, window [{start}:{start+seq_len}]")
    print(f"Frame size: {ow}x{oh}")

    # Get raw heatmap
    inp = engine._preprocess(window)
    with torch.no_grad():
        out = model(inp)
    hm = out[0, -1].cpu().numpy()

    print(f"Heatmap: min={hm.min():.4f}  max={hm.max():.4f}")
    print(f"Pixels > 0.3: {(hm > 0.3).sum()}")
    print(f"Pixels > 0.5: {(hm > 0.5).sum()}")

    # Find ALL blobs above threshold
    norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    _, binary = cv2.threshold(norm, 64, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    n_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    H, W = hm.shape
    print(f"\nAll blobs (threshold=64/255):")
    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        peak = float(hm[labels == i].max())
        ox = int(cx * ow / W)
        oy = int(cy * oh / H)
        blobs.append((peak, area, cx, cy, ox, oy))

    blobs.sort(key=lambda b: -b[0])  # sort by peak descending
    for peak, area, cx, cy, ox, oy in blobs:
        print(f"  peak={peak:.4f}  area={area:4d}  "
              f"model=({cx},{cy})  orig=({ox},{oy})")

    # Save overlay with ALL candidates marked
    frame_vis = window[-1].copy()
    hm_resized = cv2.resize(
        cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        (ow, oh)
    )
    jet = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_vis, 0.55, jet, 0.45, 0)

    for i, (peak, area, cx, cy, ox, oy) in enumerate(blobs):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.circle(overlay, (ox, oy), 14, color, 2)
        cv2.putText(overlay, f"#{i} p={peak:.2f} a={area}",
                    (ox + 16, oy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imwrite("output/frame_debug.png", overlay)
    print(f"\nSaved → output/frame_debug.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--frame", type=int, default=1122)
    run(p.parse_args().source, p.parse_args().frame)