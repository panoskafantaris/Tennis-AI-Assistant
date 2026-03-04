"""
Diagnostic v3 — visualises the margin score map overlaid on the actual frame.
Run: python diagnose.py --source tennis_match_2.mp4
"""
import sys, os, argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from config.settings import TRACKNET_WEIGHTS, DEVICE
from core.tracknet_model import TrackNetV2
from utils.device import get_device

os.makedirs("output", exist_ok=True)


def load_model(device):
    model = TrackNetV2(input_frames=3)
    state = torch.load(str(TRACKNET_WEIGHTS), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess(frames, H, W, device):
    ch = []
    for bgr in frames:
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        ch.append(torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1))
    return torch.cat(ch, dim=0).unsqueeze(0).to(device)


def run(source):
    device = get_device(DEVICE)
    model  = load_model(device)
    H, W   = 360, 640

    cap, frames = cv2.VideoCapture(source), []
    for _ in range(5):
        ret, f = cap.read()
        if ret: frames.append(f)
    cap.release()

    orig_h, orig_w = frames[0].shape[:2]
    tensor = preprocess(frames[:3], H, W, device)

    with torch.no_grad():
        out = model(tensor)[0].float()   # [256, H, W]

    bg_logit    = out[0].cpu().numpy()          # [H, W]
    non_bg_max  = out[1:].max(dim=0).values.cpu().numpy()  # [H, W]
    margin_map  = non_bg_max - bg_logit         # [H, W]  positive = ball likely

    print(f"Margin map  : min={margin_map.min():.3f}  max={margin_map.max():.3f}  mean={margin_map.mean():.3f}")
    print(f"Margin > 0  : {(margin_map > 0).sum()} pixels")
    print(f"Margin > 1  : {(margin_map > 1).sum()} pixels")
    print(f"Margin > 2  : {(margin_map > 2).sum()} pixels")
    print(f"Margin > 5  : {(margin_map > 5).sum()} pixels")

    # Find peak of margin map
    flat_idx = margin_map.argmax()
    py, px   = divmod(int(flat_idx), W)
    peak_val = margin_map[py, px]
    print(f"\nPeak margin : {peak_val:.3f}  at model coords ({px}, {py})")
    # Scale to original
    ox = int(px * orig_w / W)
    oy = int(py * orig_h / H)
    print(f"Peak in original frame: ({ox}, {oy})")

    # ── Save margin map heatmap ─────────────────────────────────────────────
    # Clip at 0 (only show positive margin = ball-favoured pixels)
    pos_margin = np.clip(margin_map, 0, None)
    norm_m = cv2.normalize(pos_margin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("output/margin_map.png", cv2.applyColorMap(norm_m, cv2.COLORMAP_JET))
    print("Saved → output/margin_map.png")

    # ── Overlay on actual frame ─────────────────────────────────────────────
    frame_vis = frames[2].copy()
    # Resize margin map to original frame size and overlay
    margin_resized = cv2.resize(pos_margin, (orig_w, orig_h))
    norm_full      = cv2.normalize(margin_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color  = cv2.applyColorMap(norm_full, cv2.COLORMAP_JET)
    overlay        = cv2.addWeighted(frame_vis, 0.55, heatmap_color, 0.45, 0)
    # Draw peak location
    cv2.circle(overlay, (ox, oy), 12, (0, 255, 255), 3)
    cv2.putText(overlay, f"Peak ({ox},{oy}) margin={peak_val:.1f}",
                (ox + 15, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite("output/margin_overlay.png", overlay)
    print("Saved → output/margin_overlay.png")

    # ── Also try: show top-20 smallest high-scoring blobs ──────────────────
    _, binary = cv2.threshold(norm_m, int(255 * 0.90), 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_px = H * W
    blobs = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx   = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  / 2)
        cy   = int(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] / 2)
        peak = float(margin_map[labels == i].max())
        blobs.append((area, cx, cy, peak))

    blobs.sort(key=lambda b: b[0])   # sort by area ascending
    print(f"\nTop-10 smallest blobs (area, model_cx, model_cy, peak_margin):")
    for area, cx, cy, peak in blobs[:10]:
        print(f"  area={area:5d}  cx={cx:3d}  cy={cy:3d}  peak={peak:.2f}"
              f"  ({int(cx*orig_w/W)}, {int(cy*orig_h/H)}) orig")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    run(p.parse_args().source)