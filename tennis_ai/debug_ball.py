"""
Samples HSV + motion at the known ball location and saves mask images.
Run: python debug_ball.py --source tennis_match_2.mp4
"""
import sys, os, argparse
import cv2
import numpy as np
sys.path.insert(0, ".")
os.makedirs("output", exist_ok=True)

# Ball is visible at ~(855, 258) in the 1456x816 overlay image.
# The overlay was resized from original 1920x1080, so scale back:
# x = 855 * (1920/1456) = ~1127,  y = 258 * (1080/816) = ~341
BALL_X, BALL_Y = 1127, 341   # approximate ball pixel in original frame

def run(source):
    cap, frames = cv2.VideoCapture(source), []
    for _ in range(5):
        ret, f = cap.read()
        if ret: frames.append(f)
    cap.release()

    f1, f2, f3 = frames[0], frames[1], frames[2]
    h, w = f3.shape[:2]
    print(f"Frame size: {w}x{h}")
    print(f"Sampling ball at pixel ({BALL_X}, {BALL_Y})")

    # ── HSV at ball pixel ─────────────────────────────────────────────────
    patch = f3[max(0,BALL_Y-5):BALL_Y+5, max(0,BALL_X-5):BALL_X+5]
    hsv_frame = cv2.cvtColor(f3, cv2.COLOR_BGR2HSV)
    hsv_patch = hsv_frame[max(0,BALL_Y-5):BALL_Y+5, max(0,BALL_X-5):BALL_X+5]
    bgr_val   = f3[BALL_Y, BALL_X].tolist()
    hsv_val   = hsv_frame[BALL_Y, BALL_X].tolist()

    print(f"\nBall pixel BGR : {bgr_val}")
    print(f"Ball pixel HSV : H={hsv_val[0]}  S={hsv_val[1]}  V={hsv_val[2]}")
    print(f"Patch HSV mean : {hsv_patch.mean(axis=(0,1)).astype(int).tolist()}")
    print(f"Patch HSV min  : {hsv_patch.min(axis=(0,1)).tolist()}")
    print(f"Patch HSV max  : {hsv_patch.max(axis=(0,1)).tolist()}")

    # ── Motion at ball pixel ──────────────────────────────────────────────
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2, f3)]
    d1 = cv2.absdiff(g[0], g[1])
    d2 = cv2.absdiff(g[1], g[2])
    print(f"\nMotion diff1 at ball : {int(d1[BALL_Y, BALL_X])}")
    print(f"Motion diff2 at ball : {int(d2[BALL_Y, BALL_X])}")

    # ── Save close-up crop ────────────────────────────────────────────────
    pad = 40
    crop = f3[max(0,BALL_Y-pad):BALL_Y+pad, max(0,BALL_X-pad):BALL_X+pad].copy()
    crop_big = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.circle(crop_big, (160, 160), 20, (0,255,255), 2)
    cv2.imwrite("output/ball_crop.png", crop_big)
    print("\nSaved → output/ball_crop.png")

    # ── Try widened HSV ranges and save masks ─────────────────────────────
    ranges = [
        ("tight",   np.array([20,80,80]),   np.array([45,255,255])),
        ("wide_h",  np.array([15,50,50]),   np.array([55,255,255])),
        ("widest",  np.array([10,30,30]),   np.array([65,255,255])),
        ("nosat",   np.array([15, 0, 80]),  np.array([55,255,255])),
    ]
    print("\nHSV mask pixel counts at ball location (10x10 patch):")
    for name, lo, hi in ranges:
        mask = cv2.inRange(hsv_frame, lo, hi)
        ball_px = int(mask[BALL_Y, BALL_X])
        patch_count = int(mask[max(0,BALL_Y-5):BALL_Y+5,
                               max(0,BALL_X-5):BALL_X+5].sum() // 255)
        total_white = int(mask.sum() // 255)
        print(f"  {name:10s}  ball_pixel={ball_px}  patch_hits={patch_count}/100  "
              f"total_mask_px={total_white}")
        out_mask = cv2.resize(mask, (w//4, h//4))
        cv2.imwrite(f"output/mask_{name}.png", out_mask)
    print("Saved masks to output/mask_*.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    run(p.parse_args().source)