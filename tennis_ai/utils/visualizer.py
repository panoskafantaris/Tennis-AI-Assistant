"""
Drawing helpers for ball position, trajectory trail, and HUD overlay.
All functions accept and return numpy BGR frames (OpenCV convention).
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from config.settings import VIZ, TRACKER


def draw_ball(
    frame: np.ndarray,
    x: int,
    y: int,
    confidence: float,
    color: Tuple[int, int, int] = None,
    radius: int = None,
) -> np.ndarray:
    color  = color  or VIZ["ball_color"]
    radius = radius or VIZ["ball_radius"]

    # outer glow ring
    cv2.circle(frame, (x, y), radius + 4, (0, 0, 0), 2)
    # filled ball
    cv2.circle(frame, (x, y), radius, color, -1)
    # confidence label
    cv2.putText(
        frame, f"{confidence:.2f}",
        (x + radius + 4, y - radius),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        VIZ["text_color"], 1, cv2.LINE_AA,
    )
    return frame


def draw_trail(
    frame: np.ndarray,
    trail: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Draw a fading trajectory trail behind the ball.
    Older points are more transparent / smaller.
    """
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        alpha  = (i + 1) / n                     # 0 → transparent, 1 → opaque
        radius = max(1, int(VIZ["ball_radius"] * alpha * 0.6))

        if VIZ["trail_fade"]:
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), radius, VIZ["trail_color"], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            cv2.circle(frame, (x, y), radius, VIZ["trail_color"], -1)
    return frame


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    ball_detected: bool,
    ball_xy: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Heads-up display: frame counter, FPS, ball coordinates."""
    h, w = frame.shape[:2]
    status_color = (0, 255, 0) if ball_detected else (0, 0, 255)
    status_text  = "BALL TRACKED" if ball_detected else "BALL LOST"

    lines = [
        f"Frame : {frame_idx}",
        f"FPS   : {fps:.1f}",
        f"Status: {status_text}",
    ]
    if ball_xy:
        lines.append(f"Pos   : ({ball_xy[0]}, {ball_xy[1]})")

    for i, line in enumerate(lines):
        y_pos = 24 + i * 22
        # shadow
        cv2.putText(frame, line, (11, y_pos + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        # text
        color = status_color if "Status" in line else VIZ["text_color"]
        cv2.putText(frame, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame