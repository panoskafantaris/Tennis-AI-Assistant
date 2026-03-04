"""Drawing helpers for ball position, trajectory trail, and HUD overlay."""
import cv2
import numpy as np
from typing import List, Optional, Tuple

from config.settings import VIZ


def draw_ball(
    frame: np.ndarray, x: int, y: int, confidence: float,
) -> np.ndarray:
    r = VIZ["ball_radius"]
    cv2.circle(frame, (x, y), r + 4, (0, 0, 0), 2)
    cv2.circle(frame, (x, y), r, VIZ["ball_color"], -1)
    cv2.putText(
        frame, f"{confidence:.2f}", (x + r + 4, y - r),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, VIZ["text_color"], 1, cv2.LINE_AA,
    )
    return frame


def draw_trail(
    frame: np.ndarray, trail: List[Tuple[int, int]],
) -> np.ndarray:
    """Fading trajectory trail — older points are smaller/transparent."""
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / n
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
    detected: bool,
    ball_xy: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Heads-up display: frame counter, FPS, ball status."""
    color = (0, 255, 0) if detected else (0, 0, 255)
    status = "TRACKED" if detected else "LOST"

    lines = [f"Frame: {frame_idx}", f"FPS: {fps:.1f}", f"Ball: {status}"]
    if ball_xy:
        lines.append(f"Pos: ({ball_xy[0]}, {ball_xy[1]})")

    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(
            frame, line, (11, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
        )
        c = color if "Ball" in line else VIZ["text_color"]
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA,
        )
    return frame
