"""
ROI (Region of Interest) filter for ball detections.

Broadcast tennis footage has scoreboards, logos, and banners
at the edges of the frame. These contain yellow/green elements
that trigger false positives. This filter rejects detections
outside the playable court region.

Default zones excluded:
  - Bottom 18% of frame  (scoreboard / sponsor banners)
  - Top 3% of frame      (broadcast banner)
  - Left/right 2%        (edge graphics)
"""
from typing import Optional, Tuple

from config.settings import ROI


class ROIFilter:
    """
    Reject detections outside the court region.

    Usage:
        roi = ROIFilter()
        detection = engine.predict(frames)
        detection = roi.filter(detection, frame_h, frame_w)
    """

    def __init__(self):
        self.top_pct    = ROI["exclude_top_pct"]
        self.bottom_pct = ROI["exclude_bottom_pct"]
        self.left_pct   = ROI["exclude_left_pct"]
        self.right_pct  = ROI["exclude_right_pct"]

    def filter(
        self,
        detection: Optional[Tuple[int, int, float]],
        frame_h: int,
        frame_w: int,
    ) -> Optional[Tuple[int, int, float]]:
        """Return detection if inside ROI, else None."""
        if detection is None:
            return None

        x, y, conf = detection

        y_min = int(frame_h * self.top_pct)
        y_max = int(frame_h * (1.0 - self.bottom_pct))
        x_min = int(frame_w * self.left_pct)
        x_max = int(frame_w * (1.0 - self.right_pct))

        if y_min <= y <= y_max and x_min <= x <= x_max:
            return detection

        return None

    def get_bounds(self, frame_h: int, frame_w: int) -> dict:
        """Return pixel boundaries (useful for debug overlay)."""
        return {
            "y_min": int(frame_h * self.top_pct),
            "y_max": int(frame_h * (1.0 - self.bottom_pct)),
            "x_min": int(frame_w * self.left_pct),
            "x_max": int(frame_w * (1.0 - self.right_pct)),
        }