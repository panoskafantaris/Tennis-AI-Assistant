"""
Ensemble detector — multi-candidate trajectory linking with Kalman.
Uses court zone polygon + blue-surround check to reject player-body FPs.
"""
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
from core.base import BaseDetector
from core.background_detector import BackgroundDetector
from core.hybrid import HybridDetector
from core.color_boost import ColorBallDetector, color_score_at
from tracking.kalman import BallKalmanFilter
from config.settings import KALMAN, ENSEMBLE, COLOR_BOOST

logger = logging.getLogger(__name__)
Detection = Optional[Tuple[int, int, float]]

# Blue court HSV range (broadcast tennis)
_BLUE_LO = np.array([90, 50, 50], dtype=np.uint8)
_BLUE_HI = np.array([130, 255, 255], dtype=np.uint8)


class EnsembleDetector(BaseDetector):
    def __init__(self):
        self._bg_det = BackgroundDetector()
        self._hybrid = HybridDetector()
        self._color_det = ColorBallDetector()
        self._kalman = BallKalmanFilter()
        self._frame_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._last_hsv: Optional[np.ndarray] = None
        self._frame_h = self._frame_w = 0
        self._court_mask: Optional[np.ndarray] = None
        self._init_candidates: List[List[Tuple[int, int, float]]] = []
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._static_count = 0

    @property
    def window_size(self) -> int:
        return max(self._bg_det.window_size, self._hybrid.window_size)

    def set_background(self, frames: List[np.ndarray]) -> None:
        self._bg_det.set_background(frames)

    def set_court_mask(self, mask: np.ndarray) -> None:
        self._court_mask = mask

    def predict(self, frames: List[np.ndarray]) -> Detection:
        self._frame_count += 1
        self._last_frame = frames[-1] if frames else None
        if self._last_frame is not None:
            self._frame_h, self._frame_w = self._last_frame.shape[:2]
            self._last_hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2HSV)
        cands = self._collect_candidates(frames)
        if not self._kalman.initialized:
            return self._init_phase(cands)
        return self._tracking_phase(cands)

    def _in_court(self, x: int, y: int) -> bool:
        if self._court_mask is not None:
            if 0 <= y < self._court_mask.shape[0] and 0 <= x < self._court_mask.shape[1]:
                return self._court_mask[y, x] > 0
            return False
        if self._frame_h == 0:
            return True
        return (self._frame_h * 0.13 < y < self._frame_h * 0.75
                and self._frame_w * 0.10 < x < self._frame_w * 0.85)

    def _on_blue_court(self, x: int, y: int) -> bool:
        """Check if detection is surrounded by blue court, not player body.
        Ball on open court: ring is >70% blue.
        Ball on player: ring is <50% blue."""
        if self._last_hsv is None:
            return True
        h, w = self._frame_h, self._frame_w
        r_in, r_out = 15, 40
        y1, y2 = max(0, y - r_out), min(h, y + r_out)
        x1, x2 = max(0, x - r_out), min(w, x + r_out)
        ring = self._last_hsv[y1:y2, x1:x2]
        hr, wr = ring.shape[:2]
        cy, cx = hr // 2, wr // 2
        yy, xx = np.ogrid[:hr, :wr]
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
        mask = (dist >= r_in) & (dist <= r_out)
        blue = cv2.inRange(ring, _BLUE_LO, _BLUE_HI)
        n = mask.sum()
        if n == 0:
            return True
        return float((blue > 0)[mask].sum()) / n >= ENSEMBLE.get("blue_surround_min", 0.55)

    def _collect_candidates(self, frames):
        raw = []
        for c in self._bg_det.predict_topk(frames, k=ENSEMBLE["top_k"]):
            raw.append((*c, "bgsub"))
        hybrid_det = self._hybrid.predict(frames)
        if hybrid_det is not None:
            raw.append((*hybrid_det, "hybrid"))
        if (COLOR_BOOST["enabled"] and self._last_frame is not None
                and self._kalman.initialized):
            for c in self._color_det.detect(self._last_frame, k=3):
                raw.append((*c, "color"))
        # Filter: court polygon → blue surround → color boost
        cands = [c for c in raw if self._in_court(c[0], c[1])]
        cands = [c for c in cands if self._on_blue_court(c[0], c[1])]
        if COLOR_BOOST["enabled"] and self._last_frame is not None:
            w = COLOR_BOOST["weight"]
            cands = [
                (cx, cy, s*(1-w)+color_score_at(self._last_frame, cx, cy)*w, src)
                for cx, cy, s, src in cands
            ]
        return cands

    def _init_phase(self, candidates) -> Detection:
        frame_cands = [(c[0], c[1], c[2]) for c in candidates]
        self._init_candidates.append(frame_cands)
        if len(self._init_candidates) < ENSEMBLE["init_window"]:
            return None
        best = self._find_best_init_trajectory()
        self._init_candidates.clear()
        if best is not None:
            lx, ly, ls = best[-1]
            self._kalman.update(lx, ly)
            self._prev_pos = (lx, ly)
            self._static_count = 0
            logger.info(f"Kalman init ({lx},{ly}) frame {self._frame_count}")
            return (lx, ly, ls)
        return None

    def _find_best_init_trajectory(self):
        first = self._init_candidates[0]
        if not first:
            return None
        best_disp, best_traj = 0.0, None
        for start in first:
            traj = [start]
            for fcands in self._init_candidates[1:]:
                if not fcands:
                    break
                lx, ly = traj[-1][0], traj[-1][1]
                nearest = min(fcands, key=lambda c: (
                    ((c[0]-lx)**2+(c[1]-ly)**2)**0.5 / max(c[2], 0.01)
                ))
                d = ((nearest[0]-lx)**2+(nearest[1]-ly)**2)**0.5
                if d < 300:
                    traj.append(nearest)
            if len(traj) < 3:
                continue
            disp = sum(
                ((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2)**0.5
                for i in range(1, len(traj))
            )
            if disp > best_disp and disp >= ENSEMBLE["init_min_disp"]:
                best_disp = disp
                best_traj = traj
        return best_traj

    def _tracking_phase(self, candidates) -> Detection:
        predicted_pos = self._kalman.predict()
        gated = []
        for cx, cy, score, _ in candidates:
            if self._kalman.gate(cx, cy):
                pred = self._kalman.position or (cx, cy)
                dist = ((cx-pred[0])**2+(cy-pred[1])**2)**0.5
                gated.append((cx, cy, score, dist))
        if gated:
            gated.sort(key=lambda c: c[3])
            bx, by, score = gated[0][0], gated[0][1], gated[0][2]
            self._kalman.update(bx, by)
            if self._prev_pos:
                d = abs(bx-self._prev_pos[0]) + abs(by-self._prev_pos[1])
                self._static_count = self._static_count+1 if d < 8 else 0
            self._prev_pos = (bx, by)
            if self._static_count >= ENSEMBLE["static_limit"]:
                self.reset()
                return None
            return (bx, by, score)
        if candidates and self._kalman.frames_since_update > ENSEMBLE["reacquire_gap"]:
            self.reset()
            return None
        if predicted_pos and self._kalman.prediction_confidence() >= 0.25:
            px, py = predicted_pos
            return (px, py, self._kalman.prediction_confidence())
        return None

    def reset(self) -> None:
        self._kalman.reset()
        self._init_candidates.clear()
        self._prev_pos = None
        self._static_count = 0