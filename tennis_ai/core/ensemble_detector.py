"""
Ensemble detector — multi-candidate trajectory linking.
Collects top-K candidates per frame and uses Kalman prediction
to pick the trajectory-consistent one. Init requires displacement.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

from core.base import BaseDetector
from core.background_detector import BackgroundDetector
from core.hybrid import HybridDetector
from tracking.kalman import BallKalmanFilter
from config.settings import KALMAN

logger = logging.getLogger(__name__)

Detection = Optional[Tuple[int, int, float]]
_TOP_K = 5             # candidates per frame from bg detector
_INIT_MIN_DISP = 25    # min total displacement to confirm motion
_INIT_WINDOW = 5       # frames to collect init candidates
_STATIC_LIMIT = 10     # static this many frames → reset


class EnsembleDetector(BaseDetector):
    """Multi-candidate detector with trajectory-consistent selection."""

    def __init__(self):
        self._bg_det = BackgroundDetector()
        self._hybrid = HybridDetector()
        self._kalman = BallKalmanFilter()
        self._frame_count = 0

        # Init state
        self._init_candidates: List[List[Tuple[int, int, float]]] = []
        self._prev_pos: Optional[Tuple[int, int]] = None
        self._static_count = 0

    @property
    def window_size(self) -> int:
        return max(self._bg_det.window_size, self._hybrid.window_size)

    def set_background(self, frames: List[np.ndarray]) -> None:
        self._bg_det.set_background(frames)

    def predict(
        self, frames: List[np.ndarray],
    ) -> Detection:
        self._frame_count += 1

        # Collect all candidates
        all_cands = self._collect_candidates(frames)

        if not self._kalman.initialized:
            return self._init_phase(all_cands)
        return self._tracking_phase(all_cands)

    def _collect_candidates(
        self, frames: List[np.ndarray],
    ) -> List[Tuple[int, int, float, str]]:
        """Get candidates from all detection strategies."""
        cands = []
        for c in self._bg_det.predict_topk(frames, k=_TOP_K):
            cands.append((*c, "bgsub"))
        hybrid_det = self._hybrid.predict(frames)
        if hybrid_det is not None:
            cands.append((*hybrid_det, "hybrid"))
        return cands

    def _init_phase(
        self, candidates: List[Tuple[int, int, float, str]],
    ) -> Detection:
        """
        Collect candidates across _INIT_WINDOW frames.
        Find the candidate trajectory with greatest displacement.
        """
        frame_cands = [(c[0], c[1], c[2]) for c in candidates]
        self._init_candidates.append(frame_cands)

        if len(self._init_candidates) < _INIT_WINDOW:
            return None

        # Find best trajectory: greedily link nearest candidates
        best_traj = self._find_best_init_trajectory()
        self._init_candidates.clear()

        if best_traj is not None:
            last_x, last_y, last_s = best_traj[-1]
            self._kalman.update(last_x, last_y)
            self._prev_pos = (last_x, last_y)
            self._static_count = 0
            logger.info(
                f"Kalman init at ({last_x},{last_y}) frame {self._frame_count}"
            )
            return (last_x, last_y, last_s)
        return None

    def _find_best_init_trajectory(
        self,
    ) -> Optional[List[Tuple[int, int, float]]]:
        """
        From _INIT_WINDOW frames of candidates, find the trajectory
        with maximum displacement (= genuine ball motion).
        """
        best_disp = 0.0
        best_traj = None

        # Try each candidate in frame 0 as a starting point
        first_cands = self._init_candidates[0]
        if not first_cands:
            return None

        for start in first_cands:
            traj = [start]
            for frame_cands in self._init_candidates[1:]:
                if not frame_cands:
                    break
                # Find nearest candidate to last trajectory point
                lx, ly = traj[-1][0], traj[-1][1]
                nearest = min(
                    frame_cands,
                    key=lambda c: (c[0] - lx) ** 2 + (c[1] - ly) ** 2,
                )
                dist = ((nearest[0] - lx) ** 2 + (nearest[1] - ly) ** 2) ** 0.5
                if dist < 300:  # reasonable inter-frame distance
                    traj.append(nearest)

            if len(traj) < 3:
                continue

            # Total displacement
            disp = sum(
                ((traj[i][0] - traj[i - 1][0]) ** 2
                 + (traj[i][1] - traj[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(traj))
            )

            if disp > best_disp and disp >= _INIT_MIN_DISP:
                best_disp = disp
                best_traj = traj

        return best_traj

    def _tracking_phase(
        self, candidates: List[Tuple[int, int, float, str]],
    ) -> Detection:
        """Kalman active: predict → select nearest gated candidate."""
        predicted_pos = self._kalman.predict()

        # Gate all candidates and pick nearest to prediction
        gated = []
        for cx, cy, score, source in candidates:
            if self._kalman.gate(cx, cy):
                pred = self._kalman.position or (cx, cy)
                dist = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
                gated.append((cx, cy, score, dist))

        if gated:
            # Pick candidate nearest to Kalman prediction
            gated.sort(key=lambda c: c[3])
            bx, by, score = gated[0][0], gated[0][1], gated[0][2]
            self._kalman.update(bx, by)

            # Static check
            if self._prev_pos:
                disp = abs(bx - self._prev_pos[0]) + abs(by - self._prev_pos[1])
                if disp < 8:
                    self._static_count += 1
                else:
                    self._static_count = 0
            self._prev_pos = (bx, by)

            if self._static_count >= _STATIC_LIMIT:
                logger.info(f"Reset: static at ({bx},{by})")
                self.reset()
                return None

            return (bx, by, score)

        # No gated candidates — re-acquire after long gap
        if candidates and self._kalman.frames_since_update > 12:
            self.reset()
            return None

        # Fallback: Kalman prediction
        if predicted_pos and self._kalman.prediction_confidence() >= 0.25:
            px, py = predicted_pos
            return (px, py, self._kalman.prediction_confidence())

        return None

    def reset(self) -> None:
        self._kalman.reset()
        self._init_candidates.clear()
        self._prev_pos = None
        self._static_count = 0
