"""
Kalman filter for tennis ball tracking.

State vector: [x, y, vx, vy] — position and velocity.
Constant-velocity model with acceleration absorbed by process noise.

Provides:
  - predict()  : advance state one frame (even without detection)
  - update()   : correct state with a new measurement
  - gate()     : check if a detection is plausible given current state
  - position   : best estimate of current ball position
"""
import logging
from typing import Optional, Tuple

import numpy as np

from config.settings import KALMAN

logger = logging.getLogger(__name__)


class BallKalmanFilter:
    """2D Kalman filter for ball position and velocity."""

    def __init__(self):
        self._dt = 1.0  # one frame step
        self._initialized = False
        self._frames_since_update = 0

        # State: [x, y, vx, vy]
        self._x = np.zeros(4, dtype=np.float64)

        # State transition: constant velocity
        self._F = np.array([
            [1, 0, self._dt, 0],
            [0, 1, 0, self._dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix: we observe [x, y]
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Covariance
        init_p = KALMAN["init_covariance"]
        self._P = np.eye(4, dtype=np.float64) * init_p

        # Process noise
        q = KALMAN["process_noise"]
        dt = self._dt
        self._Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2],
        ], dtype=np.float64) * q

        # Measurement noise
        r = KALMAN["measurement_noise"]
        self._R = np.eye(2, dtype=np.float64) * (r ** 2)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def position(self) -> Optional[Tuple[int, int]]:
        """Current estimated position, or None if not initialized."""
        if not self._initialized:
            return None
        return (int(round(self._x[0])), int(round(self._x[1])))

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        if not self._initialized:
            return None
        return (float(self._x[2]), float(self._x[3]))

    @property
    def frames_since_update(self) -> int:
        return self._frames_since_update

    def predict(self) -> Optional[Tuple[int, int]]:
        """Advance state by one frame. Returns predicted position."""
        if not self._initialized:
            return None

        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self._frames_since_update += 1
        return self.position

    def update(self, x: int, y: int) -> Tuple[int, int]:
        """
        Correct state with measurement (x, y).
        If not initialized, initializes at this position.
        Returns corrected position.
        """
        z = np.array([float(x), float(y)], dtype=np.float64)

        if not self._initialized:
            self._x[:2] = z
            self._x[2:] = 0.0  # zero initial velocity
            self._initialized = True
            self._frames_since_update = 0
            return (x, y)

        # Kalman update
        y_resid = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y_resid
        I4 = np.eye(4, dtype=np.float64)
        self._P = (I4 - K @ self._H) @ self._P

        self._frames_since_update = 0
        return self.position

    def gate(self, x: int, y: int) -> bool:
        """Check if (x, y) is within the gating distance of prediction."""
        if not self._initialized:
            return True  # accept anything when uninitialized

        pred = self.position
        dist = ((x - pred[0]) ** 2 + (y - pred[1]) ** 2) ** 0.5

        # Scale gate with frames since update (allow wider gate after gap)
        gate_dist = KALMAN["gate_distance"] * max(
            1, self._frames_since_update * 0.5,
        )
        return dist <= gate_dist

    def prediction_confidence(self) -> float:
        """Confidence in current prediction (decays with gap length)."""
        if not self._initialized:
            return 0.0
        if self._frames_since_update == 0:
            return 1.0
        # Exponential decay
        return max(0.05, 0.9 ** self._frames_since_update)

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._initialized = False
        self._frames_since_update = 0
        self._x = np.zeros(4, dtype=np.float64)
        init_p = KALMAN["init_covariance"]
        self._P = np.eye(4, dtype=np.float64) * init_p
