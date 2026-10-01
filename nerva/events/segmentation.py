"""Event boundaries from prediction error (refactor stage H; RQ9 in docs/research_questions.md).

A small online predictor learns to predict the next world/self feature vector from the current one.
Moments where the prediction error is unusually high are candidate event boundaries. The idea of
segmenting experience at prediction-error peaks is motivated by Event Segmentation Theory (Zacks et al.
2007); this implementation is a NERVA engineering choice, not a model of human event perception.

  OnlinePredictor   recursive least squares, x_{t+1} ≈ A·[x_t, 1], forgetting factor LAMBDA
  BoundaryDetector  error e_t = ‖x_{t+1} − x̂_{t+1}‖ (features pre-scaled); boundary when
                    e_t > running mean + K_SIGMA · running std (exponential, HALF_LIFE frames) AND
                    e_t > MIN_ERROR (features are pre-scaled to O(1) units, so tiny noise outliers in a
                    very predictable stream are not boundaries), after WARMUP frames, with a refractory
                    period of REFRACTORY frames
Pure NumPy, O(d²) per step for d features: fits embedded hardware for small d.
"""

from __future__ import annotations

import numpy as np

LAMBDA = 0.995
K_SIGMA = 3.0
HALF_LIFE = 100  # frames
WARMUP = 30  # frames
REFRACTORY = 10  # frames
MIN_ERROR = 0.25  # scaled feature units


class OnlinePredictor:
    def __init__(self, dim: int, delta: float = 100.0, lam: float = LAMBDA):
        self.dim, self.lam = dim, lam
        self.A = np.zeros((dim, dim + 1))
        self.P = np.eye(dim + 1) * delta

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.A @ np.append(x, 1.0)

    def update(self, x: np.ndarray, x_next: np.ndarray) -> None:
        phi = np.append(x, 1.0)
        p_phi = self.P @ phi
        k = p_phi / (self.lam + phi @ p_phi)
        self.A += np.outer(x_next - self.A @ phi, k)
        self.P = (self.P - np.outer(k, p_phi)) / self.lam


class BoundaryDetector:
    """Feed feature vectors in time order; `step` returns True when x is the first frame after a boundary."""

    def __init__(self, dim: int, k_sigma: float = K_SIGMA, min_error: float = MIN_ERROR):
        self.pred = OnlinePredictor(dim)
        self.k_sigma, self.min_error = k_sigma, min_error
        self.mean, self.var = 0.0, 0.0
        self.n = 0
        self._last = -10**9
        self._prev: np.ndarray | None = None
        self.errors: list[float] = []

    def step(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        boundary = False
        if self._prev is not None:
            err = float(np.linalg.norm(x - self.pred.predict(self._prev)))
            self.errors.append(err)
            if (self.n >= WARMUP and err > max(self.min_error, self.mean + self.k_sigma * np.sqrt(self.var))
                    and self.n - self._last >= REFRACTORY):
                boundary, self._last = True, self.n
            a = 1.0 - 0.5 ** (1.0 / HALF_LIFE)
            d = err - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
            self.pred.update(self._prev, x)
            self.n += 1
        self._prev = x
        return boundary
