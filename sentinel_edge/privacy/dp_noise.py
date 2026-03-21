"""Differential privacy noise injection for federated learning.

Implements the Gaussian mechanism for (epsilon, delta)-differential
privacy, applied to gradient deltas before they are sent from an
edge device to the central aggregator.
"""

from __future__ import annotations

import math

import numpy as np


class DPNoiseInjector:
    """Calibrated Gaussian noise injector for gradient privacy.

    Uses the analytic Gaussian mechanism: given a function with L2
    sensitivity *S*, adding Gaussian noise with

        sigma = S * sqrt(2 * ln(1.25 / delta)) / epsilon

    satisfies (epsilon, delta)-differential privacy.

    Parameters
    ----------
    epsilon : float
        Privacy budget.  Smaller values yield stronger privacy but more
        noise (and slower convergence).  Typical range: 0.1 -- 1.0.
    delta : float
        Probability of privacy breach.  Must be negligibly small
        relative to the dataset size.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        delta: float = 1e-5,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_sigma(self, sensitivity: float) -> float:
        """Compute the noise standard deviation for a given sensitivity.

        Parameters
        ----------
        sensitivity : float
            The L2 sensitivity of the query / gradient computation.

        Returns
        -------
        float
            Standard deviation (sigma) for the Gaussian noise.
        """
        return (
            sensitivity
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.epsilon
        )

    def add_noise(
        self,
        gradient_delta: np.ndarray,
        n_local_samples: int,
    ) -> np.ndarray:
        """Add calibrated Gaussian noise to a gradient delta vector.

        The L2 sensitivity is taken as ``1 / n_local_samples`` under the
        assumption that per-sample gradient contributions have been
        clipped to unit norm (see :meth:`clip_gradient`).

        Parameters
        ----------
        gradient_delta : np.ndarray
            The (clipped) gradient delta to be transmitted.
        n_local_samples : int
            Number of local training samples used to compute the delta.
            Larger values reduce the sensitivity and therefore the noise.

        Returns
        -------
        np.ndarray
            Noisy gradient delta with the same shape and dtype as the
            input.
        """
        if n_local_samples <= 0:
            raise ValueError(
                f"n_local_samples must be positive, got {n_local_samples}"
            )

        sensitivity = 1.0 / n_local_samples
        sigma = self.compute_sigma(sensitivity)
        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=gradient_delta.shape,
        )
        return gradient_delta + noise.astype(gradient_delta.dtype)

    def clip_gradient(
        self,
        gradient: np.ndarray,
        max_norm: float = 1.0,
    ) -> np.ndarray:
        """Clip a gradient vector to bounded L2 sensitivity.

        Parameters
        ----------
        gradient : np.ndarray
            Raw gradient vector (any shape).
        max_norm : float
            Maximum allowed L2 norm.  Gradients exceeding this are
            scaled down proportionally.

        Returns
        -------
        np.ndarray
            Gradient with ``||g||_2 <= max_norm``.
        """
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm > max_norm and grad_norm > 0.0:
            return gradient * (max_norm / grad_norm)
        return gradient.copy()
