"""Differential privacy noise injection for federated MLP updates."""
import numpy as np


class DPInjector:
    """Add calibrated Gaussian noise to MLP gradient deltas before transmission.

    Implements the Gaussian mechanism for (epsilon, delta)-differential privacy:
        1. Clip gradient delta to bounded L2 norm (max_grad_norm)
        2. Add Gaussian noise calibrated to the sensitivity
        3. Track cumulative privacy budget across rounds
    """

    def __init__(self, epsilon: float = 0.3, delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.rounds_participated = 0

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient delta to max_grad_norm for bounded sensitivity."""
        norm = np.linalg.norm(gradient)
        if norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / norm)
        return gradient

    def compute_sigma(self, n_local_samples: int) -> float:
        """Compute noise standard deviation.

        sensitivity = max_grad_norm / n_local_samples
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        sensitivity = self.max_grad_norm / max(n_local_samples, 1)
        sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def add_noise(self, gradient_delta: np.ndarray,
                  n_local_samples: int) -> tuple:
        """Apply DP to a gradient delta: clip, noise, return.

        Args:
            gradient_delta: Flat 1D array (MLP weight delta).
            n_local_samples: Number of local training samples.

        Returns:
            (noised_delta, sigma_used, epsilon_spent_this_round)
        """
        # Step 1: Clip the gradient delta to bounded L2 norm
        clipped = self.clip_gradient(gradient_delta)

        # Step 2: Compute calibrated noise scale
        sigma = self.compute_sigma(n_local_samples)

        # Step 3: Add Gaussian noise
        noise = np.random.normal(0.0, sigma, size=clipped.shape)
        noised_delta = clipped + noise

        # Track rounds
        self.rounds_participated += 1

        return noised_delta, sigma, self.epsilon

    def privacy_budget_spent(self, n_rounds: int = None) -> float:
        """Track cumulative privacy budget using advanced composition theorem.

        Total epsilon after k rounds:
            eps_total = epsilon * sqrt(2 * k * ln(1/delta)) + k * epsilon * (exp(epsilon) - 1)

        Args:
            n_rounds: Number of rounds to compute budget for.
                      If None, uses self.rounds_participated.
        """
        k = n_rounds if n_rounds is not None else self.rounds_participated
        if k == 0:
            return 0.0

        eps = self.epsilon
        d = self.delta

        term1 = eps * np.sqrt(2.0 * k * np.log(1.0 / d))
        term2 = k * eps * (np.exp(eps) - 1.0)
        return term1 + term2
