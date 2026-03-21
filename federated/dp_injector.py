"""Differential privacy noise injection for federated updates."""
import numpy as np


class DPInjector:
    """Add calibrated Gaussian noise to gradient deltas before transmission."""

    def __init__(self, epsilon: float = 0.3, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient to max_grad_norm for bounded sensitivity."""
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

    def add_noise(self, gradient_delta: np.ndarray, n_local_samples: int) -> tuple:
        """Add calibrated Gaussian noise. Returns (noised_delta, sigma_used)."""
        # First clip the gradient
        clipped = self.clip_gradient(gradient_delta)

        # Compute the noise scale
        sigma = self.compute_sigma(n_local_samples)

        # Add Gaussian noise
        noise = np.random.normal(0.0, sigma, size=clipped.shape)
        noised_delta = clipped + noise

        return noised_delta, sigma

    def privacy_budget_spent(self, n_rounds: int) -> float:
        """Track cumulative privacy budget using advanced composition theorem.
        Total epsilon after k rounds:
            eps_total = epsilon * sqrt(2 * k * ln(1/delta)) + k * epsilon * (exp(epsilon) - 1)
        """
        k = n_rounds
        eps = self.epsilon
        d = self.delta

        term1 = eps * np.sqrt(2.0 * k * np.log(1.0 / d))
        term2 = k * eps * (np.exp(eps) - 1.0)
        return term1 + term2
