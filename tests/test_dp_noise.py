"""Tests for differential privacy."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel_edge.privacy.dp_noise import DPNoiseInjector
from sentinel_edge.privacy.gradient_delta import GradientDeltaComputer


class TestDPNoiseInjector:
    def test_noise_is_added(self):
        dp = DPNoiseInjector(epsilon=0.3)
        gradient = np.ones(100)
        noised = dp.add_noise(gradient, n_local_samples=50)
        assert not np.allclose(gradient, noised)

    def test_noise_scale_with_epsilon(self):
        """Lower epsilon = more noise."""
        gradient = np.ones(1000)

        dp_high_privacy = DPNoiseInjector(epsilon=0.1)
        dp_low_privacy = DPNoiseInjector(epsilon=1.0)

        noised_high = dp_high_privacy.add_noise(gradient, n_local_samples=50)
        noised_low = dp_low_privacy.add_noise(gradient, n_local_samples=50)

        # Higher privacy (lower epsilon) should have more deviation
        deviation_high = np.std(noised_high - gradient)
        deviation_low = np.std(noised_low - gradient)
        assert deviation_high > deviation_low

    def test_sigma_computation(self):
        dp = DPNoiseInjector(epsilon=0.3, delta=1e-5)
        sigma = dp.compute_sigma(sensitivity=1.0)
        assert sigma > 0

    def test_gradient_clipping(self):
        dp = DPNoiseInjector()
        gradient = np.ones(100) * 10
        clipped = dp.clip_gradient(gradient)
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_noise_is_zero_mean(self):
        """Over many samples, noise should average to ~0."""
        dp = DPNoiseInjector(epsilon=0.3)
        gradient = np.zeros(1000)

        noised_sum = np.zeros(1000)
        n_trials = 100
        for _ in range(n_trials):
            noised_sum += dp.add_noise(gradient, n_local_samples=50)

        avg_noise = noised_sum / n_trials
        assert np.abs(avg_noise.mean()) < 0.5  # Should be near zero


class TestGradientDeltaComputer:
    def test_compute_delta(self):
        computer = GradientDeltaComputer()
        local = np.array([1.0, 2.0, 3.0])
        glob = np.array([0.5, 1.5, 2.5])
        delta = computer.compute_delta(local, glob)
        np.testing.assert_array_almost_equal(delta, [0.5, 0.5, 0.5])

    def test_apply_delta(self):
        computer = GradientDeltaComputer()
        glob = np.array([1.0, 2.0, 3.0])
        delta = np.array([0.1, 0.2, 0.3])
        result = computer.apply_delta(glob, delta)
        np.testing.assert_array_almost_equal(result, [1.1, 2.2, 3.3])
