"""Tests for federated learning simulation."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.local_trainer import LocalTrainer, LocalTrainingBuffer
from federated.dp_injector import DPInjector
from federated.simulate import FederatedSimulation


class TestLocalTrainingBuffer:
    def test_add_and_size(self):
        buf = LocalTrainingBuffer(max_samples=10)
        buf.add(np.ones(5), 1)
        buf.add(np.zeros(5), 0)
        assert buf.size() == 2

    def test_max_capacity(self):
        buf = LocalTrainingBuffer(max_samples=3)
        for i in range(5):
            buf.add(np.ones(5) * i, i % 2)
        assert buf.size() == 3  # Should evict oldest

    def test_to_arrays(self):
        buf = LocalTrainingBuffer()
        buf.add(np.ones(5), 1)
        buf.add(np.zeros(5), 0)
        X, y = buf.to_arrays()
        assert X.shape == (2, 5)
        assert y.shape == (2,)


class TestDPInjector:
    def test_clip_gradient(self):
        dp = DPInjector(max_grad_norm=1.0)
        gradient = np.ones(100) * 10
        clipped = dp.clip_gradient(gradient)
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_add_noise_returns_tuple(self):
        dp = DPInjector()
        gradient = np.ones(50)
        noised, sigma = dp.add_noise(gradient, n_local_samples=20)
        assert noised.shape == gradient.shape
        assert sigma > 0

    def test_privacy_budget(self):
        dp = DPInjector(epsilon=0.3)
        budget = dp.privacy_budget_spent(n_rounds=5)
        assert budget > dp.epsilon  # Budget grows with rounds


class TestFederatedSimulation:
    def test_simulation_runs(self):
        sim = FederatedSimulation(n_devices=3, n_rounds=2, n_features=20)
        results = sim.run()
        assert len(results) == 2
        assert all('accuracy' in r for r in results)
        assert all('f1' in r for r in results)

    def test_accuracy_improves(self):
        """Accuracy should generally improve over rounds."""
        np.random.seed(42)
        sim = FederatedSimulation(n_devices=5, n_rounds=5, n_features=20)
        results = sim.run()
        # Last round should be better than first (with some tolerance)
        assert results[-1]['accuracy'] >= results[0]['accuracy'] - 0.1
