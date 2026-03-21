"""Tests for hub server API."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.schemas import FederatedUpdate, RoundStatus
from hub.aggregator import FedAvgAggregator
import numpy as np


class TestFedAvgAggregator:
    def test_weighted_average(self):
        agg = FedAvgAggregator()
        updates = [
            FederatedUpdate(
                device_id="d1",
                model_version=1,
                gradient_delta=[1.0, 2.0, 3.0],
                feature_importances=[0.5, 0.3, 0.2],
                n_samples=100,
                dp_epsilon=0.3,
                dp_sigma=0.1,
            ),
            FederatedUpdate(
                device_id="d2",
                model_version=1,
                gradient_delta=[3.0, 4.0, 5.0],
                feature_importances=[0.4, 0.4, 0.2],
                n_samples=100,
                dp_epsilon=0.3,
                dp_sigma=0.1,
            ),
        ]
        result = agg.aggregate(updates)
        # Equal weights, so should be simple average
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_weighted_by_samples(self):
        agg = FedAvgAggregator()
        updates = [
            FederatedUpdate(
                device_id="d1",
                model_version=1,
                gradient_delta=[0.0, 0.0],
                feature_importances=[0.5, 0.5],
                n_samples=10,
                dp_epsilon=0.3,
                dp_sigma=0.1,
            ),
            FederatedUpdate(
                device_id="d2",
                model_version=1,
                gradient_delta=[10.0, 10.0],
                feature_importances=[0.5, 0.5],
                n_samples=90,
                dp_epsilon=0.3,
                dp_sigma=0.1,
            ),
        ]
        result = agg.aggregate(updates)
        # Device 2 has 9x more samples, so result should be close to [10, 10]
        assert result[0] > 8.0

    def test_outlier_detection(self):
        agg = FedAvgAggregator()
        updates = [
            FederatedUpdate(device_id=f"d{i}", model_version=1,
                          gradient_delta=[1.0], feature_importances=[0.5],
                          n_samples=50, dp_epsilon=0.3, dp_sigma=0.1)
            for i in range(9)
        ]
        # Add one outlier
        updates.append(FederatedUpdate(
            device_id="outlier", model_version=1,
            gradient_delta=[1000.0], feature_importances=[0.5],
            n_samples=50, dp_epsilon=0.3, dp_sigma=0.1
        ))
        outliers = agg.detect_outliers(updates, threshold=2.0)
        assert len(outliers) >= 1
