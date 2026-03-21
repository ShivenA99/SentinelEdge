"""Tests for fraud classifier components."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel_edge.classifier.score_accumulator import ScoreAccumulator
from sentinel_edge.classifier.alert_engine import AlertEngine, RiskLevel


class TestScoreAccumulator:
    def test_initial_score(self):
        acc = ScoreAccumulator(alpha=0.3)
        assert acc.current_score == 0.0

    def test_single_update(self):
        acc = ScoreAccumulator(alpha=0.3)
        # First update bootstraps EMA to the new_score value
        result = acc.update(1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_second_update_applies_ema(self):
        """The second update should apply the EMA formula."""
        acc = ScoreAccumulator(alpha=0.3)
        acc.update(0.0)  # Bootstrap: EMA = 0.0
        result = acc.update(1.0)
        # EMA = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        assert result == pytest.approx(0.3, abs=0.01)

    def test_ema_convergence(self):
        """High scores should cause EMA to converge upward."""
        acc = ScoreAccumulator(alpha=0.3)
        for _ in range(20):
            acc.update(0.9)
        assert acc.current_score > 0.85

    def test_ema_smoothing(self):
        """Single spike should not cause huge EMA jump."""
        acc = ScoreAccumulator(alpha=0.3)
        acc.update(0.0)
        acc.update(0.0)
        result = acc.update(1.0)
        assert result < 0.5  # EMA smooths the spike

    def test_reset(self):
        acc = ScoreAccumulator(alpha=0.3)
        acc.update(0.9)
        acc.reset()
        assert acc.current_score == 0.0
        assert acc.count == 0

    def test_history(self):
        acc = ScoreAccumulator(alpha=0.3)
        acc.update(0.5)
        acc.update(0.7)
        acc.update(0.3)
        assert len(acc.history) == 3


class TestAlertEngine:
    def test_safe_score(self):
        engine = AlertEngine()
        alert = engine.evaluate(0.1, {'urgency_count': 0, 'has_threat': 0, 'impersonation_count': 0})
        assert not alert.should_alert
        assert alert.risk_level in (RiskLevel.SAFE, RiskLevel.LOW)

    def test_critical_score(self):
        engine = AlertEngine()
        features = {
            'urgency_count': 4,
            'impersonation_count': 2,
            'has_threat': 1,
            'financial_count': 3,
            'action_count': 2,
            'has_account_ref': 1,
        }
        alert = engine.evaluate(0.85, features)
        assert alert.should_alert
        assert alert.risk_level == RiskLevel.CRITICAL
        assert len(alert.reasons) > 0

    def test_amber_threshold(self):
        engine = AlertEngine(amber_threshold=0.5)
        alert = engine.evaluate(0.6, {'urgency_count': 2, 'has_threat': 0, 'impersonation_count': 0})
        assert alert.should_alert
        assert alert.risk_level == RiskLevel.HIGH

    def test_reasons_generation(self):
        engine = AlertEngine()
        features = {
            'impersonation_count': 3,
            'has_threat': 1,
            'urgency_count': 5,
        }
        alert = engine.evaluate(0.8, features)
        assert any('impersonation' in r.lower() for r in alert.reasons)

    def test_custom_thresholds(self):
        engine = AlertEngine(red_threshold=0.9, amber_threshold=0.7)
        alert = engine.evaluate(0.8, {})
        assert alert.risk_level == RiskLevel.HIGH  # Between amber and red
