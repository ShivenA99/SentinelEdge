"""Shared test fixtures for SentinelEdge test suite."""
import pytest
import numpy as np


@pytest.fixture
def sample_scam_text():
    return "This is the IRS. You owe $5,000 in back taxes. Pay immediately or you will be arrested."


@pytest.fixture
def sample_legit_text():
    return "Hi, this is Sarah from Doctor Smith's office calling to confirm your appointment tomorrow at 3pm."


@pytest.fixture
def sample_features():
    return np.random.randn(518).astype(np.float32)


@pytest.fixture
def sample_gradient():
    return np.random.randn(518).astype(np.float32) * 0.01
