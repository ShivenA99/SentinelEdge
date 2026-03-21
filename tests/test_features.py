"""Tests for feature extraction."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.url_features import extract_url_features


class TestHandcraftedFeatures:
    """Test the 18 handcrafted features."""

    def test_urgency_count(self):
        text = "Act now! This is urgent, you must respond immediately!"
        features = extract_handcrafted_features(text)
        assert features['urgency_count'] >= 3  # now, urgent, immediately

    def test_action_count(self):
        text = "Click here to verify your account and confirm your identity"
        features = extract_handcrafted_features(text)
        assert features['action_count'] >= 2  # click, verify, confirm

    def test_financial_count(self):
        text = "Your bank account requires a wire transfer payment"
        features = extract_handcrafted_features(text)
        assert features['financial_count'] >= 3  # bank, account, wire, transfer, payment

    def test_impersonation_count(self):
        text = "This is the IRS and FBI calling about your federal case"
        features = extract_handcrafted_features(text)
        assert features['impersonation_count'] >= 3  # IRS, FBI, federal

    def test_has_url(self):
        text = "Visit https://example.com to verify"
        features = extract_handcrafted_features(text)
        assert features['has_url'] == 1

    def test_no_url(self):
        text = "Hello, how are you today?"
        features = extract_handcrafted_features(text)
        assert features['has_url'] == 0

    def test_has_shortened_url(self):
        text = "Click bit.ly/abc123 now"
        features = extract_handcrafted_features(text)
        assert features['has_shortened_url'] == 1

    def test_has_threat(self):
        text = "You will be arrested and your account will be suspended"
        features = extract_handcrafted_features(text)
        assert features['has_threat'] == 1

    def test_has_prize(self):
        text = "Congratulations! You are the lucky winner of our prize"
        features = extract_handcrafted_features(text)
        assert features['has_prize'] == 1

    def test_has_account_ref(self):
        text = "Please provide your password and social security number"
        features = extract_handcrafted_features(text)
        assert features['has_account_ref'] == 1

    def test_dollar_sign(self):
        text = "You owe $5,000 in back taxes"
        features = extract_handcrafted_features(text)
        assert features['dollar_sign'] == 1

    def test_has_phone_number(self):
        text = "Call us back at 800-555-1234"
        features = extract_handcrafted_features(text)
        assert features['has_phone_number'] == 1

    def test_caps_ratio(self):
        text = "THIS IS URGENT"
        features = extract_handcrafted_features(text)
        assert features['caps_ratio'] > 0.7

    def test_legitimate_text_low_scores(self):
        """A normal sentence should have low fraud indicators."""
        text = "Hi, this is Sarah calling to confirm your dental appointment tomorrow at 3pm."
        features = extract_handcrafted_features(text)
        assert features['urgency_count'] == 0
        assert features['has_threat'] == 0
        assert features['impersonation_count'] == 0
        assert features['has_url'] == 0

    def test_scam_text_high_scores(self):
        """A scam sentence should have high fraud indicators."""
        text = "This is the IRS. You must pay $3,000 immediately or face arrest. Call 800-555-0000 now!"
        features = extract_handcrafted_features(text)
        assert features['urgency_count'] >= 2
        assert features['impersonation_count'] >= 1
        assert features['has_threat'] == 1
        assert features['dollar_sign'] == 1

    def test_feature_count(self):
        """Ensure we get exactly 18 features."""
        features = extract_handcrafted_features("Test sentence")
        assert len(features) == 18

    def test_empty_text(self):
        """Empty text should return zero features."""
        features = extract_handcrafted_features("")
        assert features['urgency_count'] == 0
        assert features['char_count'] == 0
        assert features['word_count'] == 0


class TestUrlFeatures:
    """Test URL feature extraction."""

    def test_basic_url(self):
        features = extract_url_features("https://www.google.com")
        assert features['has_https'] == 1
        assert features['has_ip'] == 0

    def test_suspicious_url(self):
        features = extract_url_features("http://192.168.1.1:8080/login?user=admin&pass=123")
        assert features['has_ip'] == 1
        assert features['has_https'] == 0
        assert features['has_port'] == 1
        assert features['query_param_count'] >= 2

    def test_phishing_url(self):
        features = extract_url_features("http://secure-bankofamerica-login.suspicious-site.tk/verify")
        assert features['hyphen_count'] >= 3
        assert features['suspicious_tld'] == 1  # .tk is suspicious

    def test_feature_count(self):
        features = extract_url_features("https://example.com")
        assert len(features) >= 12
