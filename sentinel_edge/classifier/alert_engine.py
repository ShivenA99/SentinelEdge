"""Alert engine -- threshold logic and human-readable reason generation.

Translates a smoothed EMA fraud score and the underlying handcrafted
features into an ``AlertDecision`` that the UI layer can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """Discrete risk tiers displayed to the end-user."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertDecision:
    """Immutable decision produced by :class:`AlertEngine.evaluate`.

    Attributes
    ----------
    should_alert : bool
        Whether the UI should display an active warning overlay.
    risk_level : RiskLevel
        Categorical risk tier.
    score : float
        The smoothed EMA score that drove the decision.
    reasons : list[str]
        Human-readable explanations derived from the handcrafted features
        (e.g. "Contains IRS impersonation language").
    """

    should_alert: bool
    risk_level: RiskLevel
    score: float
    reasons: list[str] = field(default_factory=list)


class AlertEngine:
    """Stateless evaluator that maps scores + features to alert decisions.

    Parameters
    ----------
    red_threshold : float
        EMA score at or above which a CRITICAL alert is issued and the
        full red overlay is shown.
    amber_threshold : float
        EMA score at or above which a HIGH (amber) caution is issued.
    """

    def __init__(
        self,
        red_threshold: float = 0.75,
        amber_threshold: float = 0.5,
    ) -> None:
        if not (0.0 < amber_threshold < red_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < amber < red <= 1.0, "
                f"got amber={amber_threshold}, red={red_threshold}"
            )
        self.red_threshold = red_threshold
        self.amber_threshold = amber_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ema_score: float,
        features: dict[str, float],
    ) -> AlertDecision:
        """Produce an alert decision from the current EMA score and features.

        Parameters
        ----------
        ema_score : float
            Smoothed fraud score from the ``ScoreAccumulator``.
        features : dict[str, float]
            Handcrafted feature dict (18 keys) from ``extract_handcrafted_features``.

        Returns
        -------
        AlertDecision
        """
        risk_level = self._classify_risk(ema_score)
        reasons = self._generate_reasons(features)
        should_alert = risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)

        return AlertDecision(
            should_alert=should_alert,
            risk_level=risk_level,
            score=ema_score,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify_risk(self, ema_score: float) -> RiskLevel:
        """Map a continuous score to a discrete risk tier."""
        if ema_score >= self.red_threshold:
            return RiskLevel.CRITICAL
        if ema_score >= self.amber_threshold:
            return RiskLevel.HIGH
        if ema_score >= 0.3:
            return RiskLevel.MEDIUM
        if ema_score >= 0.15:
            return RiskLevel.LOW
        return RiskLevel.SAFE

    def _generate_reasons(self, features: dict[str, float]) -> list[str]:
        """Derive human-readable explanations from the handcrafted features.

        Parameters
        ----------
        features : dict[str, float]
            Output of ``extract_handcrafted_features``.

        Returns
        -------
        list[str]
            Possibly empty list of explanation strings.
        """
        reasons: list[str] = []

        # --- Impersonation ---
        imp = features.get("impersonation_count", 0.0)
        if imp >= 1:
            reasons.append(
                "Contains government/corporate impersonation language"
            )

        # --- Urgency ---
        urg = features.get("urgency_count", 0.0)
        if urg >= 2:
            reasons.append("Multiple urgency tactics detected")
        elif urg == 1:
            reasons.append("Urgency language detected")

        # --- Action pressure ---
        act = features.get("action_count", 0.0)
        if act >= 2:
            reasons.append("Pressure to take immediate action")
        elif act == 1:
            reasons.append("Prompts user to perform a specific action")

        # --- Financial references ---
        fin = features.get("financial_count", 0.0)
        if fin >= 1:
            reasons.append("References financial accounts or transactions")

        # --- Threats ---
        if features.get("has_threat", 0.0):
            reasons.append("Threatening language detected (arrest, suspension, etc.)")

        # --- Prize / lottery ---
        if features.get("has_prize", 0.0):
            reasons.append("Prize or lottery claim detected")

        # --- URLs ---
        if features.get("has_url", 0.0):
            reasons.append("Contains a URL -- possible phishing link")
        if features.get("has_shortened_url", 0.0):
            reasons.append("Contains a shortened URL hiding the true destination")

        # --- Account reference ---
        if features.get("has_account_ref", 0.0):
            reasons.append("Requests sensitive account or credential information")

        # --- Verify pattern ---
        if features.get("has_verify_pattern", 0.0):
            reasons.append("Asks to verify or confirm identity/account")

        # --- Dollar sign ---
        if features.get("dollar_sign", 0.0):
            reasons.append("References a monetary amount")

        # --- Phone number ---
        if features.get("has_phone_number", 0.0):
            reasons.append("Includes a phone number -- possible callback scam")

        # --- Excessive caps ---
        caps = features.get("caps_ratio", 0.0)
        if caps >= 0.5:
            reasons.append("Excessive use of capital letters")

        # --- Exclamations ---
        excl = features.get("exclamation_count", 0.0)
        if excl >= 3:
            reasons.append("Excessive exclamation marks suggest high-pressure tactics")

        return reasons
