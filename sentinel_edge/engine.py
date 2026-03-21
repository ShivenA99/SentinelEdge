"""Top-level SentinelEdge inference engine.

Ties together the feature pipeline, classifier, score accumulator,
and alert engine into a single, easy-to-use API for analysing SMS
messages, URLs, and phone call transcripts in real time.

Supports three classifier back-ends:
  - ``"mlp"``     -- MiniLM embeddings + MLP (federable)
  - ``"xgboost"`` -- TF-IDF + XGBoost (legacy)
  - ``"auto"``    -- tries MLP first, then XGBoost, then heuristic
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel_edge.classifier.alert_engine import AlertEngine, RiskLevel
from sentinel_edge.classifier.score_accumulator import ScoreAccumulator
from sentinel_edge.classifier.xgb_classifier import FraudClassifier
from sentinel_edge.features.feature_pipeline import FeaturePipeline
from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.url_features import extract_url_features
from sentinel_edge.audio.sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)

# Simple heuristic patterns for auto-detection
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)


@dataclass
class DetectionResult:
    """Structured output from every ``analyze_*`` method.

    Attributes
    ----------
    channel : str
        Detection channel: ``'sms'``, ``'url'``, or ``'call'``.
    is_fraud : bool
        Binary fraud verdict.
    confidence : float
        Fraud probability in ``[0.0, 1.0]``.
    risk_level : str
        One of ``'critical'``, ``'high'``, ``'medium'``, ``'low'``,
        ``'safe'``.
    reasons : list[str]
        Human-readable explanations for the verdict.
    inference_ms : float
        Wall-clock latency of the prediction in milliseconds.
    """

    channel: str
    is_fraud: bool
    confidence: float
    risk_level: str
    reasons: list[str] = field(default_factory=list)
    inference_ms: float = 0.0


class SentinelEngine:
    """High-level facade for SentinelEdge fraud detection.

    Parameters
    ----------
    models_dir : str
        Directory containing trained model artefacts:

        - ``call_fraud_mlp.npz``           -- MLP classifier
        - ``xgb_model.json`` / ``.onnx``   -- XGBoost classifier
        - ``tfidf.joblib``                  -- fitted TF-IDF vectorizer

        If the directory (or individual files) does not exist, the engine
        falls back to handcrafted-features-only mode so that development
        and testing can proceed without trained models.
    threshold : float
        Decision threshold for the binary fraud prediction.
    alpha : float
        EMA smoothing factor for phone-call score accumulation.
    pipeline : str
        Classifier pipeline to use:

        - ``"auto"``    -- MLP if available, then XGBoost, then heuristic.
        - ``"mlp"``     -- MiniLM embeddings + MLP.
        - ``"xgboost"`` -- TF-IDF + XGBoost (legacy behaviour).
    """

    def __init__(
        self,
        models_dir: str = "models",
        threshold: float = 0.5,
        alpha: float = 0.3,
        pipeline: str = "auto",
    ) -> None:
        self._models_dir = Path(models_dir)
        self.threshold = threshold
        self._pipeline_mode = pipeline

        # Resolved at init time
        self._active_backend: str = "heuristic"  # "mlp" | "xgboost" | "heuristic"
        self.classifier = None       # FraudClassifier or MLPClassifier
        self.pipeline: FeaturePipeline = None  # type: ignore[assignment]

        # Channel-specific classifiers (loaded separately)
        self._sms_classifier = None   # XGBClassifier for SMS
        self._sms_pipeline: FeaturePipeline | None = None
        self._url_classifier = None   # XGBClassifier for URLs

        self._init_pipeline(pipeline)
        self._init_sms_classifier()
        self._init_url_classifier()

        # --- Score accumulator (per-call state) ---
        self.accumulator = ScoreAccumulator(alpha=alpha)

        # --- Alert engine ---
        self.alert_engine = AlertEngine()

        # --- Sentence splitter (reusable across calls) ---
        self._splitter = SentenceSplitter()

    # ------------------------------------------------------------------
    # Pipeline initialisation
    # ------------------------------------------------------------------

    def _init_pipeline(self, pipeline: str) -> None:
        """Resolve which classifier back-end to use and set up the
        corresponding feature pipeline."""
        if pipeline == "mlp" or pipeline == "auto":
            if self._try_load_mlp():
                return

        if pipeline == "xgboost" or pipeline == "auto":
            if self._try_load_xgboost():
                return

        if pipeline not in ("auto", "mlp", "xgboost"):
            raise ValueError(
                f"Unknown pipeline '{pipeline}'. "
                "Expected 'auto', 'mlp', or 'xgboost'."
            )

        # Heuristic fallback
        logger.warning(
            "No trained model found in %s -- running in "
            "handcrafted-features-only mode (scores will be "
            "heuristic-based).",
            self._models_dir,
        )
        tfidf_path = self._resolve_model("tfidf.joblib")
        self.pipeline = FeaturePipeline(tfidf_path, mode="tfidf")
        self._active_backend = "heuristic"

    def _try_load_mlp(self) -> bool:
        """Attempt to load the MLP classifier and embedding pipeline."""
        mlp_path = self._resolve_model("call_fraud_mlp.npz")
        if mlp_path is None:
            return False

        try:
            from sentinel_edge.classifier.mlp_classifier import MLPClassifier

            self.classifier = MLPClassifier.load(mlp_path)
            self.pipeline = FeaturePipeline(mode="embedding")
            self._active_backend = "mlp"
            logger.info("Using MLP pipeline (embedding + MLP classifier).")
            return True
        except Exception as exc:
            logger.warning(
                "Could not load MLP classifier from %s: %s", mlp_path, exc
            )
            return False

    def _try_load_xgboost(self) -> bool:
        """Attempt to load the XGBoost classifier and TF-IDF pipeline."""
        tfidf_path = self._resolve_model("tfidf.joblib")
        for ext in ("json", "onnx"):
            model_path = self._resolve_model(f"xgb_model.{ext}")
            if model_path is not None:
                try:
                    self.classifier = FraudClassifier(model_path)
                    self.pipeline = FeaturePipeline(tfidf_path, mode="tfidf")
                    self._active_backend = "xgboost"
                    logger.info(
                        "Using XGBoost pipeline (TF-IDF + XGBoost classifier)."
                    )
                    return True
                except Exception as exc:
                    logger.warning(
                        "Could not load XGBoost model from %s: %s",
                        model_path,
                        exc,
                    )
        return False

    # ------------------------------------------------------------------
    # Channel-specific model loading (SMS, URL)
    # ------------------------------------------------------------------

    def _init_sms_classifier(self) -> None:
        """Load the SMS-specific XGBoost model and TF-IDF vectorizer."""
        model_path = self._resolve_model("sms_fraud_xgb.json")
        tfidf_path = self._resolve_model("tfidf_sms_vectorizer.pkl")

        if model_path is None:
            logger.debug("SMS model sms_fraud_xgb.json not found in %s", self._models_dir)
            return

        try:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(model_path)
            self._sms_classifier = model

            # Load the SMS-specific TF-IDF vectorizer into a FeaturePipeline
            from sentinel_edge.features.tfidf import TfidfFeatureExtractor

            if tfidf_path is not None:
                sms_tfidf = TfidfFeatureExtractor(tfidf_path)
            else:
                sms_tfidf = TfidfFeatureExtractor()

            self._sms_pipeline = FeaturePipeline(tfidf_path, mode="tfidf")
            logger.info(
                "Loaded SMS fraud classifier from %s (tfidf: %s)",
                model_path, tfidf_path or "default",
            )
        except Exception as exc:
            logger.warning("Could not load SMS classifier: %s", exc)
            self._sms_classifier = None
            self._sms_pipeline = None

    def _init_url_classifier(self) -> None:
        """Load the URL-specific XGBoost model."""
        model_path = self._resolve_model("url_fraud_xgb.json")

        if model_path is None:
            logger.debug("URL model url_fraud_xgb.json not found in %s", self._models_dir)
            return

        try:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(model_path)
            self._url_classifier = model
            logger.info("Loaded URL fraud classifier from %s", model_path)
        except Exception as exc:
            logger.warning("Could not load URL classifier: %s", exc)
            self._url_classifier = None

    # ------------------------------------------------------------------
    # Channel-specific analysis
    # ------------------------------------------------------------------

    def analyze_sms(self, text: str) -> DetectionResult:
        """Analyse an SMS / text message for fraud indicators.

        Uses the dedicated SMS XGBoost classifier if available,
        otherwise falls back to the general pipeline.

        Parameters
        ----------
        text : str
            The full SMS body.

        Returns
        -------
        DetectionResult
        """
        t0 = time.perf_counter()

        features = extract_handcrafted_features(text)

        if self._sms_classifier is not None and self._sms_pipeline is not None:
            # Use the dedicated SMS classifier
            feature_vec = self._sms_pipeline.extract(text)
            feature_vec = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
            proba = self._sms_classifier.predict_proba(feature_vec)
            # predict_proba returns shape (n_samples, n_classes) for XGBClassifier
            score = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])
        else:
            # Fallback to general pipeline
            score, features = self.analyze_sentence(text)

        alert = self.alert_engine.evaluate(score, features)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            channel="sms",
            is_fraud=score >= self.threshold,
            confidence=score,
            risk_level=alert.risk_level.value,
            reasons=alert.reasons,
            inference_ms=elapsed_ms,
        )

    def analyze_url(self, url: str) -> DetectionResult:
        """Analyse a URL for phishing indicators.

        Uses the dedicated URL XGBoost classifier if available,
        otherwise falls back to heuristic scoring from URL features.

        Parameters
        ----------
        url : str
            The URL to evaluate.

        Returns
        -------
        DetectionResult
        """
        t0 = time.perf_counter()

        url_feats = extract_url_features(url)

        if self._url_classifier is not None:
            # Use the dedicated URL classifier with the training-time
            # feature extraction (from train_url_classifier.py).
            from training.train_url_classifier import (
                extract_url_features as extract_train_url_features,
            )

            train_feats = extract_train_url_features(url)
            feature_vec = np.array(
                list(train_feats.values()), dtype=np.float32
            ).reshape(1, -1)
            proba = self._url_classifier.predict_proba(feature_vec)
            score = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])
        else:
            # Heuristic score from URL features (weighted sum)
            score = self._heuristic_url_score(url_feats)

        reasons: list[str] = []
        if url_feats["has_ip"]:
            reasons.append("URL uses a raw IP address instead of a domain")
        if url_feats["suspicious_tld"]:
            reasons.append("URL uses a suspicious top-level domain")
        if not url_feats["has_https"]:
            reasons.append("URL does not use HTTPS")
        if url_feats["domain_entropy"] > 4.0:
            reasons.append("Domain name has high entropy (possibly generated)")
        if url_feats["at_symbol"]:
            reasons.append("URL contains an @ symbol (possible credential trick)")
        if url_feats["subdomain_count"] > 3:
            reasons.append("Excessive subdomains detected")
        if url_feats["url_length"] > 75:
            reasons.append("Unusually long URL")
        if url_feats["has_port"]:
            reasons.append("Non-standard port in URL")

        risk = self.alert_engine._classify_risk(score)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            channel="url",
            is_fraud=score >= self.threshold,
            confidence=score,
            risk_level=risk.value,
            reasons=reasons,
            inference_ms=elapsed_ms,
        )

    def analyze_call(self, transcript: str) -> DetectionResult:
        """Analyse a full or partial phone call transcript.

        The transcript is split into sentences, each scored individually,
        and the EMA accumulator aggregates the per-sentence scores.

        Parameters
        ----------
        transcript : str
            Full or incremental transcript text.

        Returns
        -------
        DetectionResult
        """
        t0 = time.perf_counter()

        self._splitter.reset()
        sentences = self._splitter.feed(transcript)
        leftover = self._splitter.flush()
        if leftover:
            sentences.append(leftover)

        all_features: dict[str, float] = {}
        for sentence in sentences:
            score, features = self.analyze_sentence(sentence)
            self.accumulator.update(score)
            # Merge features (keep the max across sentences for each key)
            for k, v in features.items():
                all_features[k] = max(all_features.get(k, 0.0), v)

        ema_score = self.accumulator.current_score
        alert = self.alert_engine.evaluate(ema_score, all_features)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            channel="call",
            is_fraud=ema_score >= self.threshold,
            confidence=ema_score,
            risk_level=alert.risk_level.value,
            reasons=alert.reasons,
            inference_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Sentence-level analysis
    # ------------------------------------------------------------------

    def analyze_sentence(
        self, sentence: str
    ) -> tuple[float, dict[str, float]]:
        """Score a single sentence and return raw features.

        Parameters
        ----------
        sentence : str
            One sentence of text.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(fraud_probability, handcrafted_features)``.
        """
        features = self.pipeline.extract_handcrafted_only(sentence)

        if self.classifier is not None:
            feature_vec = self.pipeline.extract(sentence)
            score = self.classifier.predict_proba(feature_vec)
        else:
            # Fallback heuristic when no model is available
            score = self._heuristic_text_score(features)

        return score, features

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def analyze_auto(self, text: str) -> DetectionResult:
        """Auto-detect the channel and analyse accordingly.

        Heuristic:
        - If the text looks like a URL (starts with ``http`` / ``www``
          and is a single token), use ``analyze_url``.
        - Otherwise, use ``analyze_sms`` (which also works well for
          single transcript sentences).

        Parameters
        ----------
        text : str
            Input text to analyse.

        Returns
        -------
        DetectionResult
        """
        stripped = text.strip()

        # Single-token URL check
        if (
            " " not in stripped
            and _URL_RE.match(stripped)
        ):
            return self.analyze_url(stripped)

        return self.analyze_sms(stripped)

    # ------------------------------------------------------------------
    # Call-state management
    # ------------------------------------------------------------------

    def reset_call_state(self) -> None:
        """Reset the EMA accumulator for a new phone call."""
        self.accumulator.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        """Return the name of the active classifier back-end.

        One of ``"mlp"``, ``"xgboost"``, or ``"heuristic"``.
        """
        return self._active_backend

    # ------------------------------------------------------------------
    # Heuristic fallback scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_text_score(features: dict[str, float]) -> float:
        """Produce a rough fraud score from handcrafted features alone.

        This is used only when no trained classifier is loaded.  It is
        intentionally conservative (high false-negative rate) to avoid
        nuisance alerts.
        """
        score = 0.0
        score += min(features.get("urgency_count", 0) * 0.08, 0.24)
        score += min(features.get("action_count", 0) * 0.06, 0.18)
        score += min(features.get("financial_count", 0) * 0.06, 0.18)
        score += min(features.get("impersonation_count", 0) * 0.10, 0.20)
        score += features.get("has_threat", 0) * 0.10
        score += features.get("has_prize", 0) * 0.08
        score += features.get("has_url", 0) * 0.04
        score += features.get("has_shortened_url", 0) * 0.06
        score += features.get("has_account_ref", 0) * 0.05
        score += features.get("has_verify_pattern", 0) * 0.04
        score += features.get("dollar_sign", 0) * 0.02
        score += features.get("has_phone_number", 0) * 0.02
        # Clamp to [0, 1]
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _heuristic_url_score(url_feats: dict[str, float]) -> float:
        """Produce a rough phishing score from URL features alone."""
        score = 0.0
        score += url_feats.get("has_ip", 0) * 0.20
        score += url_feats.get("suspicious_tld", 0) * 0.15
        score += (1.0 - url_feats.get("has_https", 0)) * 0.10
        score += url_feats.get("at_symbol", 0) * 0.15

        entropy = url_feats.get("domain_entropy", 0)
        if entropy > 4.0:
            score += min((entropy - 4.0) * 0.05, 0.15)

        subdomain_count = url_feats.get("subdomain_count", 0)
        if subdomain_count > 3:
            score += min((subdomain_count - 3) * 0.05, 0.10)

        url_length = url_feats.get("url_length", 0)
        if url_length > 75:
            score += min((url_length - 75) * 0.002, 0.10)

        score += url_feats.get("has_port", 0) * 0.05

        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, filename: str) -> str | None:
        """Return the full path to a model file if it exists, else None."""
        path = self._models_dir / filename
        if path.exists():
            return str(path)
        return None
