"""Train XGBoost fraud classifier for phishing URL detection.

Uses URL-specific structural features rather than TF-IDF, since URLs have
a different structure from natural language text.  Extracts 14 features
from each URL and trains an XGBoost binary classifier.

Usage
-----
    python training/train_url_classifier.py \
        [--train-csv data/processed/url_fraud_train.csv] \
        [--test-csv  data/processed/url_fraud_test.csv]  \
        [--output models/url_fraud_xgb.json]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ======================================================================
# URL feature extraction (14 features)
# ======================================================================

# Known URL shorteners
_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "t.ly", "v.gd", "qr.ae",
}

# Suspicious TLDs commonly used in phishing
_SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".pw",
    ".cc", ".club", ".work", ".date", ".racing", ".stream",
    ".download", ".win", ".bid", ".trade", ".webcam", ".review",
    ".party", ".cricket", ".science", ".click", ".link", ".info",
    ".zip", ".mov",
}

# IP address pattern
_IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)


def extract_url_features(url: str) -> dict[str, float]:
    """Extract 14 structural features from a URL.

    Features
    --------
    1. url_length           - Total character length of the URL
    2. hostname_length      - Character length of the hostname
    3. path_length          - Character length of the path
    4. num_dots             - Number of dots in the URL
    5. num_hyphens          - Number of hyphens in the URL
    6. num_underscores      - Number of underscores in the URL
    7. num_slashes          - Number of forward slashes in the URL
    8. num_query_params     - Number of query parameters
    9. has_ip_address       - Whether the host is an IP address (0/1)
    10. has_at_symbol       - Whether the URL contains @ (0/1)
    11. is_https            - Whether the URL uses HTTPS (0/1)
    12. is_shortened        - Whether the domain is a known shortener (0/1)
    13. has_suspicious_tld  - Whether the TLD is suspicious (0/1)
    14. entropy             - Shannon entropy of the URL string
    """
    # Ensure url is a string
    url = str(url).strip()
    url_lower = url.lower()

    # Parse the URL
    if not url_lower.startswith(("http://", "https://", "ftp://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    try:
        parsed = urlparse(url_for_parse)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        scheme = parsed.scheme or ""
    except Exception:
        hostname = ""
        path = ""
        query = ""
        scheme = ""

    # Feature 1: URL length
    url_length = float(len(url))

    # Feature 2: Hostname length
    hostname_length = float(len(hostname))

    # Feature 3: Path length
    path_length = float(len(path))

    # Feature 4: Number of dots
    num_dots = float(url.count("."))

    # Feature 5: Number of hyphens
    num_hyphens = float(url.count("-"))

    # Feature 6: Number of underscores
    num_underscores = float(url.count("_"))

    # Feature 7: Number of slashes (excluding protocol)
    stripped = url_for_parse.split("://", 1)[-1] if "://" in url_for_parse else url
    num_slashes = float(stripped.count("/"))

    # Feature 8: Number of query parameters
    if query:
        num_query_params = float(query.count("&") + 1)
    else:
        num_query_params = 0.0

    # Feature 9: Has IP address as host
    has_ip_address = 1.0 if _IP_PATTERN.match(hostname) else 0.0

    # Feature 10: Has @ symbol (used in URL obfuscation)
    has_at_symbol = 1.0 if "@" in url else 0.0

    # Feature 11: Is HTTPS
    is_https = 1.0 if scheme == "https" else 0.0

    # Feature 12: Is shortened URL
    is_shortened = 1.0 if hostname.lower() in _SHORTENERS else 0.0

    # Feature 13: Has suspicious TLD
    has_suspicious_tld = 0.0
    for tld in _SUSPICIOUS_TLDS:
        if hostname.lower().endswith(tld):
            has_suspicious_tld = 1.0
            break

    # Feature 14: Shannon entropy of URL string
    if len(url) > 0:
        freq = Counter(url)
        probs = [count / len(url) for count in freq.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_query_params": num_query_params,
        "has_ip_address": has_ip_address,
        "has_at_symbol": has_at_symbol,
        "is_https": is_https,
        "is_shortened": is_shortened,
        "has_suspicious_tld": has_suspicious_tld,
        "entropy": entropy,
    }


URL_FEATURE_NAMES = list(extract_url_features("http://example.com").keys())
N_URL_FEATURES = len(URL_FEATURE_NAMES)  # 14


def extract_features_batch(urls: list[str]) -> np.ndarray:
    """Extract URL features for a batch of URLs.

    Returns
    -------
    np.ndarray
        Shape ``(len(urls), 14)`` float32 feature matrix.
    """
    rows = []
    for url in urls:
        feats = extract_url_features(url)
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float32)


# ======================================================================
# Training
# ======================================================================

def train_classifier(
    train_csv: str,
    output_path: str,
    test_csv: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> None:
    """Train an XGBoost URL fraud classifier.

    Parameters
    ----------
    train_csv : str
        Path to training CSV (columns: text (url), label).
    output_path : str
        Where to save the trained XGBoost model (.json).
    test_csv : str | None
        Optional path to test CSV for evaluation.
    """
    # ---- Load data ----
    print(f"Loading training data from {train_csv} ...")
    train_df = pd.read_csv(train_csv, dtype={"text": str, "label": int})
    train_urls = train_df["text"].fillna("").tolist()
    train_labels = train_df["label"].values

    print(f"  {len(train_urls):,} training samples")
    print(f"  Label distribution: {dict(pd.Series(train_labels).value_counts().sort_index())}")

    # ---- Extract features ----
    print("Extracting URL features ...")
    t0 = time.time()
    X_train = extract_features_batch(train_urls)
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Feature extraction took {time.time() - t0:.1f}s")

    # ---- Compute scale_pos_weight ----
    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    # ---- Train XGBoost ----
    print(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}) ...")
    t0 = time.time()

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, train_labels, verbose=True)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # ---- Save model ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save_model(output_path)
    print(f"  Model saved to {output_path}")

    # ---- Evaluate on training data ----
    print("\n--- Training Set Evaluation ---")
    train_preds = model.predict(X_train)
    _print_metrics(train_labels, train_preds)

    # ---- Evaluate on test data ----
    if test_csv and os.path.exists(test_csv):
        print(f"\nLoading test data from {test_csv} ...")
        test_df = pd.read_csv(test_csv, dtype={"text": str, "label": int})
        test_urls = test_df["text"].fillna("").tolist()
        test_labels = test_df["label"].values

        print(f"  {len(test_urls):,} test samples")

        print("Extracting test features ...")
        X_test = extract_features_batch(test_urls)

        print("\n--- Test Set Evaluation ---")
        test_preds = model.predict(X_test)
        _print_metrics(test_labels, test_preds)

        # Per-category breakdown
        if "category" in test_df.columns:
            _print_category_breakdown(test_df, test_preds)

    # ---- Feature importance ----
    _print_feature_importance(model)


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification metrics."""
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print()
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print()
    print("  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["legit", "phishing"]))


def _print_category_breakdown(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """Print per-category accuracy breakdown."""
    test_df = test_df.copy()
    test_df["pred"] = predictions

    print("  Per-Category Breakdown:")
    print(f"  {'Category':<30} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")

    for cat in sorted(test_df["category"].unique()):
        mask = test_df["category"] == cat
        cat_true = test_df.loc[mask, "label"].values
        cat_pred = test_df.loc[mask, "pred"].values
        acc = accuracy_score(cat_true, cat_pred)
        f1 = f1_score(cat_true, cat_pred, zero_division=0)
        print(f"  {cat:<30} {mask.sum():>8} {acc:>10.4f} {f1:>8.4f}")
    print()


def _print_feature_importance(model: xgb.XGBClassifier) -> None:
    """Print feature importances for URL features."""
    importances = model.feature_importances_
    if len(importances) != len(URL_FEATURE_NAMES):
        print(f"  (Feature importance: dimension mismatch, skipping)")
        return

    indices = np.argsort(importances)[::-1]

    print("  Feature Importances (all 14 URL features):")
    print(f"  {'Rank':>4}  {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*4}  {'-'*25} {'-'*12}")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>4}  {URL_FEATURE_NAMES[idx]:<25} {importances[idx]:>12.6f}")
    print()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost URL fraud classifier."
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/url_fraud_train.csv",
        help="Training CSV path",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed/url_fraud_test.csv",
        help="Test CSV path (optional, for evaluation)",
    )
    parser.add_argument(
        "--output",
        default="models/url_fraud_xgb.json",
        help="Output model path (.json)",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train_classifier(
        train_csv=args.train_csv,
        output_path=args.output,
        test_csv=args.test_csv,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
