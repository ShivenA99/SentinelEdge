"""Train XGBoost fraud classifier for SMS messages.

Same feature pipeline as the call classifier: handcrafted (18) + TF-IDF (500)
= 518-dimensional feature vectors, then XGBoost binary classification.

Usage
-----
    python training/train_sms_classifier.py \
        [--train-csv data/processed/sms_fraud_train.csv] \
        [--test-csv  data/processed/sms_fraud_test.csv]  \
        [--tfidf-path models/tfidf_sms_vectorizer.pkl]   \
        [--output models/sms_fraud_xgb.json]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.tfidf import TfidfFeatureExtractor


# ======================================================================
# Feature extraction (same pipeline as call classifier)
# ======================================================================

def extract_features(
    texts: list[str],
    tfidf: TfidfFeatureExtractor,
) -> np.ndarray:
    """Extract combined feature matrix: handcrafted (18) + TF-IDF (500) = 518.

    Parameters
    ----------
    texts : list[str]
        Raw SMS text messages.
    tfidf : TfidfFeatureExtractor
        Fitted TF-IDF extractor.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 518)`` float32 feature matrix.
    """
    # Handcrafted features
    hc_list = []
    for t in texts:
        feats = extract_handcrafted_features(t)
        hc_list.append(list(feats.values()))
    hc_matrix = np.array(hc_list, dtype=np.float32)

    # TF-IDF features
    tfidf_matrix = np.zeros((len(texts), tfidf.n_features), dtype=np.float32)
    for i, t in enumerate(texts):
        tfidf_matrix[i] = tfidf.transform(t).astype(np.float32)

    # Concatenate
    combined = np.hstack([hc_matrix, tfidf_matrix])
    return combined


# ======================================================================
# Training
# ======================================================================

def train_classifier(
    train_csv: str,
    tfidf_path: str,
    output_path: str,
    test_csv: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> None:
    """Train an XGBoost SMS fraud classifier.

    Parameters
    ----------
    train_csv : str
        Path to training CSV (columns: text, label).
    tfidf_path : str
        Path to fitted TF-IDF vectorizer for SMS.
    output_path : str
        Where to save the trained XGBoost model (.json).
    test_csv : str | None
        Optional path to test CSV for evaluation.
    """
    # ---- Load data ----
    print(f"Loading training data from {train_csv} ...")
    train_df = pd.read_csv(train_csv, dtype={"text": str, "label": int})
    train_texts = train_df["text"].fillna("").tolist()
    train_labels = train_df["label"].values

    print(f"  {len(train_texts):,} training samples")
    print(f"  Label distribution: {dict(pd.Series(train_labels).value_counts().sort_index())}")

    # ---- Fit TF-IDF if needed, otherwise load ----
    if os.path.exists(tfidf_path):
        print(f"Loading TF-IDF vectorizer from {tfidf_path} ...")
        tfidf = TfidfFeatureExtractor.load(tfidf_path)
    else:
        print(f"TF-IDF model not found at {tfidf_path}, fitting new vectorizer ...")
        tfidf = TfidfFeatureExtractor()
        tfidf.fit(train_texts)
        os.makedirs(os.path.dirname(tfidf_path) or ".", exist_ok=True)
        tfidf.save(tfidf_path)
        print(f"  Saved new TF-IDF vectorizer to {tfidf_path}")
    print(f"  Vocabulary size: {tfidf.n_features}")

    # ---- Extract features ----
    print("Extracting features ...")
    t0 = time.time()
    X_train = extract_features(train_texts, tfidf)
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
        test_texts = test_df["text"].fillna("").tolist()
        test_labels = test_df["label"].values

        print(f"  {len(test_texts):,} test samples")

        print("Extracting test features ...")
        X_test = extract_features(test_texts, tfidf)

        print("\n--- Test Set Evaluation ---")
        test_preds = model.predict(X_test)
        _print_metrics(test_labels, test_preds)

        # Per-category breakdown
        if "category" in test_df.columns:
            _print_category_breakdown(test_df, test_preds)

    # ---- Feature importance ----
    _print_feature_importance(model, tfidf)


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
    print(classification_report(y_true, y_pred, target_names=["legit", "spam"]))


def _print_category_breakdown(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """Print per-category accuracy breakdown."""
    test_df = test_df.copy()
    test_df["pred"] = predictions
    test_df["correct"] = (test_df["label"] == test_df["pred"]).astype(int)

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


def _print_feature_importance(
    model: xgb.XGBClassifier,
    tfidf: TfidfFeatureExtractor,
) -> None:
    """Print top-20 most important features."""
    hc_names = list(extract_handcrafted_features("dummy text").keys())
    try:
        tfidf_names = [f"tfidf_{w}" for w in tfidf.vectorizer.get_feature_names_out()]
    except Exception:
        tfidf_names = [f"tfidf_{i}" for i in range(tfidf.n_features)]
    all_names = hc_names + tfidf_names

    importances = model.feature_importances_
    if len(importances) != len(all_names):
        print(f"  (Feature importance: dimension mismatch "
              f"{len(importances)} vs {len(all_names)}, skipping)")
        return

    indices = np.argsort(importances)[::-1]

    print("  Top-20 Feature Importances:")
    print(f"  {'Rank':>4}  {'Feature':<40} {'Importance':>12}")
    print(f"  {'-'*4}  {'-'*40} {'-'*12}")
    for rank, idx in enumerate(indices[:20], 1):
        print(f"  {rank:>4}  {all_names[idx]:<40} {importances[idx]:>12.6f}")
    print()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost SMS fraud classifier."
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/sms_fraud_train.csv",
        help="Training CSV path",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed/sms_fraud_test.csv",
        help="Test CSV path (optional, for evaluation)",
    )
    parser.add_argument(
        "--tfidf-path",
        default="models/tfidf_sms_vectorizer.pkl",
        help="Path to fitted TF-IDF vectorizer",
    )
    parser.add_argument(
        "--output",
        default="models/sms_fraud_xgb.json",
        help="Output model path (.json)",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train_classifier(
        train_csv=args.train_csv,
        tfidf_path=args.tfidf_path,
        output_path=args.output,
        test_csv=args.test_csv,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
