"""Benchmark: TF-IDF vs. sentence embedding features for fraud detection.

Compares three configurations:
  1. TF-IDF (500) + handcrafted (18) = 518 dims  -> XGBoost
  2. Embedding (384) + handcrafted (18) = 402 dims -> XGBoost
  3. Embedding (384) + handcrafted (18) = 402 dims -> MLP

Reports F1, accuracy, precision, recall, and feature extraction time
for each configuration.

Usage
-----
    python training/benchmark_embeddings.py \
        [--train-csv data/processed/call_fraud_train.csv] \
        [--test-csv  data/processed/call_fraud_test.csv]  \
        [--tfidf-path models/tfidf_call_vectorizer.pkl]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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

from sentinel_edge.classifier.mlp_classifier import MLPClassifier
from sentinel_edge.features.feature_pipeline import FeaturePipeline
from sentinel_edge.features.tfidf import TfidfFeatureExtractor


# ======================================================================
# Feature extraction helpers
# ======================================================================

def extract_tfidf_features(
    texts: list[str],
    tfidf: TfidfFeatureExtractor,
) -> np.ndarray:
    """Extract TF-IDF + handcrafted features (518 dims)."""
    pipeline = FeaturePipeline(mode="tfidf")
    pipeline.tfidf = tfidf
    return pipeline.extract_batch(texts)


def extract_embedding_features(texts: list[str]) -> np.ndarray:
    """Extract embedding + handcrafted features (402 dims)."""
    pipeline = FeaturePipeline(mode="embedding")
    return pipeline.extract_batch(texts)


# ======================================================================
# Model training helpers
# ======================================================================

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Train XGBoost and return test metrics."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [SKIP] xgboost not installed, skipping XGBoost benchmark.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
) -> dict[str, float]:
    """Train MLP and return test metrics."""
    from training.train_mlp_classifier import train_numpy_mlp

    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim=input_dim)
    model = train_numpy_mlp(
        model, X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=epochs, batch_size=64, lr=0.001,
    )

    probs = model.forward_batch(X_test)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


# ======================================================================
# Main benchmark
# ======================================================================

def run_benchmark(
    train_csv: str,
    test_csv: str,
    tfidf_path: str,
    mlp_epochs: int = 50,
) -> None:
    """Run the full three-way comparison benchmark."""

    # ---- Load data ----
    print("=" * 70)
    print("SentinelEdge Feature Benchmark: TF-IDF vs. Sentence Embeddings")
    print("=" * 70)

    print(f"\nLoading training data from {train_csv} ...")
    train_df = pd.read_csv(train_csv, dtype={"text": str, "label": int})
    train_texts = train_df["text"].fillna("").tolist()
    y_train = train_df["label"].values

    print(f"  {len(train_texts):,} training samples")
    print(
        f"  Label distribution: "
        f"{dict(pd.Series(y_train).value_counts().sort_index())}"
    )

    print(f"\nLoading test data from {test_csv} ...")
    test_df = pd.read_csv(test_csv, dtype={"text": str, "label": int})
    test_texts = test_df["text"].fillna("").tolist()
    y_test = test_df["label"].values

    print(f"  {len(test_texts):,} test samples")

    # ---- Configuration 1: TF-IDF + XGBoost (baseline) ----
    print("\n" + "-" * 70)
    print("Config 1: TF-IDF (500) + Handcrafted (18) = 518 dims -> XGBoost")
    print("-" * 70)

    print("Loading TF-IDF vectorizer ...")
    tfidf = TfidfFeatureExtractor.load(tfidf_path)

    print("Extracting TF-IDF features ...")
    t0 = time.time()
    X_train_tfidf = extract_tfidf_features(train_texts, tfidf)
    tfidf_train_time = time.time() - t0
    print(f"  Train features: {X_train_tfidf.shape} in {tfidf_train_time:.1f}s")

    t0 = time.time()
    X_test_tfidf = extract_tfidf_features(test_texts, tfidf)
    tfidf_test_time = time.time() - t0
    print(f"  Test features:  {X_test_tfidf.shape} in {tfidf_test_time:.1f}s")

    print("Training XGBoost ...")
    t0 = time.time()
    metrics_tfidf_xgb = train_xgboost(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    tfidf_xgb_train_time = time.time() - t0
    print(f"  Training took {tfidf_xgb_train_time:.1f}s")

    # ---- Configuration 2: Embedding + XGBoost ----
    print("\n" + "-" * 70)
    print("Config 2: Embedding (384) + Handcrafted (18) = 402 dims -> XGBoost")
    print("-" * 70)

    print("Extracting embedding features ...")
    t0 = time.time()
    X_train_emb = extract_embedding_features(train_texts)
    emb_train_time = time.time() - t0
    print(f"  Train features: {X_train_emb.shape} in {emb_train_time:.1f}s")

    t0 = time.time()
    X_test_emb = extract_embedding_features(test_texts)
    emb_test_time = time.time() - t0
    print(f"  Test features:  {X_test_emb.shape} in {emb_test_time:.1f}s")

    print("Training XGBoost ...")
    t0 = time.time()
    metrics_emb_xgb = train_xgboost(X_train_emb, y_train, X_test_emb, y_test)
    emb_xgb_train_time = time.time() - t0
    print(f"  Training took {emb_xgb_train_time:.1f}s")

    # ---- Configuration 3: Embedding + MLP ----
    print("\n" + "-" * 70)
    print("Config 3: Embedding (384) + Handcrafted (18) = 402 dims -> MLP")
    print("-" * 70)

    print(f"Training MLP ({mlp_epochs} epochs) ...")
    t0 = time.time()
    metrics_emb_mlp = train_mlp(
        X_train_emb, y_train, X_test_emb, y_test, epochs=mlp_epochs
    )
    emb_mlp_train_time = time.time() - t0
    print(f"  Training took {emb_mlp_train_time:.1f}s")

    # ---- Results table ----
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    header = (
        f"{'Configuration':<45} {'Acc':>7} {'Prec':>7} "
        f"{'Rec':>7} {'F1':>7} {'Feat(s)':>8} {'Train(s)':>9}"
    )
    print(header)
    print("-" * len(header))

    configs = [
        (
            "TF-IDF(518) + XGBoost",
            metrics_tfidf_xgb,
            tfidf_train_time,
            tfidf_xgb_train_time,
        ),
        (
            "Embedding(402) + XGBoost",
            metrics_emb_xgb,
            emb_train_time,
            emb_xgb_train_time,
        ),
        (
            "Embedding(402) + MLP (federable)",
            metrics_emb_mlp,
            emb_train_time,
            emb_mlp_train_time,
        ),
    ]

    for name, metrics, feat_time, model_train_time in configs:
        print(
            f"  {name:<43} "
            f"{metrics['accuracy']:>7.4f} "
            f"{metrics['precision']:>7.4f} "
            f"{metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f} "
            f"{feat_time:>8.1f} "
            f"{model_train_time:>9.1f}"
        )

    print()

    # ---- Highlight best ----
    best_config = max(configs, key=lambda c: c[1]["f1"])
    print(f"Best F1: {best_config[0]} ({best_config[1]['f1']:.4f})")

    # ---- Feature dimension comparison ----
    print("\nFeature Dimension Summary:")
    print(f"  TF-IDF pipeline:     {X_train_tfidf.shape[1]} dims (18 handcrafted + 500 TF-IDF)")
    print(f"  Embedding pipeline:  {X_train_emb.shape[1]} dims (18 handcrafted + 384 embedding)")
    reduction = (
        (X_train_tfidf.shape[1] - X_train_emb.shape[1])
        / X_train_tfidf.shape[1]
        * 100
    )
    print(f"  Dimension reduction: {reduction:.1f}%")

    # ---- MLP model size ----
    mlp_temp = MLPClassifier(input_dim=X_train_emb.shape[1])
    print(f"\nMLP Model Size:")
    print(f"  Parameters:  {mlp_temp.n_params:,}")
    print(f"  Weight size: {mlp_temp.n_params * 4 / 1024:.1f} KB (float32)")
    print(f"  Federable:   Yes (get_weights/set_weights supported)")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TF-IDF vs. embedding features."
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/call_fraud_train.csv",
        help="Training CSV path",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed/call_fraud_test.csv",
        help="Test CSV path",
    )
    parser.add_argument(
        "--tfidf-path",
        default="models/tfidf_call_vectorizer.pkl",
        help="Path to fitted TF-IDF vectorizer",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=50,
        help="Number of MLP training epochs",
    )
    args = parser.parse_args()

    run_benchmark(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        tfidf_path=args.tfidf_path,
        mlp_epochs=args.mlp_epochs,
    )


if __name__ == "__main__":
    main()
