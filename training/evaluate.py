"""Evaluate trained models with detailed metrics.

Supports evaluation of call fraud, SMS fraud, and URL fraud models.
Produces accuracy, precision, recall, F1, AUC-PR, AUC-ROC, confusion
matrix, per-category breakdown, and feature importance analysis.

Usage
-----
    python training/evaluate.py \
        --model models/call_fraud_xgb.json \
        --test-csv data/processed/call_fraud_test.csv \
        --tfidf models/tfidf_call_vectorizer.pkl \
        --model-type call

    python training/evaluate.py \
        --model models/url_fraud_xgb.json \
        --test-csv data/processed/url_fraud_test.csv \
        --model-type url
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
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
# Feature extraction (reused from training scripts)
# ======================================================================

def extract_text_features(
    texts: list[str],
    tfidf: TfidfFeatureExtractor,
) -> np.ndarray:
    """Extract combined handcrafted + TF-IDF features for text-based models."""
    hc_list = []
    for t in texts:
        feats = extract_handcrafted_features(t)
        hc_list.append(list(feats.values()))
    hc_matrix = np.array(hc_list, dtype=np.float32)

    tfidf_matrix = np.zeros((len(texts), tfidf.n_features), dtype=np.float32)
    for i, t in enumerate(texts):
        tfidf_matrix[i] = tfidf.transform(t).astype(np.float32)

    return np.hstack([hc_matrix, tfidf_matrix])


def extract_url_features_batch(urls: list[str]) -> np.ndarray:
    """Extract URL structural features (imported from train_url_classifier)."""
    # Import here to avoid circular dependency issues
    from training.train_url_classifier import extract_url_features

    rows = []
    for url in urls:
        feats = extract_url_features(url)
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float32)


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_model(
    model_path: str,
    test_csv: str,
    tfidf_path: str | None = None,
    model_type: str = "call",
    output_dir: str | None = None,
) -> dict[str, float]:
    """Run comprehensive evaluation on a trained model.

    Parameters
    ----------
    model_path : str
        Path to the XGBoost model (.json).
    test_csv : str
        Path to the test CSV.
    tfidf_path : str | None
        Path to the TF-IDF vectorizer (required for call/sms models).
    model_type : str
        One of "call", "sms", "url".
    output_dir : str | None
        If provided, save evaluation plots here.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    print("=" * 70)
    print(f"Evaluating: {model_path}")
    print(f"Test data:  {test_csv}")
    print(f"Model type: {model_type}")
    print("=" * 70)

    # ---- Load model ----
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found.")
        return {}

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Model loaded successfully.")

    # ---- Load test data ----
    if not os.path.exists(test_csv):
        print(f"ERROR: Test CSV {test_csv} not found.")
        return {}

    test_df = pd.read_csv(test_csv, dtype={"text": str, "label": int})
    texts = test_df["text"].fillna("").tolist()
    y_true = test_df["label"].values

    print(f"Test samples: {len(texts):,}")
    print(f"Label distribution: {dict(pd.Series(y_true).value_counts().sort_index())}")

    # ---- Extract features ----
    print("Extracting features ...")
    t0 = time.time()

    if model_type in ("call", "sms"):
        if tfidf_path is None:
            print("ERROR: TF-IDF path required for call/sms models.")
            return {}
        if not os.path.exists(tfidf_path):
            print(f"ERROR: TF-IDF file {tfidf_path} not found.")
            return {}
        tfidf = TfidfFeatureExtractor.load(tfidf_path)
        X_test = extract_text_features(texts, tfidf)
    elif model_type == "url":
        X_test = extract_url_features_batch(texts)
    else:
        print(f"ERROR: Unknown model type '{model_type}'. Use 'call', 'sms', or 'url'.")
        return {}

    feature_time = time.time() - t0
    print(f"Feature extraction: {feature_time:.2f}s ({X_test.shape})")

    # ---- Predictions ----
    print("Running inference ...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.3f}s "
          f"({inference_time/len(texts)*1000:.3f} ms/sample)")

    # ---- Core Metrics ----
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    _print_core_metrics(metrics)

    # ---- Confusion Matrix ----
    _print_confusion_matrix(y_true, y_pred, model_type)

    # ---- Classification Report ----
    label_names = {
        "call": ["legit", "scam"],
        "sms": ["legit", "spam"],
        "url": ["legit", "phishing"],
    }
    names = label_names.get(model_type, ["class_0", "class_1"])
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    # ---- Per-category breakdown ----
    if "category" in test_df.columns:
        _print_category_breakdown(test_df, y_pred, y_prob)

    # ---- Threshold analysis ----
    _print_threshold_analysis(y_true, y_prob)

    # ---- Feature importance ----
    if model_type in ("call", "sms") and tfidf_path:
        tfidf = TfidfFeatureExtractor.load(tfidf_path)
        _print_text_feature_importance(model, tfidf)
    elif model_type == "url":
        _print_url_feature_importance(model)

    # ---- Latency statistics ----
    _print_latency_stats(model, X_test)

    # ---- Save plots if output dir specified ----
    if output_dir:
        _save_evaluation_plots(y_true, y_prob, y_pred, output_dir, model_type)

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)

    return metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC metrics (need at least 2 classes present)
    if len(np.unique(y_true)) >= 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    return metrics


def _print_core_metrics(metrics: dict[str, float]) -> None:
    """Print core evaluation metrics."""
    print("\n" + "-" * 50)
    print("Core Metrics")
    print("-" * 50)
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1']:.4f}")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR (AP):    {metrics['auc_pr']:.4f}")
    print("-" * 50)


def _print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
) -> None:
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    print("\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Legit     Fraud")
    print(f"  Actual Legit  {tn:>6}    {fp:>6}   (FPR: {fp/(tn+fp):.4f})" if (tn+fp) > 0 else "")
    print(f"  Actual Fraud  {fn:>6}    {tp:>6}   (FNR: {fn/(fn+tp):.4f})" if (fn+tp) > 0 else "")
    print()
    print(f"  True Negatives  (TN): {tn:>6} ({100*tn/total:.1f}%)")
    print(f"  False Positives (FP): {fp:>6} ({100*fp/total:.1f}%)")
    print(f"  False Negatives (FN): {fn:>6} ({100*fn/total:.1f}%)")
    print(f"  True Positives  (TP): {tp:>6} ({100*tp/total:.1f}%)")


def _print_category_breakdown(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Print per-category evaluation metrics."""
    df = test_df.copy()
    df["pred"] = y_pred
    df["prob"] = y_prob

    print("\nPer-Category Breakdown:")
    print(f"  {'Category':<30} {'N':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
          f"{'F1':>8} {'Avg Prob':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for cat in sorted(df["category"].unique()):
        mask = df["category"] == cat
        ct = df.loc[mask, "label"].values
        cp = df.loc[mask, "pred"].values
        cprob = df.loc[mask, "prob"].values
        n = mask.sum()

        acc = accuracy_score(ct, cp)
        prec = precision_score(ct, cp, zero_division=0)
        rec = recall_score(ct, cp, zero_division=0)
        f1 = f1_score(ct, cp, zero_division=0)
        avg_prob = float(cprob.mean())

        print(f"  {cat:<30} {n:>6} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} "
              f"{f1:>8.4f} {avg_prob:>10.4f}")
    print()


def _print_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Print metrics at various probability thresholds."""
    print("\nThreshold Analysis:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'FP Count':>10} {'FN Count':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        print(f"  {thresh:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} "
              f"{fp:>10} {fn:>10}")
    print()


def _print_text_feature_importance(
    model: xgb.XGBClassifier,
    tfidf: TfidfFeatureExtractor,
) -> None:
    """Print top-20 feature importances for text-based models."""
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

    print("\nTop-20 Feature Importances:")
    print(f"  {'Rank':>4}  {'Feature':<40} {'Importance':>12} {'Type':<12}")
    print(f"  {'-'*4}  {'-'*40} {'-'*12} {'-'*12}")
    for rank, idx in enumerate(indices[:20], 1):
        name = all_names[idx]
        ftype = "handcrafted" if idx < len(hc_names) else "tfidf"
        print(f"  {rank:>4}  {name:<40} {importances[idx]:>12.6f} {ftype:<12}")

    # Summary: handcrafted vs TF-IDF importance
    hc_total = float(importances[:len(hc_names)].sum())
    tfidf_total = float(importances[len(hc_names):].sum())
    total = hc_total + tfidf_total
    print(f"\n  Handcrafted features: {hc_total:.4f} ({100*hc_total/total:.1f}%)")
    print(f"  TF-IDF features:     {tfidf_total:.4f} ({100*tfidf_total/total:.1f}%)")
    print()


def _print_url_feature_importance(model: xgb.XGBClassifier) -> None:
    """Print feature importances for URL models."""
    from training.train_url_classifier import URL_FEATURE_NAMES

    importances = model.feature_importances_
    if len(importances) != len(URL_FEATURE_NAMES):
        print(f"  (Feature importance: dimension mismatch, skipping)")
        return

    indices = np.argsort(importances)[::-1]

    print("\nURL Feature Importances (all 14 features):")
    print(f"  {'Rank':>4}  {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*4}  {'-'*25} {'-'*12}")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>4}  {URL_FEATURE_NAMES[idx]:<25} {importances[idx]:>12.6f}")
    print()


def _print_latency_stats(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
) -> None:
    """Benchmark inference latency on the test set."""
    n_samples = min(len(X_test), 1000)  # Use up to 1000 samples
    X_bench = X_test[:n_samples]

    # Warm up
    model.predict(X_bench[:1])

    # Benchmark
    latencies = []
    for i in range(n_samples):
        t0 = time.time()
        model.predict(X_bench[i:i+1])
        latencies.append(time.time() - t0)

    latencies_ms = np.array(latencies) * 1000

    print("\nInference Latency (per sample):")
    print(f"  Mean:   {latencies_ms.mean():.3f} ms")
    print(f"  Median: {np.median(latencies_ms):.3f} ms")
    print(f"  P95:    {np.percentile(latencies_ms, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies_ms, 99):.3f} ms")
    print(f"  Min:    {latencies_ms.min():.3f} ms")
    print(f"  Max:    {latencies_ms.max():.3f} ms")
    print(f"  Throughput: ~{1000/latencies_ms.mean():.0f} samples/sec")
    print()


def _save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    model_type: str,
) -> None:
    """Save evaluation visualisation plots (ROC, PR curves, etc.)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    prefix = model_type

    # ---- ROC Curve ----
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_type.upper()} Fraud Classifier")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ROC curve: {roc_path}")

        # ---- Precision-Recall Curve ----
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_type.upper()} Fraud Classifier")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.png")
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved PR curve: {pr_path}")

    # ---- Probability Distribution ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        y_prob[y_true == 0], bins=50, alpha=0.6, label="Legit", color="green",
        density=True,
    )
    ax.hist(
        y_prob[y_true == 1], bins=50, alpha=0.6, label="Fraud", color="red",
        density=True,
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution - {model_type.upper()} Fraud Classifier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    dist_path = os.path.join(output_dir, f"{prefix}_score_distribution.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved score distribution: {dist_path}")


# ======================================================================
# Batch evaluation for all models
# ======================================================================

def evaluate_all(
    models_dir: str = "models",
    data_dir: str = "data/processed",
    output_dir: str | None = None,
) -> None:
    """Evaluate all available trained models."""
    configs = [
        {
            "model_path": os.path.join(models_dir, "call_fraud_xgb.json"),
            "test_csv": os.path.join(data_dir, "call_fraud_test.csv"),
            "tfidf_path": os.path.join(models_dir, "tfidf_call_vectorizer.pkl"),
            "model_type": "call",
        },
        {
            "model_path": os.path.join(models_dir, "sms_fraud_xgb.json"),
            "test_csv": os.path.join(data_dir, "sms_fraud_test.csv"),
            "tfidf_path": os.path.join(models_dir, "tfidf_sms_vectorizer.pkl"),
            "model_type": "sms",
        },
        {
            "model_path": os.path.join(models_dir, "url_fraud_xgb.json"),
            "test_csv": os.path.join(data_dir, "url_fraud_test.csv"),
            "tfidf_path": None,
            "model_type": "url",
        },
    ]

    all_metrics: dict[str, dict[str, float]] = {}

    for cfg in configs:
        if os.path.exists(cfg["model_path"]) and os.path.exists(cfg["test_csv"]):
            model_output = os.path.join(output_dir, cfg["model_type"]) if output_dir else None
            metrics = evaluate_model(
                model_path=cfg["model_path"],
                test_csv=cfg["test_csv"],
                tfidf_path=cfg["tfidf_path"],
                model_type=cfg["model_type"],
                output_dir=model_output,
            )
            all_metrics[cfg["model_type"]] = metrics
        else:
            print(f"\nSkipping {cfg['model_type']}: model or test data not found.")

    # ---- Summary table ----
    if all_metrics:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  {'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'AUC-ROC':>10} {'AUC-PR':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for name, m in all_metrics.items():
            print(f"  {name:<10} {m.get('accuracy',0):>10.4f} {m.get('precision',0):>10.4f} "
                  f"{m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f} "
                  f"{m.get('auc_roc',0):>10.4f} {m.get('auc_pr',0):>10.4f}")
        print("=" * 70)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained fraud detection models."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to XGBoost model (.json). If not specified, evaluates all models.",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Path to test CSV.",
    )
    parser.add_argument(
        "--tfidf",
        default=None,
        help="Path to TF-IDF vectorizer (for call/sms models).",
    )
    parser.add_argument(
        "--model-type",
        choices=["call", "sms", "url"],
        default="call",
        help="Model type (default: call)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Models directory (for batch evaluation)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Processed data directory (for batch evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save evaluation plots",
    )
    args = parser.parse_args()

    if args.model:
        # Single model evaluation
        evaluate_model(
            model_path=args.model,
            test_csv=args.test_csv or f"data/processed/{args.model_type}_fraud_test.csv",
            tfidf_path=args.tfidf,
            model_type=args.model_type,
            output_dir=args.output_dir,
        )
    else:
        # Batch evaluation of all available models
        evaluate_all(
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
