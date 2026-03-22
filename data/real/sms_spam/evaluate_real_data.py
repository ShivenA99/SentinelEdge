"""Evaluate the trained SMS fraud classifier against the real UCI SMS Spam Collection.

This script loads the real-world SMS Spam Collection dataset (5,574 messages),
extracts features using the same pipeline used during training, runs predictions
with the trained XGBoost model, and computes detailed metrics.

Usage
-----
    python data/real/sms_spam/evaluate_real_data.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.tfidf import TfidfFeatureExtractor


# ======================================================================
# Data loading
# ======================================================================

def load_sms_spam_collection(path: str) -> tuple[list[str], list[int]]:
    """Load the UCI SMS Spam Collection TSV file.

    Parameters
    ----------
    path : str
        Path to the SMSSpamCollection.tsv file (tab-separated, label\\ttext).

    Returns
    -------
    tuple[list[str], list[int]]
        (texts, labels) where labels: 1 = spam, 0 = ham.
    """
    texts: list[str] = []
    labels: list[int] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                print(f"  WARNING: Skipping malformed line {line_num}: {line[:80]}")
                continue
            label_str, text = parts
            label_str = label_str.strip().lower()
            if label_str == "spam":
                labels.append(1)
            elif label_str == "ham":
                labels.append(0)
            else:
                print(f"  WARNING: Unknown label '{label_str}' on line {line_num}, skipping")
                continue
            texts.append(text)

    return texts, labels


# ======================================================================
# Feature extraction (same pipeline as training)
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
    hc_list = []
    for t in texts:
        feats = extract_handcrafted_features(t)
        hc_list.append(list(feats.values()))
    hc_matrix = np.array(hc_list, dtype=np.float32)

    tfidf_matrix = np.zeros((len(texts), tfidf.n_features), dtype=np.float32)
    for i, t in enumerate(texts):
        tfidf_matrix[i] = tfidf.transform(t).astype(np.float32)

    combined = np.hstack([hc_matrix, tfidf_matrix])
    return combined


# ======================================================================
# Analysis helpers
# ======================================================================

def analyze_errors(
    texts: list[str],
    y_true: list[int],
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, list[tuple[str, float]]]:
    """Collect false positives and false negatives with their predicted probabilities."""
    false_positives: list[tuple[str, float]] = []
    false_negatives: list[tuple[str, float]] = []

    for i in range(len(texts)):
        if y_true[i] == 0 and y_pred[i] == 1:
            false_positives.append((texts[i], float(y_prob[i])))
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives.append((texts[i], float(y_prob[i])))

    # Sort by probability (most confident errors first)
    false_positives.sort(key=lambda x: x[1], reverse=True)
    false_negatives.sort(key=lambda x: x[1])

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def analyze_handcrafted_feature_distribution(
    texts: list[str],
    labels: list[int],
) -> str:
    """Compare handcrafted feature distributions between ham and spam in real data."""
    ham_feats: dict[str, list[float]] = {}
    spam_feats: dict[str, list[float]] = {}

    for text, label in zip(texts, labels):
        feats = extract_handcrafted_features(text)
        target = spam_feats if label == 1 else ham_feats
        for k, v in feats.items():
            target.setdefault(k, []).append(v)

    lines = []
    lines.append(f"{'Feature':<25} {'Ham Mean':>10} {'Ham Std':>10} {'Spam Mean':>10} {'Spam Std':>10} {'Sep':>8}")
    lines.append("-" * 75)
    for feat_name in extract_handcrafted_features("dummy").keys():
        ham_vals = np.array(ham_feats.get(feat_name, [0]))
        spam_vals = np.array(spam_feats.get(feat_name, [0]))
        ham_mean = ham_vals.mean()
        ham_std = ham_vals.std()
        spam_mean = spam_vals.mean()
        spam_std = spam_vals.std()
        # Separation metric: difference in means / pooled std
        pooled_std = np.sqrt((ham_std**2 + spam_std**2) / 2) if (ham_std + spam_std) > 0 else 1.0
        separation = abs(spam_mean - ham_mean) / pooled_std if pooled_std > 0 else 0.0
        lines.append(
            f"{feat_name:<25} {ham_mean:>10.3f} {ham_std:>10.3f} {spam_mean:>10.3f} {spam_std:>10.3f} {separation:>8.3f}"
        )
    return "\n".join(lines)


# ======================================================================
# Main evaluation
# ======================================================================

def main() -> None:
    # Paths
    data_path = os.path.join(_SCRIPT_DIR, "SMSSpamCollection.tsv")
    model_path = os.path.join(_PROJECT_ROOT, "models", "sms_fraud_xgb.json")
    tfidf_path = os.path.join(_PROJECT_ROOT, "models", "tfidf_sms_vectorizer.pkl")
    output_path = os.path.join(_SCRIPT_DIR, "real_data_results.txt")

    # Collect output lines
    results: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        results.append(msg)

    log("=" * 80)
    log("SentinelEdge SMS Classifier - Real Data Evaluation")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("Dataset: UCI SMS Spam Collection (5,574 real SMS messages)")
    log("Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")
    log("=" * 80)

    # ---- Check files exist ----
    for label, path in [("Dataset", data_path), ("Model", model_path), ("TF-IDF", tfidf_path)]:
        if not os.path.exists(path):
            log(f"ERROR: {label} not found at {path}")
            sys.exit(1)
        log(f"{label}: {path}")
    log()

    # ---- Load data ----
    log("Loading real SMS data ...")
    texts, labels = load_sms_spam_collection(data_path)
    label_counts = Counter(labels)
    log(f"  Total messages: {len(texts):,}")
    log(f"  Ham (legitimate): {label_counts[0]:,} ({100*label_counts[0]/len(texts):.1f}%)")
    log(f"  Spam: {label_counts[1]:,} ({100*label_counts[1]/len(texts):.1f}%)")
    log()

    # ---- Load model and vectorizer ----
    log("Loading trained model ...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    log(f"  Model: {model_path}")
    log(f"  Num features expected: {model.n_features_in_}")

    log("Loading TF-IDF vectorizer ...")
    tfidf = TfidfFeatureExtractor.load(tfidf_path)
    log(f"  Vectorizer: {tfidf_path}")
    log(f"  Vocabulary size: {tfidf.n_features}")
    log()

    # ---- Extract features ----
    log("Extracting features from real data ...")
    t0 = time.time()
    X = extract_features(texts, tfidf)
    feat_time = time.time() - t0
    log(f"  Feature matrix shape: {X.shape}")
    log(f"  Feature extraction time: {feat_time:.1f}s ({feat_time/len(texts)*1000:.2f}ms per message)")
    log()

    # ---- Run predictions ----
    log("Running predictions ...")
    t0 = time.time()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    pred_time = time.time() - t0
    log(f"  Prediction time: {pred_time:.2f}s ({pred_time/len(texts)*1000:.3f}ms per message)")
    log()

    # ---- Overall metrics ----
    y_true = np.array(labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    log("=" * 80)
    log("OVERALL RESULTS")
    log("=" * 80)
    log(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    log(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    log(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    log(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    log(f"  ROC AUC:   {auc:.4f} ({auc*100:.2f}%)")
    log()

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    log("Confusion Matrix:")
    log(f"                  Predicted Ham   Predicted Spam")
    log(f"  Actual Ham      TN = {cm[0,0]:>6}     FP = {cm[0,1]:>6}")
    log(f"  Actual Spam     FN = {cm[1,0]:>6}     TP = {cm[1,1]:>6}")
    log()

    # ---- Classification report ----
    log("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"])
    log(report)

    # ---- Prediction distribution ----
    log("Prediction Probability Distribution:")
    log(f"  Ham messages (n={label_counts[0]}):")
    ham_probs = y_prob[y_true == 0]
    log(f"    Mean prob(spam): {ham_probs.mean():.4f}")
    log(f"    Std:             {ham_probs.std():.4f}")
    log(f"    Min:             {ham_probs.min():.4f}")
    log(f"    Max:             {ham_probs.max():.4f}")
    log(f"    Median:          {np.median(ham_probs):.4f}")
    log()
    log(f"  Spam messages (n={label_counts[1]}):")
    spam_probs = y_prob[y_true == 1]
    log(f"    Mean prob(spam): {spam_probs.mean():.4f}")
    log(f"    Std:             {spam_probs.std():.4f}")
    log(f"    Min:             {spam_probs.min():.4f}")
    log(f"    Max:             {spam_probs.max():.4f}")
    log(f"    Median:          {np.median(spam_probs):.4f}")
    log()

    # ---- Threshold analysis ----
    log("=" * 80)
    log("THRESHOLD ANALYSIS")
    log("=" * 80)
    log(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6}")
    log("-" * 64)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds_t = (y_prob >= thresh).astype(int)
        acc_t = accuracy_score(y_true, preds_t)
        prec_t = precision_score(y_true, preds_t, zero_division=0)
        rec_t = recall_score(y_true, preds_t, zero_division=0)
        f1_t = f1_score(y_true, preds_t, zero_division=0)
        cm_t = confusion_matrix(y_true, preds_t)
        fp_t = cm_t[0, 1] if cm_t.shape[1] > 1 else 0
        fn_t = cm_t[1, 0]
        log(f"{thresh:>10.1f} {acc_t:>10.4f} {prec_t:>10.4f} {rec_t:>10.4f} {f1_t:>10.4f} {fp_t:>6} {fn_t:>6}")
    log()

    # ---- Error analysis ----
    errors = analyze_errors(texts, labels, y_pred, y_prob)

    log("=" * 80)
    log("ERROR ANALYSIS")
    log("=" * 80)

    fp_list = errors["false_positives"]
    fn_list = errors["false_negatives"]

    log(f"\nFalse Positives (ham classified as spam): {len(fp_list)}")
    log("-" * 60)
    for i, (text, prob) in enumerate(fp_list[:20]):
        log(f"  [{i+1}] prob={prob:.3f} | {text[:120]}")
    if len(fp_list) > 20:
        log(f"  ... and {len(fp_list) - 20} more")

    log(f"\nFalse Negatives (spam classified as ham): {len(fn_list)}")
    log("-" * 60)
    for i, (text, prob) in enumerate(fn_list[:20]):
        log(f"  [{i+1}] prob={prob:.3f} | {text[:120]}")
    if len(fn_list) > 20:
        log(f"  ... and {len(fn_list) - 20} more")
    log()

    # ---- Handcrafted feature distribution ----
    log("=" * 80)
    log("HANDCRAFTED FEATURE DISTRIBUTION (Real Data)")
    log("=" * 80)
    log(analyze_handcrafted_feature_distribution(texts, labels))
    log()

    # ---- Honest assessment ----
    log("=" * 80)
    log("HONEST ASSESSMENT")
    log("=" * 80)

    # Determine if model is performing well or poorly
    if f1 >= 0.85:
        log("The model performs WELL on real data.")
        log(f"F1={f1:.4f} indicates strong generalization from synthetic training data.")
    elif f1 >= 0.60:
        log("The model performs MODERATELY on real data.")
        log(f"F1={f1:.4f} shows partial generalization but significant room for improvement.")
    else:
        log("The model performs POORLY on real data.")
        log(f"F1={f1:.4f} indicates the model fails to generalize from synthetic to real data.")

    log()
    log("Key observations:")

    # Check if model is biased toward one class
    pred_counts = Counter(y_pred)
    if pred_counts.get(1, 0) == 0:
        log("- CRITICAL: Model predicts ALL messages as ham (never detects spam)")
        log("  This means the model learned nothing useful from synthetic training data")
    elif pred_counts.get(0, 0) == 0:
        log("- CRITICAL: Model predicts ALL messages as spam")
    else:
        log(f"- Model predicts {pred_counts[0]} ham and {pred_counts[1]} spam")

    if len(fn_list) > 0:
        fn_rate = len(fn_list) / label_counts[1]
        log(f"- Missed {len(fn_list)} out of {label_counts[1]} real spam messages ({fn_rate*100:.1f}% miss rate)")

    if len(fp_list) > 0:
        fp_rate = len(fp_list) / label_counts[0]
        log(f"- Falsely flagged {len(fp_list)} out of {label_counts[0]} legitimate messages ({fp_rate*100:.1f}% false alarm rate)")

    log()
    log("Training data was synthetic (template-generated). Real-world SMS data differs in:")
    log("- Language: informal abbreviations, slang, non-English text")
    log("- Spam patterns: UK-specific shortcodes, premium rate numbers, different scam types")
    log("- Message length: real messages vary more widely")
    log("- TF-IDF vocabulary: trained on synthetic templates, real vocabulary differs significantly")
    log()

    # ---- Save results ----
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
