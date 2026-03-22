"""Evaluate the URL fraud classifier against real-world phishing and legitimate URLs.

Data sources:
  - Phishing:   PhishTank verified online feed (data.phishtank.com)
  - Legitimate: Majestic Million top domains (majestic.com)

Usage:
    python data/real/phishing_urls/evaluate_real_urls.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
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

from training.train_url_classifier import (
    URL_FEATURE_NAMES,
    extract_features_batch,
    extract_url_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHISHTANK_CSV = "/tmp/phishtank.csv"
MAJESTIC_CSV = "/tmp/majestic.csv"

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "url_fraud_xgb.json")

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "real", "phishing_urls")
DATASET_CSV = os.path.join(OUTPUT_DIR, "combined_real_urls.csv")
RESULTS_TXT = os.path.join(OUTPUT_DIR, "real_data_results.txt")

# How many URLs to sample from each class
N_PHISHING = 2000
N_LEGIT = 2000

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phishtank(path: str, n: int) -> list[str]:
    """Load verified phishing URLs from PhishTank CSV."""
    urls: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            if url:
                urls.append(url)
    print(f"  PhishTank: loaded {len(urls):,} total phishing URLs")

    # Sample
    rng = np.random.RandomState(RANDOM_SEED)
    if len(urls) > n:
        indices = rng.choice(len(urls), size=n, replace=False)
        urls = [urls[i] for i in indices]
    print(f"  PhishTank: sampled {len(urls):,} phishing URLs")
    return urls


def load_majestic(path: str, n: int) -> list[str]:
    """Load top legitimate domains from Majestic Million, construct full URLs."""
    domains: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            domain = row.get("Domain", "").strip()
            if domain:
                domains.append(domain)
    print(f"  Majestic: loaded top {len(domains):,} legitimate domains")

    # Construct URLs from domains
    urls = [f"https://{domain}" for domain in domains]
    return urls


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> str:
    """Run the full evaluation pipeline and return the results as a string."""
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("SentinelEdge URL Classifier -- Real-World Data Evaluation")
    log(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log("=" * 72)
    log()

    # ---- Load model ----
    log(f"Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log(f"ERROR: Model file not found at {MODEL_PATH}")
        return "\n".join(lines)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    log(f"  Loaded XGBoost model successfully")
    log(f"  Number of trees: {model.n_estimators}")
    log(f"  Max depth: {model.max_depth}")
    log()

    # ---- Load data ----
    log("--- Data Sources ---")

    phishing_urls = load_phishtank(PHISHTANK_CSV, N_PHISHING)
    legit_urls = load_majestic(MAJESTIC_CSV, N_LEGIT)
    log()

    all_urls = phishing_urls + legit_urls
    labels = np.array(
        [1] * len(phishing_urls) + [0] * len(legit_urls), dtype=np.int32
    )

    log(f"Combined dataset: {len(all_urls):,} URLs")
    log(f"  Phishing (label=1): {len(phishing_urls):,}")
    log(f"  Legitimate (label=0): {len(legit_urls):,}")
    log()

    # ---- Save combined dataset ----
    log(f"Saving combined dataset to {DATASET_CSV}")
    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label", "source"])
        for url in phishing_urls:
            writer.writerow([url, 1, "phishtank"])
        for url in legit_urls:
            writer.writerow([url, 0, "majestic_top2000"])
    log()

    # ---- Feature extraction ----
    log("--- Feature Extraction ---")
    t0 = time.time()
    X = extract_features_batch(all_urls)
    feat_time = time.time() - t0
    log(f"  Feature matrix shape: {X.shape}")
    log(f"  Feature extraction time: {feat_time:.2f}s ({feat_time/len(all_urls)*1000:.2f}ms per URL)")
    log()

    # Show feature statistics for each class
    log("  Feature means by class:")
    log(f"  {'Feature':<25} {'Phishing':>12} {'Legitimate':>12} {'Diff':>12}")
    log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for i, fname in enumerate(URL_FEATURE_NAMES):
        phish_mean = X[labels == 1, i].mean()
        legit_mean = X[labels == 0, i].mean()
        diff = phish_mean - legit_mean
        log(f"  {fname:<25} {phish_mean:>12.4f} {legit_mean:>12.4f} {diff:>+12.4f}")
    log()

    # ---- Model prediction ----
    log("--- Model Predictions ---")
    t0 = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    if y_proba.ndim == 2:
        y_scores = y_proba[:, 1]
    else:
        y_scores = y_proba
    pred_time = time.time() - t0
    log(f"  Prediction time: {pred_time:.2f}s ({pred_time/len(all_urls)*1000:.2f}ms per URL)")
    log()

    # ---- Overall metrics ----
    log("--- Overall Metrics ---")
    acc = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred, zero_division=0)
    rec = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(labels, y_scores)
    except ValueError:
        auc = float("nan")

    log(f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)")
    log(f"  Precision:   {prec:.4f}  ({prec*100:.1f}%)")
    log(f"  Recall:      {rec:.4f}  ({rec*100:.1f}%)")
    log(f"  F1 Score:    {f1:.4f}  ({f1*100:.1f}%)")
    log(f"  ROC AUC:     {auc:.4f}")
    log()

    cm = confusion_matrix(labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log("  Confusion Matrix:")
    log(f"                  Predicted Legit   Predicted Phish")
    log(f"    Actual Legit       TN={tn:>5}         FP={fp:>5}")
    log(f"    Actual Phish       FN={fn:>5}         TP={tp:>5}")
    log()
    log(f"  False Positive Rate: {fp/(fp+tn)*100:.1f}% (legit URLs flagged as phishing)")
    log(f"  False Negative Rate: {fn/(fn+tp)*100:.1f}% (phishing URLs missed)")
    log()

    log("  Classification Report:")
    report = classification_report(
        labels, y_pred, target_names=["legitimate", "phishing"]
    )
    for line in report.split("\n"):
        log(f"  {line}")
    log()

    # ---- Score distribution ----
    log("--- Score Distribution ---")
    phish_scores = y_scores[labels == 1]
    legit_scores = y_scores[labels == 0]

    log(f"  Phishing URLs:")
    log(f"    Mean score:   {phish_scores.mean():.4f}")
    log(f"    Median score: {np.median(phish_scores):.4f}")
    log(f"    Std dev:      {phish_scores.std():.4f}")
    log(f"    Min/Max:      {phish_scores.min():.4f} / {phish_scores.max():.4f}")
    log()
    log(f"  Legitimate URLs:")
    log(f"    Mean score:   {legit_scores.mean():.4f}")
    log(f"    Median score: {np.median(legit_scores):.4f}")
    log(f"    Std dev:      {legit_scores.std():.4f}")
    log(f"    Min/Max:      {legit_scores.min():.4f} / {legit_scores.max():.4f}")
    log()

    # Score buckets
    log("  Score distribution by bucket:")
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    log(f"  {'Bucket':<12} {'Phishing':>10} {'Legitimate':>12} {'Total':>8}")
    log(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
    for lo, hi in buckets:
        p_count = int(((phish_scores >= lo) & (phish_scores < hi)).sum())
        l_count = int(((legit_scores >= lo) & (legit_scores < hi)).sum())
        label = f"[{lo:.1f}, {hi:.1f})"
        log(f"  {label:<12} {p_count:>10} {l_count:>12} {p_count+l_count:>8}")
    log()

    # ---- Threshold analysis ----
    log("--- Threshold Analysis ---")
    log(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds_t = (y_scores >= thresh).astype(int)
        acc_t = accuracy_score(labels, preds_t)
        prec_t = precision_score(labels, preds_t, zero_division=0)
        rec_t = recall_score(labels, preds_t, zero_division=0)
        f1_t = f1_score(labels, preds_t, zero_division=0)
        cm_t = confusion_matrix(labels, preds_t)
        tn_t, fp_t = cm_t[0, 0], cm_t[0, 1]
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
        log(f"  {thresh:>10.1f} {acc_t:>10.4f} {prec_t:>10.4f} {rec_t:>10.4f} {f1_t:>10.4f} {fpr_t:>10.4f}")
    log()

    # ---- Misclassified examples ----
    log("--- Misclassified Examples ---")
    log()

    # False positives (legit flagged as phishing)
    fp_mask = (labels == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    log(f"  False Positives (legitimate URLs flagged as phishing): {fp_indices.size}")
    if fp_indices.size > 0:
        for idx in fp_indices[:15]:
            log(f"    score={y_scores[idx]:.4f}  {all_urls[idx]}")
    log()

    # False negatives (phishing missed)
    fn_mask = (labels == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    log(f"  False Negatives (phishing URLs missed): {fn_indices.size}")
    if fn_indices.size > 0:
        for idx in fn_indices[:15]:
            log(f"    score={y_scores[idx]:.4f}  {all_urls[idx]}")
    log()

    # ---- Feature importance from the trained model ----
    log("--- Feature Importance (from trained model) ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    log(f"  {'Rank':>4}  {'Feature':<25} {'Importance':>12}")
    log(f"  {'-'*4}  {'-'*25} {'-'*12}")
    for rank, idx in enumerate(indices, 1):
        log(f"  {rank:>4}  {URL_FEATURE_NAMES[idx]:<25} {importances[idx]:>12.6f}")
    log()

    # ---- Analysis and diagnosis ----
    log("--- Analysis ---")
    log()

    if f1 < 0.5:
        log("  VERDICT: The model FAILS on real-world data.")
        log()
        log("  Root cause analysis:")
        log("  The model was trained on synthetic data generated by")
        log("  training/generate_synthetic_data.py. The feature distributions")
        log("  in that synthetic data likely do not match real phishing URLs.")
        log()
        if rec < 0.5:
            log(f"  - LOW RECALL ({rec:.1%}): The model misses {fn_indices.size}/{len(phishing_urls)} real phishing URLs.")
            log("    Real phishing URLs may have feature distributions that differ")
            log("    significantly from the synthetic training data.")
        if prec < 0.5:
            log(f"  - LOW PRECISION ({prec:.1%}): {fp_indices.size}/{len(legit_urls)} legitimate URLs falsely flagged.")
            log("    The model's decision boundary may not generalize to real domains.")
        log()
        log("  Recommendations:")
        log("  1. Retrain on real phishing data (PhishTank + legitimate domains)")
        log("  2. Add more discriminative features (WHOIS age, certificate info, etc.)")
        log("  3. Use domain reputation lists as additional features")
        log("  4. Consider an ensemble with a deep learning model on raw URL strings")
    elif f1 < 0.8:
        log("  VERDICT: The model shows moderate performance on real-world data.")
        log(f"  F1={f1:.4f}, which is below production-ready thresholds.")
        log()
        log("  Recommendations:")
        log("  1. Fine-tune on a mix of real and synthetic data")
        log("  2. Adjust decision threshold based on the threshold analysis above")
        log("  3. Add more features (WHOIS data, page content analysis)")
    else:
        log("  VERDICT: The model performs well on real-world data.")
        log(f"  F1={f1:.4f}")
    log()

    log("=" * 72)
    log("End of evaluation report")
    log("=" * 72)

    return "\n".join(lines)


def main() -> None:
    results = run_evaluation()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(results)
        f.write("\n")

    print(f"\nResults saved to {RESULTS_TXT}")


if __name__ == "__main__":
    main()
