"""Retrain XGBoost classifier with adversarial data and evaluate.

This script:
1. Generates adversarial data (if the CSV does not exist)
2. Loads original training data + adversarial data
3. Re-fits the TF-IDF vectorizer on the combined corpus
4. Retrains the XGBoost model
5. Evaluates overall accuracy on the standard test set
6. Measures accuracy specifically on adversarial scam examples
7. Measures false-positive rate on hard negatives

Usage
-----
    python training/retrain_with_adversarial.py

Outputs
-------
- models/call_fraud_xgb_adversarial.json  (retrained model)
- models/tfidf_call_vectorizer_adversarial.pkl  (re-fitted TF-IDF)
- Prints: "Model accuracy on adversarial scams: X%"
- Prints: "False positive rate on hard negatives: X%"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.tfidf import TfidfFeatureExtractor


# ======================================================================
# Feature extraction  (mirrors train_call_classifier.py)
# ======================================================================

def extract_features(
    texts: list[str],
    tfidf: TfidfFeatureExtractor,
) -> np.ndarray:
    """Extract combined handcrafted (18) + TF-IDF (N) feature matrix."""
    hc_list = []
    for t in texts:
        feats = extract_handcrafted_features(t)
        hc_list.append(list(feats.values()))
    hc_matrix = np.array(hc_list, dtype=np.float32)

    tfidf_matrix = np.zeros((len(texts), tfidf.n_features), dtype=np.float32)
    for i, t in enumerate(texts):
        tfidf_matrix[i] = tfidf.transform(t).astype(np.float32)

    return np.hstack([hc_matrix, tfidf_matrix])


# ======================================================================
# Evaluate helper
# ======================================================================

def _evaluate_subset(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    subset_name: str,
) -> dict[str, float]:
    """Evaluate model on a subset and print metrics."""
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(tn + fp, 1)

    print(f"\n  --- {subset_name} ---")
    print(f"  Samples:   {len(y_true):,}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  TN={tn:>5}  FP={fp:>5}")
    print(f"  FN={fn:>5}  TP={tp:>5}")
    print(f"  FPR:       {fpr:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _evaluate_per_category(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    categories: pd.Series,
) -> None:
    """Print per-category breakdown."""
    y_pred = model.predict(X)

    print(f"\n  {'Category':<35} {'N':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for cat in sorted(categories.unique()):
        mask = (categories == cat).values
        ct = y_true[mask]
        cp = y_pred[mask]
        n = mask.sum()
        acc = accuracy_score(ct, cp)
        prec = precision_score(ct, cp, zero_division=0)
        rec = recall_score(ct, cp, zero_division=0)
        f1 = f1_score(ct, cp, zero_division=0)
        print(f"  {cat:<35} {n:>6} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain call fraud classifier with adversarial data."
    )
    parser.add_argument(
        "--original-train-csv",
        default="data/processed/call_fraud_train.csv",
        help="Original training CSV",
    )
    parser.add_argument(
        "--original-test-csv",
        default="data/processed/call_fraud_test.csv",
        help="Original test CSV",
    )
    parser.add_argument(
        "--adversarial-csv",
        default="data/raw/synthetic_transcripts/adversarial_calls.csv",
        help="Adversarial data CSV",
    )
    parser.add_argument(
        "--output-model",
        default="models/call_fraud_xgb_adversarial.json",
        help="Output retrained model path",
    )
    parser.add_argument(
        "--output-tfidf",
        default="models/tfidf_call_vectorizer_adversarial.pkl",
        help="Output re-fitted TF-IDF path",
    )
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 70)
    print("Retrain XGBoost with Adversarial Data")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: Generate adversarial data if it does not exist
    # ------------------------------------------------------------------
    if not os.path.exists(args.adversarial_csv):
        print(f"\nAdversarial CSV not found at {args.adversarial_csv}")
        print("Generating adversarial data ...")
        from training.generate_adversarial_data import (
            generate_adversarial_dataset,
            write_csv,
            print_stats,
        )
        rows = generate_adversarial_dataset(seed=123)
        print_stats(rows)
        write_csv(rows, args.adversarial_csv)
    else:
        print(f"\nAdversarial CSV found at {args.adversarial_csv}")

    # ------------------------------------------------------------------
    # Step 1: Load all data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")

    # Original training data
    if os.path.exists(args.original_train_csv):
        orig_train = pd.read_csv(args.original_train_csv, dtype={"text": str, "label": int})
        orig_train["text"] = orig_train["text"].fillna("")
        print(f"  Original train: {len(orig_train):,} rows")
    else:
        print(f"  WARNING: Original train CSV not found at {args.original_train_csv}")
        orig_train = pd.DataFrame(columns=["text", "label", "category"])

    # Original test data
    if os.path.exists(args.original_test_csv):
        orig_test = pd.read_csv(args.original_test_csv, dtype={"text": str, "label": int})
        orig_test["text"] = orig_test["text"].fillna("")
        print(f"  Original test:  {len(orig_test):,} rows")
    else:
        print(f"  WARNING: Original test CSV not found at {args.original_test_csv}")
        orig_test = pd.DataFrame(columns=["text", "label", "category"])

    # Adversarial data
    adv_full = pd.read_csv(args.adversarial_csv, dtype={"text": str, "label": int})
    adv_full["text"] = adv_full["text"].fillna("")
    print(f"  Adversarial:    {len(adv_full):,} rows")

    # Split adversarial into scam and hard-negative subsets
    adv_scam = adv_full[adv_full["label"] == 1].copy()
    adv_hard_neg = adv_full[adv_full["label"] == 0].copy()
    print(f"    Adversarial scam sentences:       {len(adv_scam):,}")
    print(f"    Hard-negative legit sentences:     {len(adv_hard_neg):,}")

    # Split adversarial data 80/20 for train/test
    if len(adv_full) > 0:
        adv_train, adv_test = train_test_split(
            adv_full, test_size=0.2, random_state=42, stratify=adv_full["label"],
        )
        adv_train = adv_train.reset_index(drop=True)
        adv_test = adv_test.reset_index(drop=True)
    else:
        adv_train = pd.DataFrame(columns=["text", "label", "category"])
        adv_test = pd.DataFrame(columns=["text", "label", "category"])

    print(f"  Adversarial train split: {len(adv_train):,}")
    print(f"  Adversarial test split:  {len(adv_test):,}")

    # Combine original + adversarial for training
    combined_train = pd.concat([orig_train, adv_train], ignore_index=True)
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Combined training set: {len(combined_train):,} rows")
    print(f"  Label distribution: {dict(combined_train['label'].value_counts().sort_index())}")

    # ------------------------------------------------------------------
    # Step 2: Fit TF-IDF on combined training corpus
    # ------------------------------------------------------------------
    print("\n[2] Fitting TF-IDF vectorizer on combined corpus ...")
    t0 = time.time()
    tfidf = TfidfFeatureExtractor(model_path=None)
    tfidf.vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf.fit(combined_train["text"].tolist())
    print(f"  Vocabulary size: {tfidf.n_features}")
    print(f"  Fit time: {time.time() - t0:.1f}s")

    # Save TF-IDF
    os.makedirs(os.path.dirname(args.output_tfidf) or ".", exist_ok=True)
    tfidf.save(args.output_tfidf)
    print(f"  Saved TF-IDF to {args.output_tfidf}")

    # ------------------------------------------------------------------
    # Step 3: Extract features
    # ------------------------------------------------------------------
    print("\n[3] Extracting features ...")
    t0 = time.time()

    X_train = extract_features(combined_train["text"].tolist(), tfidf)
    y_train = combined_train["label"].values
    print(f"  Training features: {X_train.shape}  ({time.time() - t0:.1f}s)")

    feature_sets = {}

    # Original test set features
    if len(orig_test) > 0:
        t0 = time.time()
        X_orig_test = extract_features(orig_test["text"].tolist(), tfidf)
        y_orig_test = orig_test["label"].values
        feature_sets["orig_test"] = (X_orig_test, y_orig_test, orig_test)
        print(f"  Original test features: {X_orig_test.shape}  ({time.time() - t0:.1f}s)")

    # Adversarial test set features
    if len(adv_test) > 0:
        t0 = time.time()
        X_adv_test = extract_features(adv_test["text"].tolist(), tfidf)
        y_adv_test = adv_test["label"].values
        feature_sets["adv_test"] = (X_adv_test, y_adv_test, adv_test)
        print(f"  Adversarial test features: {X_adv_test.shape}  ({time.time() - t0:.1f}s)")

    # Full adversarial scam features (for the key metric)
    if len(adv_scam) > 0:
        t0 = time.time()
        X_adv_scam = extract_features(adv_scam["text"].tolist(), tfidf)
        y_adv_scam = adv_scam["label"].values
        feature_sets["adv_scam"] = (X_adv_scam, y_adv_scam, adv_scam)
        print(f"  Full adversarial scam features: {X_adv_scam.shape}  ({time.time() - t0:.1f}s)")

    # Full hard-negative features (for the key metric)
    if len(adv_hard_neg) > 0:
        t0 = time.time()
        X_hard_neg = extract_features(adv_hard_neg["text"].tolist(), tfidf)
        y_hard_neg = adv_hard_neg["label"].values
        feature_sets["hard_neg"] = (X_hard_neg, y_hard_neg, adv_hard_neg)
        print(f"  Full hard-negative features: {X_hard_neg.shape}  ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 4: Train XGBoost
    # ------------------------------------------------------------------
    print("\n[4] Training XGBoost classifier ...")
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    print(f"  Training completed in {time.time() - t0:.1f}s")

    # Save model
    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    model.save_model(args.output_model)
    print(f"  Model saved to {args.output_model}")

    # ------------------------------------------------------------------
    # Step 5: Evaluate on original test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if "orig_test" in feature_sets:
        X, y, df = feature_sets["orig_test"]
        _evaluate_subset(model, X, y, "Original Test Set")
        if "category" in df.columns:
            _evaluate_per_category(model, X, y, df["category"])

    # ------------------------------------------------------------------
    # Step 6: Evaluate on adversarial test set
    # ------------------------------------------------------------------
    if "adv_test" in feature_sets:
        X, y, df = feature_sets["adv_test"]
        _evaluate_subset(model, X, y, "Adversarial Test Set (held-out 20%)")
        if "category" in df.columns:
            _evaluate_per_category(model, X, y, df["category"])

    # ------------------------------------------------------------------
    # Step 7: Key metrics on FULL adversarial data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY ADVERSARIAL METRICS")
    print("=" * 70)

    # Adversarial scams: what fraction does the model correctly detect?
    if "adv_scam" in feature_sets:
        X_scam, y_scam, df_scam = feature_sets["adv_scam"]
        y_pred_scam = model.predict(X_scam)
        scam_acc = accuracy_score(y_scam, y_pred_scam)
        scam_recall = recall_score(y_scam, y_pred_scam, zero_division=0)
        print(f"\n  Model accuracy on adversarial scams: {100 * scam_acc:.1f}%")
        print(f"  Model recall on adversarial scams:   {100 * scam_recall:.1f}%")
        print(f"  (Detected {int(y_pred_scam.sum())} / {len(y_scam)} adversarial scam sentences)")

        # Per-category breakdown for scam
        if "category" in df_scam.columns:
            print()
            _evaluate_per_category(model, X_scam, y_scam, df_scam["category"])

    # Hard negatives: what fraction does the model incorrectly flag?
    if "hard_neg" in feature_sets:
        X_hn, y_hn, df_hn = feature_sets["hard_neg"]
        y_pred_hn = model.predict(X_hn)
        # FP rate: fraction of legit (label=0) predicted as scam (pred=1)
        fp_count = int((y_pred_hn == 1).sum())
        total_hn = len(y_hn)
        fp_rate = fp_count / max(total_hn, 1)
        print(f"\n  False positive rate on hard negatives: {100 * fp_rate:.1f}%")
        print(f"  (Incorrectly flagged {fp_count} / {total_hn} legitimate sentences as scam)")

        # Per-category breakdown for hard negatives
        if "category" in df_hn.columns:
            print()
            _evaluate_per_category(model, X_hn, y_hn, df_hn["category"])

    # ------------------------------------------------------------------
    # Step 8: Compare with baseline model (if available)
    # ------------------------------------------------------------------
    baseline_model_path = "models/call_fraud_xgb.json"
    baseline_tfidf_path = "models/tfidf_call_vectorizer.pkl"

    if os.path.exists(baseline_model_path) and os.path.exists(baseline_tfidf_path):
        print("\n" + "=" * 70)
        print("BASELINE MODEL COMPARISON (original model on adversarial data)")
        print("=" * 70)

        baseline_model = xgb.XGBClassifier()
        baseline_model.load_model(baseline_model_path)
        baseline_tfidf = TfidfFeatureExtractor.load(baseline_tfidf_path)

        # Adversarial scams through baseline
        if len(adv_scam) > 0:
            X_scam_base = extract_features(adv_scam["text"].tolist(), baseline_tfidf)
            y_pred_base_scam = baseline_model.predict(X_scam_base)
            base_scam_acc = accuracy_score(adv_scam["label"].values, y_pred_base_scam)
            base_scam_recall = recall_score(adv_scam["label"].values, y_pred_base_scam, zero_division=0)
            print(f"\n  Baseline accuracy on adversarial scams: {100 * base_scam_acc:.1f}%")
            print(f"  Baseline recall on adversarial scams:   {100 * base_scam_recall:.1f}%")
            print(f"  (Detected {int(y_pred_base_scam.sum())} / {len(adv_scam)} adversarial scam sentences)")

            if "category" in adv_scam.columns:
                _evaluate_per_category(baseline_model, X_scam_base, adv_scam["label"].values, adv_scam["category"])

        # Hard negatives through baseline
        if len(adv_hard_neg) > 0:
            X_hn_base = extract_features(adv_hard_neg["text"].tolist(), baseline_tfidf)
            y_pred_base_hn = baseline_model.predict(X_hn_base)
            base_fp = int((y_pred_base_hn == 1).sum())
            base_fp_rate = base_fp / max(len(adv_hard_neg), 1)
            print(f"\n  Baseline FP rate on hard negatives: {100 * base_fp_rate:.1f}%")
            print(f"  (Incorrectly flagged {base_fp} / {len(adv_hard_neg)} legitimate sentences)")

            if "category" in adv_hard_neg.columns:
                _evaluate_per_category(baseline_model, X_hn_base, adv_hard_neg["label"].values, adv_hard_neg["category"])

        # Improvement summary
        if len(adv_scam) > 0 and "adv_scam" in feature_sets:
            print(f"\n  {'Metric':<45} {'Baseline':>10} {'Retrained':>10} {'Delta':>10}")
            print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10}")
            print(f"  {'Adversarial scam detection (recall)':<45} "
                  f"{100*base_scam_recall:>9.1f}% {100*scam_recall:>9.1f}% "
                  f"{100*(scam_recall - base_scam_recall):>+9.1f}%")
            if len(adv_hard_neg) > 0:
                print(f"  {'Hard-negative FP rate':<45} "
                      f"{100*base_fp_rate:>9.1f}% {100*fp_rate:>9.1f}% "
                      f"{100*(fp_rate - base_fp_rate):>+9.1f}%")

    print("\n" + "=" * 70)
    print("Retraining complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
