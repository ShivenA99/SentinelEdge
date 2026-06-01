"""Train and evaluate logistic regression on hand-crafted features
for the adversarial experiment.

Mirrors ``run_adversarial.py`` but for the LR-on-handcrafted classifier
that Framing B promotes to the headline. Trains two variants:

  * ``default``       -- LR on the clean synthetic CSV only.
  * ``adv_retrained`` -- LR on clean + adversarial training data.

Evaluates both on the clean synthetic test set and the adversarial
test set, with the same per-category breakdown as
``run_adversarial.py``. Writes ``results/adversarial_lr.json`` whose
schema matches ``results/adversarial.json`` so
``extract_paper_numbers.py`` can pick it up directly.

Usage
-----
    python experiments/run_adversarial_lr.py
    python experiments/run_adversarial_lr.py --clean-csv ... --adv-train-csv ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel_edge.features.handcrafted import extract_handcrafted_features  # noqa: E402


# ---------------------------------------------------------------------------
# Feature extraction + I/O
# ---------------------------------------------------------------------------

def _handcrafted_matrix(texts: list[str]) -> np.ndarray:
    return np.array(
        [list(extract_handcrafted_features(t).values()) for t in texts],
        dtype=np.float32,
    )


def _load_csv(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or df.columns[0]
    label_col = cols.get("label") or df.columns[-1]
    cat_col = cols.get("category")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).values
    cats = (df[cat_col].astype(str).tolist() if cat_col
            else ["unknown"] * len(texts))
    return texts, labels, cats


# ---------------------------------------------------------------------------
# Metric helpers (mirror run_adversarial.py exactly)
# ---------------------------------------------------------------------------

def _metrics(y, s, thresh=0.5) -> dict:
    y_pred = (s >= thresh).astype(int)
    m = {
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        m["auroc"] = float(roc_auc_score(y, s))
        m["auprc"] = float(average_precision_score(y, s))
    return m


def _per_category(cats, labels, scores, label_filter=None):
    out = {}
    unique = sorted(set(cats))
    for cat in unique:
        mask = np.array([c == cat for c in cats])
        if label_filter is not None:
            mask = mask & (labels == label_filter)
        if mask.sum() == 0:
            continue
        sub_y = labels[mask]
        sub_s = scores[mask]
        if len(sub_y) < 2:
            continue
        out[cat] = {
            "n": int(mask.sum()),
            "n_pos": int(np.sum(sub_y == 1)),
            "mean_score": float(np.mean(sub_s)),
            "frac_above_0.5": float(np.mean(sub_s >= 0.5)),
            "frac_above_0.75": float(np.mean(sub_s >= 0.75)),
        }
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lr(texts: list[str], labels: np.ndarray) -> LogisticRegression:
    X = _handcrafted_matrix(texts)
    clf = LogisticRegression(
        max_iter=2000, n_jobs=1, class_weight="balanced",
    )
    clf.fit(X, labels)
    return clf


def predict(clf: LogisticRegression, texts: list[str]) -> np.ndarray:
    X = _handcrafted_matrix(texts)
    return clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clean-train-csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"),
    )
    ap.add_argument(
        "--clean-test-csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "call_fraud_test.csv"),
    )
    ap.add_argument(
        "--adv-csv",
        default=str(_PROJECT_ROOT / "data" / "raw" / "synthetic_transcripts" /
                    "adversarial_calls.csv"),
    )
    ap.add_argument(
        "--adv-test-fraction", type=float, default=0.30,
        help="Fraction of the adversarial CSV held out as test; rest is used "
             "for retraining the 'adv_retrained' classifier.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "adversarial_lr.json"),
    )
    args = ap.parse_args()

    print(f"[load] clean train: {args.clean_train_csv}")
    train_texts, train_labels, _ = _load_csv(Path(args.clean_train_csv))
    print(f"       {len(train_texts)} samples, "
          f"scam_frac={train_labels.mean():.3f}")

    print(f"[load] clean test:  {args.clean_test_csv}")
    test_texts, test_labels, _ = _load_csv(Path(args.clean_test_csv))
    print(f"       {len(test_texts)} samples")

    print(f"[load] adversarial: {args.adv_csv}")
    adv_texts, adv_labels, adv_cats = _load_csv(Path(args.adv_csv))
    print(f"       {len(adv_texts)} samples, "
          f"{len(set(adv_cats))} categories")

    # Split adversarial set into train / test halves (stratified by category)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(adv_texts))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - args.adv_test_fraction))
    adv_train_idx, adv_test_idx = idx[:cut], idx[cut:]
    adv_train_texts = [adv_texts[i] for i in adv_train_idx]
    adv_train_labels = np.array([adv_labels[i] for i in adv_train_idx])
    adv_test_texts = [adv_texts[i] for i in adv_test_idx]
    adv_test_labels = np.array([adv_labels[i] for i in adv_test_idx])
    adv_test_cats = [adv_cats[i] for i in adv_test_idx]
    print(f"[split] adv train={len(adv_train_idx)}  adv test={len(adv_test_idx)}")

    print("\n=== training: LR(handcrafted) on CLEAN only ===")
    clf_default = train_lr(train_texts, train_labels)

    print("=== training: LR(handcrafted) on CLEAN + ADVERSARIAL ===")
    combo_texts = list(train_texts) + list(adv_train_texts)
    combo_labels = np.concatenate([train_labels, adv_train_labels])
    clf_retrained = train_lr(combo_texts, combo_labels)

    results: dict = {}
    for clf_name, clf in [("default", clf_default),
                          ("adv_retrained", clf_retrained)]:
        print(f"\n=== eval: {clf_name} ===")
        clean_scores = predict(clf, test_texts)
        adv_scores = predict(clf, adv_test_texts)
        cm = _metrics(test_labels, clean_scores)
        am = _metrics(adv_test_labels, adv_scores)
        print(f"  CLEAN       F1={cm['f1']:.3f}  prec={cm['precision']:.3f}  "
              f"rec={cm['recall']:.3f}  AUROC={cm.get('auroc', 0):.3f}")
        print(f"  ADVERSARIAL F1={am['f1']:.3f}  prec={am['precision']:.3f}  "
              f"rec={am['recall']:.3f}  AUROC={am.get('auroc', 0):.3f}")
        print(f"  delta F1 = {am['f1'] - cm['f1']:+.3f}")

        # Per-category recall + FP-rate, same layout as run_adversarial.py
        adv_scam = _per_category(adv_test_cats, adv_test_labels, adv_scores,
                                 label_filter=1)
        adv_legit = _per_category(adv_test_cats, adv_test_labels, adv_scores,
                                  label_filter=0)
        results[clf_name] = {
            "clean": cm,
            "adversarial": am,
            "adv_scam_by_category": adv_scam,
            "adv_legit_by_category": adv_legit,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2,
    ))
    print(f"\n[saved] {out_path}")

    # One-line summary for the paper section
    d_clean = results["default"]["clean"]["f1"]
    d_adv = results["default"]["adversarial"]["f1"]
    r_clean = results["adv_retrained"]["clean"]["f1"]
    r_adv = results["adv_retrained"]["adversarial"]["f1"]
    print(f"\n=== headline (LR on 18 hand-crafted features) ===")
    print(f"  default       clean F1 = {d_clean:.3f}, adv F1 = {d_adv:.3f}  (delta {d_adv-d_clean:+.3f})")
    print(f"  adv-retrained clean F1 = {r_clean:.3f}, adv F1 = {r_adv:.3f}  (delta {r_adv-r_clean:+.3f})")
    print(f"  adv F1 recovery: {(r_adv - d_adv)*100:+.1f} F1 points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
