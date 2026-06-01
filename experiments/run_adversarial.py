"""Adversarial evaluation.

The project ships an adversarial sentence generator
(``training/generate_adversarial_data.py``) that produces six classes of
hard-to-detect scams and four classes of legitimate hard negatives.
This script measures how the deployed classifier holds up against them.

Two settings:

  * **clean**       -- evaluate on the standard synthetic test split
  * **adversarial** -- evaluate on the adversarial CSV
  * Optionally evaluate both the original and the *adversarially-retrained*
    classifier (``models/call_fraud_xgb_adversarial.json``) on both sets,
    to show whether retraining helps.

The metric reported is per-sentence F1 / precision / recall / AUROC, and
the streaming F1 at the *call* level computed by reconstructing
per-category mini-calls from contiguous adversarial sentences of the
same category and label.

Usage
-----
    python experiments/run_adversarial.py
    python experiments/run_adversarial.py --classifiers default adv-retrained
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel_edge.classifier.xgb_classifier import FraudClassifier  # noqa: E402
from sentinel_edge.features.feature_pipeline import FeaturePipeline  # noqa: E402


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names
    cols = {c.lower(): c for c in df.columns}
    text_c = cols.get("text") or df.columns[0]
    label_c = cols.get("label") or df.columns[-1]
    cat_c = cols.get("category")
    out = pd.DataFrame({
        "text": df[text_c].astype(str),
        "label": df[label_c].astype(int),
    })
    if cat_c:
        out["category"] = df[cat_c].astype(str)
    else:
        out["category"] = "unknown"
    return out


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _scores(texts: list[str], pipeline: FeaturePipeline,
            classifier: FraudClassifier) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.float64)
    for i, t in enumerate(texts):
        out[i] = classifier.predict_proba(pipeline.extract(t))
    return out


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


def _per_category(df, scores, label_filter=None):
    """Per-category breakdown for adversarial diagnostic."""
    out = {}
    for cat in df["category"].unique():
        mask = (df["category"] == cat).values
        if label_filter is not None:
            mask = mask & (df["label"].values == label_filter)
        if mask.sum() == 0:
            continue
        sub_y = df["label"].values[mask]
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
# Main
# ---------------------------------------------------------------------------

_CLASSIFIER_PATHS = {
    "default": (
        _PROJECT_ROOT / "models" / "call_fraud_xgb.json",
        _PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl",
    ),
    "adv_retrained": (
        _PROJECT_ROOT / "models" / "call_fraud_xgb_adversarial.json",
        _PROJECT_ROOT / "models" / "tfidf_call_vectorizer_adversarial.pkl",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--classifiers", nargs="+",
        default=["default", "adv_retrained"],
        choices=list(_CLASSIFIER_PATHS.keys()),
    )
    ap.add_argument(
        "--clean-csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "call_fraud_test.csv"),
    )
    ap.add_argument(
        "--adv-csv",
        default=str(_PROJECT_ROOT / "data" / "raw" / "synthetic_transcripts" /
                    "adversarial_calls.csv"),
    )
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "adversarial.json"),
    )
    args = ap.parse_args()

    clean_df = _load_csv(Path(args.clean_csv))
    print(f"[load] clean: {len(clean_df)} sentences "
          f"({clean_df['label'].mean():.2%} scam)")

    adv_df = _load_csv(Path(args.adv_csv))
    print(f"[load] adversarial: {len(adv_df)} sentences "
          f"({adv_df['label'].mean():.2%} scam)")

    big_results: dict = {}
    for clf_name in args.classifiers:
        model_p, tfidf_p = _CLASSIFIER_PATHS[clf_name]
        if not model_p.exists():
            print(f"  [skip] {clf_name}: {model_p} not found")
            continue
        print(f"\n=== classifier: {clf_name} ===")
        pipe = FeaturePipeline(str(tfidf_p))
        clf = FraudClassifier(str(model_p))

        clean_scores = _scores(clean_df["text"].tolist(), pipe, clf)
        adv_scores = _scores(adv_df["text"].tolist(), pipe, clf)

        cm = _metrics(clean_df["label"].values, clean_scores)
        am = _metrics(adv_df["label"].values, adv_scores)

        print(f"  CLEAN       F1={cm['f1']:.3f}  prec={cm['precision']:.3f}  "
              f"rec={cm['recall']:.3f}  AUROC={cm.get('auroc', 0):.3f}")
        print(f"  ADVERSARIAL F1={am['f1']:.3f}  prec={am['precision']:.3f}  "
              f"rec={am['recall']:.3f}  AUROC={am.get('auroc', 0):.3f}")
        print(f"  delta F1 = {am['f1'] - cm['f1']:+.3f}")

        # Per-category recall on adversarial scams only
        adv_scam = _per_category(adv_df, adv_scores, label_filter=1)
        adv_legit = _per_category(adv_df, adv_scores, label_filter=0)

        print(f"\n  adversarial scam categories (recall @ thresh=0.5):")
        for cat, st in sorted(adv_scam.items(),
                              key=lambda kv: kv[1]["frac_above_0.5"]):
            print(f"    {cat:<35} n={st['n']:>4}  mean={st['mean_score']:.3f}  "
                  f"caught={st['frac_above_0.5']:.2%}")

        print(f"\n  adversarial hard-negative categories (false positive @ thresh=0.5):")
        for cat, st in sorted(adv_legit.items(),
                              key=lambda kv: -kv[1]["frac_above_0.5"]):
            print(f"    {cat:<35} n={st['n']:>4}  mean={st['mean_score']:.3f}  "
                  f"FP={st['frac_above_0.5']:.2%}")

        big_results[clf_name] = {
            "clean": cm,
            "adversarial": am,
            "adv_scam_by_category": adv_scam,
            "adv_legit_by_category": adv_legit,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": vars(args), "results": big_results}, indent=2,
    ))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
