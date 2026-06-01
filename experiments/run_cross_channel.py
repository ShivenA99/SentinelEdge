"""Cross-channel transfer experiment.

The trained classifier (XGBoost on 18 handcrafted + 500 TF-IDF
features) was trained on synthetic call transcripts. This script
measures how well it transfers, zero-shot, to two adjacent fraud
channels that already exist in the repo:

  * SMS Spam Collection (Almeida et al. 2011) -- 5,574 SMS messages,
    13% spam. Each message is one "sentence" for our purposes.

  * Combined phishing URLs (PhishTank etc.) -- 4,001 URLs labelled
    phishing / benign. Each URL is treated as a single sentence; the
    classifier's URL-related handcrafted features (has_url,
    has_shortened_url) should carry most of the signal.

For each cross-channel target we report:

  * The classifier's F1, precision, recall, AUROC, AUPRC
  * The fraction of negative examples flagged (false positive rate)
  * A score histogram (so the paper can show distribution shift)
  * For SMS: per-sentence metrics so we can compare to in-domain
    per-sentence call metrics

We also report an "in-domain reference" -- evaluating the same model
on the synthetic call test split -- so the cross-channel numbers
have a baseline to drop from.

The paper's framing: lightweight handcrafted+TF-IDF features
transfer cleanly to adjacent fraud channels with degradation X%,
while heavier fine-tuned models would need per-channel adaptation.

Usage
-----
    python experiments/run_cross_channel.py
    python experiments/run_cross_channel.py --channels sms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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
# Channel loaders
# ---------------------------------------------------------------------------

def load_sms() -> tuple[list[str], np.ndarray]:
    """Load SMS Spam Collection: returns (texts, labels)."""
    path = _PROJECT_ROOT / "data" / "real" / "sms_spam" / "SMSSpamCollection.tsv"
    if not path.exists():
        return [], np.array([])
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition("\t")
            if not text:
                continue
            texts.append(text)
            labels.append(1 if label.strip().lower() == "spam" else 0)
    return texts, np.asarray(labels)


def load_urls() -> tuple[list[str], np.ndarray]:
    """Load combined phishing URLs: returns (urls, labels)."""
    path = _PROJECT_ROOT / "data" / "real" / "phishing_urls" / "combined_real_urls.csv"
    if not path.exists():
        return [], np.array([])
    import pandas as pd
    df = pd.read_csv(path)
    text_col = "url" if "url" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    texts = df[text_col].astype(str).tolist()

    def _norm_label(v) -> int:
        s = str(v).strip().lower()
        if s in {"1", "phishing", "phish", "malicious", "fraud", "true", "yes"}:
            return 1
        if s in {"0", "benign", "legitimate", "legit", "false", "no"}:
            return 0
        try:
            return int(float(s))
        except ValueError:
            return 0

    labels = np.array([_norm_label(v) for v in df[label_col]], dtype=int)
    return texts, labels


def load_in_domain_calls() -> tuple[list[str], np.ndarray]:
    """In-domain reference: synthetic call test CSV, sentence-level."""
    path = _PROJECT_ROOT / "data" / "processed" / "call_fraud_test.csv"
    if not path.exists():
        return [], np.array([])
    import pandas as pd
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    return df[text_col].astype(str).tolist(), df[label_col].astype(int).values


# ---------------------------------------------------------------------------
# Scoring + metrics
# ---------------------------------------------------------------------------

def score_batch(texts: list[str], pipeline: FeaturePipeline,
                classifier: FraudClassifier) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.float64)
    for i, t in enumerate(texts):
        out[i] = classifier.predict_proba(pipeline.extract(t))
    return out


def metrics_block(y, s, threshold: float = 0.5) -> dict:
    y_pred = (s >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "score_mean_pos": float(np.mean(s[y == 1])) if (y == 1).any() else float("nan"),
        "score_mean_neg": float(np.mean(s[y == 0])) if (y == 0).any() else float("nan"),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    return out


def score_histogram(s: np.ndarray, n_bins: int = 20) -> list[int]:
    """Score histogram for plotting / sanity checking."""
    counts, _ = np.histogram(s, bins=np.linspace(0, 1, n_bins + 1))
    return counts.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CHANNELS = {
    "calls":  ("In-domain reference (synthetic call test)", load_in_domain_calls),
    "sms":    ("SMS Spam Collection",                       load_sms),
    "urls":   ("Phishing URLs",                              load_urls),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--channels", nargs="+", default=list(_CHANNELS.keys()),
        choices=list(_CHANNELS.keys()),
    )
    ap.add_argument(
        "--model", default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"),
    )
    ap.add_argument(
        "--tfidf",
        default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"),
    )
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "cross_channel.json"),
    )
    args = ap.parse_args()

    pipeline = FeaturePipeline(args.tfidf)
    classifier = FraudClassifier(args.model)

    all_results: dict = {}
    for ch in args.channels:
        label, loader = _CHANNELS[ch]
        texts, y = loader()
        if len(texts) == 0:
            print(f"\n[skip] {ch}: no data found")
            continue

        print(f"\n=== {ch}: {label} ===")
        print(f"  n={len(texts)}  n_pos={int(np.sum(y == 1))}  "
              f"n_neg={int(np.sum(y == 0))}")
        t0 = time.perf_counter()
        scores = score_batch(texts, pipeline, classifier)
        dt = time.perf_counter() - t0
        print(f"  scored in {dt:.1f}s "
              f"({len(texts) / dt:.0f} samples/sec)")

        m = metrics_block(y, scores)
        print(f"  F1={m['f1']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}  "
              f"AUPRC={m.get('auprc', float('nan')):.3f}")
        print(f"  mean score: pos={m['score_mean_pos']:.3f}  "
              f"neg={m['score_mean_neg']:.3f}")

        all_results[ch] = {
            "label": label,
            **m,
            "score_hist": score_histogram(scores),
            "scoring_sec": dt,
        }

    # Degradation table: F1 drop from in-domain to each cross-channel
    if "calls" in all_results:
        ref_f1 = all_results["calls"]["f1"]
        print(f"\n=== Cross-channel degradation (vs in-domain F1={ref_f1:.3f}) ===")
        for ch, m in all_results.items():
            if ch == "calls":
                continue
            delta = m["f1"] - ref_f1
            print(f"  {ch:>8}: F1={m['f1']:.3f}  ({delta:+.3f} delta)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": vars(args), "results": all_results}, indent=2,
    ))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
