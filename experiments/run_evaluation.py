"""Honest evaluation of the trained SentinelEdge model on real call data.

This script does NOT train. It loads the existing
``models/call_fraud_xgb.json`` + ``models/tfidf_call_vectorizer.pkl``
and evaluates it three different ways:

  1. **Per-sentence**:  every sentence in the test set is scored
     independently and metrics are aggregated. This is what the
     classifier was trained on.

  2. **Per-call (mean)**: each call's sentences are scored and the
     average score is compared to threshold 0.5. This is the strawman
     full-transcript baseline.

  3. **Per-call (EMA streaming)**: sentences are processed in order,
     EMA accumulates, and the call is flagged ``scam`` if the EMA
     ever crosses 0.75 during the call. This matches the deployed
     streaming behaviour and is the headline number for the paper.

For each evaluation mode we report Accuracy / Precision / Recall / F1 /
AUROC / AUPRC, plus a per-source breakdown.

Usage
-----
    python experiments/run_evaluation.py
    python experiments/run_evaluation.py --sources repo_real teleantifraud_28k
    python experiments/run_evaluation.py --out results/eval_xgb.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord, load_repo_real, load_better30,
    load_wu2024, load_bothbosu, load_youtube_baiters, load_teleantifraud,
)
from sentinel_edge.classifier.score_accumulator import ScoreAccumulator  # noqa: E402
from sentinel_edge.classifier.xgb_classifier import FraudClassifier  # noqa: E402
from sentinel_edge.features.feature_pipeline import FeaturePipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

_LOADERS = {
    "repo_real": load_repo_real,
    "better30": load_better30,
    "wu2024_corpus": load_wu2024,
    "bothbosu": load_bothbosu,
    "youtube_baiters": load_youtube_baiters,
    "teleantifraud_28k": load_teleantifraud,
}


def load_sources(names: list[str]) -> list[CallRecord]:
    """Concatenate records from the requested sources."""
    records: list[CallRecord] = []
    for n in names:
        if n not in _LOADERS:
            raise ValueError(f"unknown source: {n}")
        loaded = _LOADERS[n]()
        if not loaded:
            print(f"  [warn] source {n!r} returned 0 records "
                  f"(data not downloaded?)", file=sys.stderr)
        records.extend(loaded)
    return records


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_metric(fn, y_true, y_pred, **kw):
    try:
        return float(fn(y_true, y_pred, **kw))
    except Exception:
        return float("nan")


def metrics_block(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Standard classification metrics dict for a single mode."""
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "threshold": threshold,
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
    }
    # AUROC / AUPRC need both classes present
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    # Confusion matrix counts (TN, FP, FN, TP)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    out["confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return out


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def score_call_sentences(
    record: CallRecord,
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
) -> list[float]:
    """Return per-sentence fraud probabilities for a single call."""
    out = []
    for sent in record.sentences:
        v = pipeline.extract(sent)
        out.append(classifier.predict_proba(v))
    return out


def evaluate_per_sentence(
    records: list[CallRecord],
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
) -> dict:
    """Treat each sentence as an independent example, labelled by call.

    This is the easiest setting and what XGBoost was trained on. Sentence
    label = call label (so legitimate calls contribute lots of negatives).
    """
    y_true, y_score = [], []
    for r in records:
        scores = score_call_sentences(r, pipeline, classifier)
        y_true.extend([r.label] * len(scores))
        y_score.extend(scores)
    return metrics_block(np.asarray(y_true), np.asarray(y_score))


def evaluate_per_call_mean(
    records: list[CallRecord],
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
) -> dict:
    """Score each call by mean of per-sentence probabilities."""
    y_true, y_score = [], []
    for r in records:
        scores = score_call_sentences(r, pipeline, classifier)
        if not scores:
            continue
        y_true.append(r.label)
        y_score.append(float(np.mean(scores)))
    return metrics_block(np.asarray(y_true), np.asarray(y_score))


def evaluate_per_call_streaming(
    records: list[CallRecord],
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
    ema_alpha: float = 0.3,
    ema_threshold: float = 0.75,
) -> dict:
    """Streaming EMA evaluation: call is flagged if EMA ever crosses threshold.

    The "score" reported for AUROC/AUPRC purposes is the *peak EMA* the
    call reached at any point during streaming.
    """
    y_true, y_score = [], []
    for r in records:
        acc = ScoreAccumulator(alpha=ema_alpha)
        peak = 0.0
        for sent in r.sentences:
            v = pipeline.extract(sent)
            p = classifier.predict_proba(v)
            ema = acc.update(p)
            if ema > peak:
                peak = ema
        y_true.append(r.label)
        y_score.append(peak)
    return metrics_block(
        np.asarray(y_true),
        np.asarray(y_score),
        threshold=ema_threshold,
    )


# ---------------------------------------------------------------------------
# Per-source breakdowns
# ---------------------------------------------------------------------------

def breakdown_by_source(
    records: list[CallRecord],
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
) -> dict:
    """Per-source streaming-mode metrics, for the per-dataset table."""
    by_src: dict[str, list[CallRecord]] = {}
    for r in records:
        by_src.setdefault(r.source, []).append(r)
    out = {}
    for src, recs in by_src.items():
        out[src] = evaluate_per_call_streaming(recs, pipeline, classifier)
        out[src]["n_calls"] = len(recs)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources", nargs="+",
        default=["repo_real", "teleantifraud_28k", "better30"],
        help="Datasets to include (any subset of the loader names).",
    )
    ap.add_argument(
        "--model", default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"),
    )
    ap.add_argument(
        "--tfidf",
        default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"),
    )
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)
    ap.add_argument("--out", default=str(_PROJECT_ROOT / "results" / "eval_xgb.json"))
    args = ap.parse_args()

    print(f"[load] sources={args.sources}")
    records = load_sources(args.sources)
    print(f"[load] {len(records)} total call records")
    if not records:
        print("[abort] no records to evaluate", file=sys.stderr)
        return 1

    print(f"[load] model = {args.model}")
    print(f"[load] tfidf = {args.tfidf}")
    pipeline = FeaturePipeline(args.tfidf)
    classifier = FraudClassifier(args.model)

    t0 = time.perf_counter()
    print("\n=== per-sentence ===")
    sent_metrics = evaluate_per_sentence(records, pipeline, classifier)
    print(json.dumps(sent_metrics, indent=2))

    print("\n=== per-call (mean of sentence scores) ===")
    mean_metrics = evaluate_per_call_mean(records, pipeline, classifier)
    print(json.dumps(mean_metrics, indent=2))

    print(f"\n=== per-call (streaming EMA, alpha={args.ema_alpha}, "
          f"threshold={args.ema_threshold}) ===")
    stream_metrics = evaluate_per_call_streaming(
        records, pipeline, classifier,
        ema_alpha=args.ema_alpha, ema_threshold=args.ema_threshold,
    )
    print(json.dumps(stream_metrics, indent=2))

    print("\n=== per-source streaming breakdown ===")
    by_src = breakdown_by_source(records, pipeline, classifier)
    for src, m in by_src.items():
        print(f"  {src}: n={m['n_calls']}  "
              f"F1={m['f1']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  AUROC={m['auroc']:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\n[done] {elapsed:.1f}s elapsed")

    out = {
        "config": {
            "sources": args.sources,
            "model": args.model,
            "tfidf": args.tfidf,
            "ema_alpha": args.ema_alpha,
            "ema_threshold": args.ema_threshold,
        },
        "per_sentence": sent_metrics,
        "per_call_mean": mean_metrics,
        "per_call_streaming": stream_metrics,
        "per_source_streaming": by_src,
        "elapsed_sec": elapsed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
