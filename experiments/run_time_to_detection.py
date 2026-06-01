"""Time-to-detection (TTD) experiment.

This is the most distinctive evaluation in the paper. Every scam call in
the test set is processed sentence-by-sentence; we record the index of
the *first* sentence at which the EMA score crosses the alert threshold.

Then we compare three classifiers under three regimes each:

  * "streaming"     -- per-sentence + EMA, alerts mid-call
  * "full"          -- single pass over the concatenated transcript
                       (this requires the call to finish before alerting)
  * "first-N"       -- truncated transcripts of the first N sentences,
                       N in {1,3,5,10,20}; shows degradation curve

Outputs a JSON results file with:
  - per-call: ttd_sentence_idx, ttd_seconds (assuming 4s/sentence default),
              flagged, peak_ema, final_score
  - aggregate: median / p25 / p75 / p90 TTD over true scams
  - CDF arrays for plotting

Usage
-----
    python experiments/run_time_to_detection.py
    python experiments/run_time_to_detection.py --seconds-per-sentence 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord, load_repo_real, load_better30,
    load_wu2024, load_youtube_baiters, load_teleantifraud,
)
from sentinel_edge.classifier.score_accumulator import ScoreAccumulator  # noqa: E402
from sentinel_edge.classifier.xgb_classifier import FraudClassifier  # noqa: E402
from sentinel_edge.features.feature_pipeline import FeaturePipeline  # noqa: E402


_LOADERS = {
    "repo_real": load_repo_real,
    "better30": load_better30,
    "wu2024_corpus": load_wu2024,
    "youtube_baiters": load_youtube_baiters,
    "teleantifraud_28k": load_teleantifraud,
}


def load_sources(names: list[str]) -> list[CallRecord]:
    out: list[CallRecord] = []
    for n in names:
        out.extend(_LOADERS[n]())
    return out


# ---------------------------------------------------------------------------
# Streaming evaluation: when does the EMA first cross threshold?
# ---------------------------------------------------------------------------

def trace_call_streaming(
    record: CallRecord,
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
    ema_alpha: float = 0.3,
    ema_threshold: float = 0.75,
    seconds_per_sentence: float = 4.0,
) -> dict:
    """Single-call streaming trace.

    Returns a dict with:
      - raw_scores : per-sentence fraud probabilities
      - ema_scores : EMA trajectory
      - ttd_idx    : index of first sentence with EMA >= threshold,
                     or None if never crossed
      - ttd_sec    : equivalent seconds (idx * seconds_per_sentence)
                     plus one full sentence latency
    """
    acc = ScoreAccumulator(alpha=ema_alpha)
    raw, ema = [], []
    ttd_idx: int | None = None
    for i, sent in enumerate(record.sentences):
        v = pipeline.extract(sent)
        p = classifier.predict_proba(v)
        e = acc.update(p)
        raw.append(float(p))
        ema.append(float(e))
        if ttd_idx is None and e >= ema_threshold:
            ttd_idx = i
    peak = max(ema) if ema else 0.0
    final = ema[-1] if ema else 0.0
    ttd_sec = None if ttd_idx is None else (ttd_idx + 1) * seconds_per_sentence
    return {
        "call_id": record.call_id,
        "label": record.label,
        "source": record.source,
        "category": record.category,
        "n_sentences": len(record.sentences),
        "raw_scores": raw,
        "ema_scores": ema,
        "ttd_idx": ttd_idx,
        "ttd_sec": ttd_sec,
        "peak_ema": peak,
        "final_ema": final,
    }


# ---------------------------------------------------------------------------
# Full-transcript baseline: classify the concatenated call as one document
# ---------------------------------------------------------------------------

def score_full_transcript(
    record: CallRecord,
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
) -> float:
    """Score the whole call as a single document.

    This is the "full-transcript TF-IDF" baseline the ARCHITECTURE doc
    contrasts against. It also represents what a system that waits for
    call completion would do.
    """
    full_text = " ".join(record.sentences)
    if not full_text.strip():
        return 0.0
    v = pipeline.extract(full_text)
    return float(classifier.predict_proba(v))


# ---------------------------------------------------------------------------
# First-N truncated streaming
# ---------------------------------------------------------------------------

def evaluate_first_n(
    records: list[CallRecord],
    pipeline: FeaturePipeline,
    classifier: FraudClassifier,
    ns: list[int],
    ema_alpha: float = 0.3,
    ema_threshold: float = 0.75,
) -> dict:
    """For each N, classify based only on the first N sentences (streaming).

    Returns per-N {precision, recall, F1, accuracy}.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    out = {}
    for n in ns:
        y_true, y_pred = [], []
        for r in records:
            acc = ScoreAccumulator(alpha=ema_alpha)
            flagged = False
            for sent in r.sentences[:n]:
                v = pipeline.extract(sent)
                p = classifier.predict_proba(v)
                if acc.update(p) >= ema_threshold:
                    flagged = True
                    break
            y_true.append(r.label)
            y_pred.append(int(flagged))
        y_true_a = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred)
        out[str(n)] = {
            "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
            "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
            "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
            "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_ttd(traces: list[dict]) -> dict:
    """Aggregate time-to-detection statistics over true scam calls."""
    scam_traces = [t for t in traces if t["label"] == 1]
    detected = [t for t in scam_traces if t["ttd_idx"] is not None]
    missed = len(scam_traces) - len(detected)

    if not detected:
        return {
            "n_scam_calls": len(scam_traces),
            "n_detected": 0,
            "n_missed": len(scam_traces),
            "ttd_idx_median": None,
            "ttd_sec_median": None,
        }

    ttd_idx = np.array([t["ttd_idx"] for t in detected])
    ttd_sec = np.array([t["ttd_sec"] for t in detected])
    return {
        "n_scam_calls": len(scam_traces),
        "n_detected": len(detected),
        "n_missed": missed,
        "recall_streaming": len(detected) / len(scam_traces),
        "ttd_idx_p25": float(np.percentile(ttd_idx, 25)),
        "ttd_idx_median": float(np.median(ttd_idx)),
        "ttd_idx_mean": float(np.mean(ttd_idx)),
        "ttd_idx_p75": float(np.percentile(ttd_idx, 75)),
        "ttd_idx_p90": float(np.percentile(ttd_idx, 90)),
        "ttd_sec_p25": float(np.percentile(ttd_sec, 25)),
        "ttd_sec_median": float(np.median(ttd_sec)),
        "ttd_sec_mean": float(np.mean(ttd_sec)),
        "ttd_sec_p75": float(np.percentile(ttd_sec, 75)),
        "ttd_sec_p90": float(np.percentile(ttd_sec, 90)),
        # Empirical CDF: sorted ttd_idx values for plotting
        "ttd_idx_sorted": [int(x) for x in sorted(ttd_idx.tolist())],
    }


def aggregate_full_vs_streaming(
    traces: list[dict],
    full_scores: list[float],
    full_threshold: float = 0.5,
) -> dict:
    """Compare streaming-flag vs full-transcript-flag at call level."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    y_true = np.array([t["label"] for t in traces])

    # Streaming: flagged iff EMA crossed threshold at any point
    streaming_pred = np.array(
        [1 if t["ttd_idx"] is not None else 0 for t in traces]
    )
    peak_emas = np.array([t["peak_ema"] for t in traces])
    full = np.asarray(full_scores)
    full_pred = (full >= full_threshold).astype(int)

    def block(y_true, y_pred, score=None):
        m = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if score is not None and len(np.unique(y_true)) > 1:
            m["auroc"] = float(roc_auc_score(y_true, score))
        return m

    return {
        "streaming_ema": block(y_true, streaming_pred, peak_emas),
        "full_transcript": block(y_true, full_pred, full),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources", nargs="+",
        default=["repo_real", "teleantifraud_28k", "better30"],
    )
    ap.add_argument("--model", default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"))
    ap.add_argument(
        "--tfidf",
        default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"),
    )
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)
    ap.add_argument(
        "--seconds-per-sentence", type=float, default=4.0,
        help="Wall-clock seconds attributed to each sentence (Whisper window + utterance).",
    )
    ap.add_argument("--first-ns", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "ttd.json"),
    )
    args = ap.parse_args()

    records = load_sources(args.sources)
    print(f"[load] {len(records)} call records from {args.sources}")
    if not records:
        return 1

    pipeline = FeaturePipeline(args.tfidf)
    classifier = FraudClassifier(args.model)

    # Streaming traces
    traces = [
        trace_call_streaming(
            r, pipeline, classifier,
            ema_alpha=args.ema_alpha,
            ema_threshold=args.ema_threshold,
            seconds_per_sentence=args.seconds_per_sentence,
        )
        for r in records
    ]
    # Full-transcript scores
    full_scores = [score_full_transcript(r, pipeline, classifier) for r in records]

    ttd_agg = aggregate_ttd(traces)
    head_to_head = aggregate_full_vs_streaming(traces, full_scores)
    first_n = evaluate_first_n(
        records, pipeline, classifier,
        ns=args.first_ns,
        ema_alpha=args.ema_alpha,
        ema_threshold=args.ema_threshold,
    )

    print("\n=== Time-to-Detection aggregate (true scam calls) ===")
    print(json.dumps({k: v for k, v in ttd_agg.items() if k != "ttd_idx_sorted"}, indent=2))

    print("\n=== Streaming EMA vs Full transcript (per-call classification) ===")
    print(json.dumps(head_to_head, indent=2))

    print("\n=== First-N sentences (truncated streaming) ===")
    for n, m in first_n.items():
        print(f"  N={n:>3}  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  F1={m['f1']:.3f}")

    out = {
        "config": vars(args),
        "ttd_aggregate": ttd_agg,
        "streaming_vs_full": head_to_head,
        "first_n": first_n,
        "per_call_traces": [
            {k: v for k, v in t.items() if k not in {"raw_scores", "ema_scores"}}
            for t in traces
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
