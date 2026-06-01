"""Latency and model-size benchmark for the on-device pipeline.

This script produces the headline "Why edge?" table for the paper:
how fast is XGBoost + TF-IDF, what's its disk and memory footprint,
and how does it compare to a logistic regression baseline?

(Larger neural baselines -- DistilBERT, Gemma -- live in
``run_baselines.py`` and require model downloads. We keep the size /
latency *measurements* on those models here as well, so a single
results JSON has the full comparison once everything is fetched.)

Measurements
------------
For each candidate model we measure (single CPU thread, single-sample
inference):

  * **inference_ms_p50 / p95 / p99** -- per-sentence latency
  * **throughput_sent_per_sec**      -- sustained throughput
  * **disk_size_mb**                 -- on-disk model size
  * **rss_increase_mb**              -- approximate peak memory delta

Methodology
-----------
Inputs are real sentences from ``repo_real``. We warm up for 50
samples, then time 500 samples to convergence. Each measurement is
repeated 3 times and the median across repetitions is reported. CPU
thread count is pinned to 1 to model the on-device single-core
scenario.

Usage
-----
    python experiments/run_latency.py
    python experiments/run_latency.py --reps 5 --n-warm 100 --n-time 1000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pin thread counts BEFORE importing anything that touches BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from experiments.dataset_loader import load_repo_real  # noqa: E402


# ---------------------------------------------------------------------------
# Memory probing
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Best-effort RSS in megabytes."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux returns KB, macOS returns bytes. Heuristic split:
        return usage / 1024.0 if usage > 1_000_000 else usage / 1024.0
    except Exception:
        return float("nan")


def _file_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Latency loop
# ---------------------------------------------------------------------------

def _time_predictions(
    score_fn,
    sentences: list[str],
    n_warm: int,
    n_time: int,
    reps: int,
) -> dict:
    """Run ``score_fn(sent)`` repeatedly and return latency stats.

    ``score_fn`` must take a single string and return a float.
    """
    # Cycle through sentences to avoid identical-input caching effects
    def gen(n):
        for i in range(n):
            yield sentences[i % len(sentences)]

    # Warmup
    for s in gen(n_warm):
        score_fn(s)

    rep_p50, rep_p95, rep_p99, rep_throughput = [], [], [], []
    for _ in range(reps):
        timings_ms = []
        t_start = time.perf_counter()
        for s in gen(n_time):
            t0 = time.perf_counter()
            score_fn(s)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        t_end = time.perf_counter()
        timings_ms.sort()
        rep_p50.append(timings_ms[len(timings_ms) // 2])
        rep_p95.append(timings_ms[int(len(timings_ms) * 0.95)])
        rep_p99.append(timings_ms[int(len(timings_ms) * 0.99)])
        rep_throughput.append(n_time / (t_end - t_start))

    return {
        "p50_ms": statistics.median(rep_p50),
        "p95_ms": statistics.median(rep_p95),
        "p99_ms": statistics.median(rep_p99),
        "throughput_sent_per_sec": statistics.median(rep_throughput),
        "reps": reps,
        "n_time": n_time,
        "n_warm": n_warm,
    }


# ---------------------------------------------------------------------------
# Candidate: XGBoost + TF-IDF (the deployed model)
# ---------------------------------------------------------------------------

def bench_xgb_tfidf(sentences, args) -> dict:
    from sentinel_edge.classifier.xgb_classifier import FraudClassifier
    from sentinel_edge.features.feature_pipeline import FeaturePipeline

    model_path = Path(args.model)
    tfidf_path = Path(args.tfidf)
    rss_before = _rss_mb()
    pipeline = FeaturePipeline(str(tfidf_path))
    classifier = FraudClassifier(str(model_path))
    rss_after = _rss_mb()

    def score(s: str) -> float:
        return classifier.predict_proba(pipeline.extract(s))

    latency = _time_predictions(score, sentences, args.n_warm, args.n_time, args.reps)
    return {
        "model": "xgb_tfidf_518d",
        "disk_size_mb": _file_mb(model_path) + _file_mb(tfidf_path),
        "rss_increase_mb": rss_after - rss_before,
        **latency,
    }


# ---------------------------------------------------------------------------
# Candidate: TF-IDF only + Logistic Regression baseline
# ---------------------------------------------------------------------------

def bench_logreg_tfidf(sentences, args) -> dict:
    """Quick LR-on-TF-IDF baseline trained on the synthetic data.

    This stands in as the "no handcrafted features, no trees" baseline
    so we can show how much the XGB + handcrafted features lift
    classification quality.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Pull synthetic training text for a quick fit.
    syn_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
    if not syn_csv.exists():
        return {"model": "logreg_tfidf_500d", "skipped": "no train CSV"}

    import pandas as pd
    df = pd.read_csv(syn_csv)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = (
        "label" if "label" in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else df.columns[-1])
    )

    rss_before = _rss_mb()
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vec.fit_transform(df[text_col].astype(str).tolist())
    y = df[label_col].astype(int).values
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X, y)
    rss_after = _rss_mb()

    def score(s: str) -> float:
        v = vec.transform([s])
        return float(clf.predict_proba(v)[0, 1])

    latency = _time_predictions(score, sentences, args.n_warm, args.n_time, args.reps)
    # Rough disk size: pickle both pieces in-memory and measure
    import pickle, io
    buf = io.BytesIO()
    pickle.dump((vec, clf), buf)
    return {
        "model": "logreg_tfidf_500d",
        "disk_size_mb": len(buf.getvalue()) / (1024 * 1024),
        "rss_increase_mb": rss_after - rss_before,
        **latency,
    }


# ---------------------------------------------------------------------------
# Candidate: handcrafted-only + Logistic Regression (interpretability floor)
# ---------------------------------------------------------------------------

def bench_logreg_handcrafted(sentences, args) -> dict:
    """LR on only the 18 handcrafted features."""
    from sentinel_edge.features.handcrafted import extract_handcrafted_features
    from sklearn.linear_model import LogisticRegression

    syn_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
    if not syn_csv.exists():
        return {"model": "logreg_handcrafted_18d", "skipped": "no train CSV"}

    import pandas as pd
    df = pd.read_csv(syn_csv)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = (
        "label" if "label" in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else df.columns[-1])
    )
    X = np.array([
        list(extract_handcrafted_features(t).values())
        for t in df[text_col].astype(str)
    ], dtype=np.float32)
    y = df[label_col].astype(int).values

    rss_before = _rss_mb()
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X, y)
    rss_after = _rss_mb()

    def score(s: str) -> float:
        feats = np.array(
            list(extract_handcrafted_features(s).values()), dtype=np.float32
        ).reshape(1, -1)
        return float(clf.predict_proba(feats)[0, 1])

    latency = _time_predictions(score, sentences, args.n_warm, args.n_time, args.reps)
    import pickle, io
    buf = io.BytesIO()
    pickle.dump(clf, buf)
    return {
        "model": "logreg_handcrafted_18d",
        "disk_size_mb": len(buf.getvalue()) / (1024 * 1024),
        "rss_increase_mb": rss_after - rss_before,
        **latency,
    }


# ---------------------------------------------------------------------------
# Optional: DistilBERT baseline (scaffolded, gated on availability)
# ---------------------------------------------------------------------------

def bench_distilbert(sentences, args) -> dict:
    """DistilBERT-base-uncased fine-tuned head (scaffold).

    This requires `transformers` and `torch` to be installed and a
    DistilBERT checkpoint locally. We do NOT fine-tune here -- we use
    the model's *inference* characteristics with a random classifier
    head, which is what we'd measure for true latency anyway.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )
    except Exception as e:
        return {"model": "distilbert_finetuned", "skipped": f"deps missing: {e}"}

    name = args.distilbert_model
    try:
        rss_before = _rss_mb()
        tok = AutoTokenizer.from_pretrained(name, local_files_only=args.local_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=2, local_files_only=args.local_only,
        )
        model.eval()
        torch.set_num_threads(1)
        rss_after = _rss_mb()
    except Exception as e:
        return {"model": "distilbert_finetuned", "skipped": f"load failed: {e}"}

    def score(s: str) -> float:
        with torch.no_grad():
            inputs = tok(
                s, return_tensors="pt", truncation=True, max_length=128,
            )
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            return float(prob)

    latency = _time_predictions(
        score, sentences, args.n_warm, max(50, args.n_time // 5), args.reps,
    )
    # Approximate disk size from model params
    nparams = sum(p.numel() for p in model.parameters())
    return {
        "model": "distilbert_finetuned",
        "n_params_M": nparams / 1e6,
        "disk_size_mb": nparams * 4 / (1024 * 1024),
        "rss_increase_mb": rss_after - rss_before,
        **latency,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CANDIDATES = {
    "xgb_tfidf": bench_xgb_tfidf,
    "logreg_tfidf": bench_logreg_tfidf,
    "logreg_handcrafted": bench_logreg_handcrafted,
    "distilbert": bench_distilbert,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates", nargs="+",
        default=["xgb_tfidf", "logreg_tfidf", "logreg_handcrafted"],
        choices=list(CANDIDATES.keys()),
    )
    ap.add_argument("--model", default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"))
    ap.add_argument(
        "--tfidf",
        default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"),
    )
    ap.add_argument("--distilbert-model", default="distilbert-base-uncased")
    ap.add_argument("--local-only", action="store_true",
                    help="If set, don't download from HF Hub.")
    ap.add_argument("--n-warm", type=int, default=50)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out", default=str(_PROJECT_ROOT / "results" / "latency.json"),
    )
    args = ap.parse_args()

    print(f"[load] real call sentences")
    records = load_repo_real()
    sentences = []
    for r in records:
        sentences.extend(r.sentences)
    print(f"[load] {len(sentences)} unique sentences for timing")

    results = []
    for name in args.candidates:
        print(f"\n[bench] {name}")
        gc.collect()
        try:
            r = CANDIDATES[name](sentences, args)
            print(json.dumps(r, indent=2))
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"model": name, "error": str(e)})

    out = {"config": vars(args), "results": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    print("\n=== Summary ===")
    print(f"{'model':<28}{'p50 (ms)':>10}{'p95 (ms)':>10}{'throughput':>14}{'size (MB)':>12}")
    print("-" * 74)
    for r in results:
        if "p50_ms" in r:
            print(f"{r['model']:<28}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}"
                  f"{r['throughput_sent_per_sec']:>14.0f}{r['disk_size_mb']:>12.2f}")
        else:
            print(f"{r['model']:<28}  (skipped: {r.get('skipped') or r.get('error')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
