"""Baseline classifiers for SentinelEdge comparison.

For each baseline we:
  1. Train on the synthetic training CSV (the only training data the
     project has plenty of)
  2. Evaluate on real call data (streaming EMA, per-call, per-sentence)
  3. Save model artefacts so ``run_latency.py`` can also time them

Baselines covered (all CPU-only):
  * ``handcrafted_lr``  -- LogReg on 18 handcrafted features
  * ``tfidf_lr``        -- LogReg on 500-dim TF-IDF
  * ``tfidf_handcrafted_lr`` -- LogReg on the same 518-dim feature space as XGB
  * ``handcrafted_svm`` -- Linear SVM on 18 handcrafted features
  * ``trained_xgb``     -- the deployed model (no training, just eval)

DistilBERT and Gemma baselines are scaffolded in ``run_baselines_neural.py``;
running them requires downloading ~250MB-2GB of weights and is gated by
the ``--with-neural`` flag.

Usage
-----
    python experiments/run_baselines.py
    python experiments/run_baselines.py --baselines handcrafted_lr trained_xgb
    python experiments/run_baselines.py --eval-sources repo_real teleantifraud_28k
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord, load_repo_real, load_teleantifraud, load_better30,
)
from experiments.run_evaluation import (  # noqa: E402
    evaluate_per_sentence as _eval_per_sentence_with,
    evaluate_per_call_mean as _eval_per_call_mean_with,
    evaluate_per_call_streaming as _eval_per_call_streaming_with,
    metrics_block,
)
from sentinel_edge.features.handcrafted import extract_handcrafted_features  # noqa: E402


_LOADERS = {
    "repo_real": load_repo_real,
    "teleantifraud_28k": load_teleantifraud,
    "better30": load_better30,
}


# ---------------------------------------------------------------------------
# Synthetic training set
# ---------------------------------------------------------------------------

def _load_training_data() -> tuple[list[str], np.ndarray]:
    """Load the synthetic training CSV as (texts, labels)."""
    import pandas as pd
    train_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
    df = pd.read_csv(train_csv)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = (
        "label" if "label" in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else df.columns[-1])
    )
    return df[text_col].astype(str).tolist(), df[label_col].astype(int).values


def _handcrafted_matrix(texts: list[str]) -> np.ndarray:
    return np.array(
        [list(extract_handcrafted_features(t).values()) for t in texts],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# A common Classifier protocol
# ---------------------------------------------------------------------------

class BaselineScorer:
    """Wraps any sklearn-style model so it has a uniform ``score(text)`` API.

    Compatible with the same evaluation helpers we use for XGB.
    """

    def __init__(self, name: str, score_fn):
        self.name = name
        self._score = score_fn

    def predict_proba(self, text_or_vec) -> float:
        # The evaluation helpers call ``classifier.predict_proba(v)`` after
        # feature extraction. To keep one API we instead bypass them: this
        # scorer is wrapped in a ``ScorerPipeline`` below that exposes the
        # full text path.
        raise NotImplementedError("Use ScorerPipeline.score(text) instead.")


class ScorerPipeline:
    """Pipeline + classifier wrapper sharing the API expected by run_evaluation.

    Specifically it exposes ``.extract(text) -> ndarray`` and
    ``.predict_proba(ndarray) -> float``. We just route both through the
    underlying ``score_fn(text) -> float``.
    """

    def __init__(self, score_fn):
        self._score = score_fn

    def extract(self, text: str):
        # Pass through; we treat text itself as the "vector"
        return text

    def predict_proba(self, text) -> float:
        return float(self._score(text))


def _eval_record_list(records, score_fn,
                      ema_alpha=0.3, ema_threshold=0.75) -> dict:
    """Evaluate a baseline (defined by score_fn(text)) using same modes."""
    sp = ScorerPipeline(score_fn)
    return {
        "per_sentence": _eval_per_sentence_with(records, sp, sp),
        "per_call_mean": _eval_per_call_mean_with(records, sp, sp),
        "per_call_streaming": _eval_per_call_streaming_with(
            records, sp, sp,
            ema_alpha=ema_alpha, ema_threshold=ema_threshold,
        ),
    }


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def train_handcrafted_lr(texts, y):
    from sklearn.linear_model import LogisticRegression
    X = _handcrafted_matrix(texts)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")
    clf.fit(X, y)
    def score(text: str) -> float:
        v = _handcrafted_matrix([text])
        return float(clf.predict_proba(v)[0, 1])
    return score, clf


def train_tfidf_lr(texts, y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")
    clf.fit(X, y)
    def score(text: str) -> float:
        v = vec.transform([text])
        return float(clf.predict_proba(v)[0, 1])
    return score, (vec, clf)


def train_combined_lr(texts, y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import scipy.sparse as sp
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    Xt = vec.fit_transform(texts)
    Xh = _handcrafted_matrix(texts)
    X = sp.hstack([sp.csr_matrix(Xh), Xt]).tocsr()
    clf = LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")
    clf.fit(X, y)
    def score(text: str) -> float:
        Xt1 = vec.transform([text])
        Xh1 = sp.csr_matrix(_handcrafted_matrix([text]))
        v = sp.hstack([Xh1, Xt1]).tocsr()
        return float(clf.predict_proba(v)[0, 1])
    return score, (vec, clf)


def train_handcrafted_svm(texts, y):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    X = _handcrafted_matrix(texts)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    base = LinearSVC(max_iter=5000)
    # Wrap with calibration so we get probabilities for the streaming pipeline.
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(Xs, y)
    def score(text: str) -> float:
        v = scaler.transform(_handcrafted_matrix([text]))
        return float(clf.predict_proba(v)[0, 1])
    return score, (scaler, clf)


def make_trained_xgb_scorer():
    from sentinel_edge.classifier.xgb_classifier import FraudClassifier
    from sentinel_edge.features.feature_pipeline import FeaturePipeline
    pipe = FeaturePipeline(
        str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl")
    )
    clf = FraudClassifier(
        str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json")
    )
    def score(text: str) -> float:
        return float(clf.predict_proba(pipe.extract(text)))
    return score, (pipe, clf)


BASELINES = {
    "handcrafted_lr":         lambda t, y: train_handcrafted_lr(t, y),
    "tfidf_lr":               lambda t, y: train_tfidf_lr(t, y),
    "tfidf_handcrafted_lr":   lambda t, y: train_combined_lr(t, y),
    "handcrafted_svm":        lambda t, y: train_handcrafted_svm(t, y),
    "trained_xgb":            None,  # special-cased -- no training
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baselines", nargs="+",
        default=list(BASELINES.keys()),
        choices=list(BASELINES.keys()),
    )
    ap.add_argument(
        "--eval-sources", nargs="+",
        default=["repo_real"],
    )
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "baselines.json"),
    )
    args = ap.parse_args()

    print("[load] training data")
    texts, y = _load_training_data()
    print(f"[load]  {len(texts)} samples, scam fraction {y.mean():.3f}")

    print(f"[load] eval sources={args.eval_sources}")
    eval_records: list[CallRecord] = []
    for s in args.eval_sources:
        eval_records.extend(_LOADERS[s]())
    print(f"[load]  {len(eval_records)} eval call records")

    results = {}
    for name in args.baselines:
        print(f"\n=== baseline: {name} ===")
        t0 = time.perf_counter()
        if name == "trained_xgb":
            score_fn, _artefact = make_trained_xgb_scorer()
        else:
            score_fn, _artefact = BASELINES[name](texts, y)
        train_sec = time.perf_counter() - t0
        print(f"  train+load: {train_sec:.1f}s")

        m = _eval_record_list(
            eval_records, score_fn,
            ema_alpha=args.ema_alpha, ema_threshold=args.ema_threshold,
        )

        print(f"  per-sentence       F1={m['per_sentence']['f1']:.3f}  "
              f"AUROC={m['per_sentence'].get('auroc', float('nan')):.3f}")
        print(f"  per-call (mean)    F1={m['per_call_mean']['f1']:.3f}  "
              f"AUROC={m['per_call_mean'].get('auroc', float('nan')):.3f}")
        print(f"  per-call streaming F1={m['per_call_streaming']['f1']:.3f}  "
              f"prec={m['per_call_streaming']['precision']:.3f}  "
              f"rec={m['per_call_streaming']['recall']:.3f}")

        results[name] = {
            "train_sec": train_sec,
            **m,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\n[saved] {out_path}")

    print("\n=== Summary: per-call streaming F1 ===")
    print(f"{'baseline':<28}{'F1':>8}{'prec':>8}{'rec':>8}{'AUROC':>10}")
    print("-" * 64)
    for name, m in results.items():
        s = m["per_call_streaming"]
        print(f"{name:<28}{s['f1']:>8.3f}{s['precision']:>8.3f}"
              f"{s['recall']:>8.3f}{s['auroc']:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
