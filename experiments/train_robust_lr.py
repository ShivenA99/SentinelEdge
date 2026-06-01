"""Train robust LR models for the YouTube Whisper transcript shift.

Training mix:
  * synthetic sentence data
  * ASR-style perturbed copies of the training text
  * repo_real + BothBosu calls
  * 70% of the YouTube scam-baiter calls

Evaluation:
  * held-out 30% of YouTube calls only
  * streaming EMA at a threshold selected on the 70% YouTube train fold

The script writes ``results/robust_lr.json`` and a pickled sklearn
artifact under ``models/robust_lr.joblib``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord,
    load_bothbosu,
    load_repo_real,
    load_synthetic_sentences,
    load_youtube_baiters,
)
from sentinel_edge.classifier.score_accumulator import ScoreAccumulator  # noqa: E402
from sentinel_edge.features.handcrafted import extract_handcrafted_features  # noqa: E402


def _handcrafted_matrix(texts: list[str]) -> np.ndarray:
    return np.asarray(
        [list(extract_handcrafted_features(text).values()) for text in texts],
        dtype=np.float32,
    )


def _asr_perturb(text: str, rng: random.Random) -> str:
    """Cheap deterministic ASR-noise proxy for Whisper-style transcripts."""
    text = text.lower()
    text = text.replace("$", " dollars ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "too": "to",
        "two": "to",
        "four": "for",
        "your": "you are",
        "verify": "very fy",
        "account": "acount",
        "gift": "give",
        "card": "car",
        "social": "so shall",
        "security": "secure tea",
    }
    toks = [replacements.get(tok, tok) for tok in text.split()]
    if len(toks) > 6:
        toks = [tok for tok in toks if rng.random() > 0.08]
    return " ".join(toks)


def _call_sentence_rows(
    records: list[CallRecord],
    *,
    max_per_call: int | None = None,
) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for rec in records:
        sentences = [s.strip() for s in rec.sentences if len(s.split()) >= 2]
        if max_per_call is not None and len(sentences) > max_per_call:
            idx = np.linspace(0, len(sentences) - 1, max_per_call).astype(int)
            sentences = [sentences[i] for i in idx]
        rows.extend((sentence, rec.label, rec.source) for sentence in sentences)
    return rows


def _load_sentence_training_rows(
    youtube_train: list[CallRecord],
    *,
    rng: random.Random,
    max_youtube_segments_per_call: int,
) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for split_name in ["call_fraud_train.csv", "call_fraud_test.csv"]:
        rows.extend(
            (text, label, "synthetic")
            for text, label, _category in load_synthetic_sentences(
                _PROJECT_ROOT / "data" / "processed" / split_name
            )
        )

    rows.extend(_call_sentence_rows(load_repo_real(), max_per_call=None))
    rows.extend(_call_sentence_rows(load_bothbosu(), max_per_call=40))
    rows.extend(
        _call_sentence_rows(
            youtube_train,
            max_per_call=max_youtube_segments_per_call,
        )
    )

    augmented = list(rows)
    for text, label, source in rows:
        perturbed = _asr_perturb(text, rng)
        if perturbed and perturbed != text:
            augmented.append((perturbed, label, f"{source}_asr_aug"))
    return augmented


class RobustLRScorer:
    def __init__(
        self,
        *,
        name: str,
        clf: LogisticRegression,
        scaler: StandardScaler | None,
        vectorizers: list[TfidfVectorizer],
    ) -> None:
        self.name = name
        self.clf = clf
        self.scaler = scaler
        self.vectorizers = vectorizers

    def _matrix(self, texts: list[str]):
        blocks = []
        if self.scaler is not None:
            blocks.append(sp.csr_matrix(self.scaler.transform(_handcrafted_matrix(texts))))
        for vectorizer in self.vectorizers:
            blocks.append(vectorizer.transform(texts))
        if len(blocks) == 1:
            return blocks[0]
        return sp.hstack(blocks).tocsr()

    def scores(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self._matrix(texts))[:, 1]

    def score(self, text: str) -> float:
        return float(self.scores([text])[0])


def _fit_candidate(
    name: str,
    texts: list[str],
    y: np.ndarray,
    *,
    use_handcrafted: bool,
    vectorizers: list[TfidfVectorizer],
    c_value: float,
) -> RobustLRScorer:
    blocks = []
    scaler = None
    if use_handcrafted:
        scaler = StandardScaler().fit(_handcrafted_matrix(texts))
        blocks.append(sp.csr_matrix(scaler.transform(_handcrafted_matrix(texts))))

    fitted_vectorizers = []
    for vectorizer in vectorizers:
        fitted = vectorizer.fit(texts)
        fitted_vectorizers.append(fitted)
        blocks.append(fitted.transform(texts))

    X = blocks[0] if len(blocks) == 1 else sp.hstack(blocks).tocsr()
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=c_value,
        solver="liblinear",
    )
    clf.fit(X, y)
    return RobustLRScorer(
        name=name,
        clf=clf,
        scaler=scaler,
        vectorizers=fitted_vectorizers,
    )


def _peak_ema_scores(
    records: list[CallRecord],
    scorer: RobustLRScorer,
    *,
    ema_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    peaks = []
    for rec in records:
        y_true.append(rec.label)
        if not rec.sentences:
            peaks.append(0.0)
            continue
        sentence_scores = scorer.scores(rec.sentences)
        acc = ScoreAccumulator(alpha=ema_alpha)
        peak = 0.0
        for score in sentence_scores:
            peak = max(peak, acc.update(float(score)))
        peaks.append(peak)
    return np.asarray(y_true), np.asarray(peaks)


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {
            "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
            "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
            "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
            "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        },
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    grid: np.ndarray,
) -> tuple[float, dict]:
    best_threshold = float(grid[0])
    best_metrics = _metrics(y_true, y_score, best_threshold)
    for threshold in grid[1:]:
        metrics = _metrics(y_true, y_score, float(threshold))
        better = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
        ) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
        )
        if better:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _candidate_specs():
    return [
        (
            "handcrafted_lr",
            True,
            [],
            [0.3, 1.0, 3.0],
        ),
        (
            "word_tfidf_lr",
            False,
            [
                TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                )
            ],
            [0.3, 1.0, 3.0],
        ),
        (
            "word_tfidf_hand_lr",
            True,
            [
                TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                )
            ],
            [0.3, 1.0, 3.0],
        ),
        (
            "char_tfidf_hand_lr",
            True,
            [
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=3000,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                )
            ],
            [0.3, 1.0, 3.0],
        ),
        (
            "word_char_tfidf_hand_lr",
            True,
            [
                TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=2500,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ],
            [0.3, 1.0, 3.0],
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--youtube-test-size", type=float, default=0.30)
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--threshold-min", type=float, default=0.2)
    ap.add_argument("--threshold-max", type=float, default=0.9)
    ap.add_argument("--threshold-steps", type=int, default=57)
    ap.add_argument("--max-youtube-segments-per-call", type=int, default=160)
    ap.add_argument("--out", default=str(_PROJECT_ROOT / "results" / "robust_lr.json"))
    ap.add_argument("--model-out", default=str(_PROJECT_ROOT / "models" / "robust_lr.joblib"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    youtube_records = load_youtube_baiters()
    if len(youtube_records) < 10:
        print("[abort] youtube_baiters has too few records", file=sys.stderr)
        return 1

    youtube_train, youtube_test = train_test_split(
        youtube_records,
        test_size=args.youtube_test_size,
        random_state=args.seed,
        stratify=[rec.label for rec in youtube_records],
    )
    train_rows = _load_sentence_training_rows(
        youtube_train,
        rng=rng,
        max_youtube_segments_per_call=args.max_youtube_segments_per_call,
    )
    texts = [text for text, _label, _source in train_rows]
    y = np.asarray([label for _text, label, _source in train_rows], dtype=int)

    print(
        "[split] youtube train="
        f"{len(youtube_train)} calls ({sum(r.label for r in youtube_train)} scam), "
        f"test={len(youtube_test)} calls ({sum(r.label for r in youtube_test)} scam)"
    )
    print(f"[train] {len(texts)} sentence rows after ASR augmentation; scam fraction={y.mean():.3f}")

    threshold_grid = np.linspace(
        args.threshold_min,
        args.threshold_max,
        args.threshold_steps,
    )
    results = {}
    best_name = ""
    best_scorer: RobustLRScorer | None = None
    best_result: dict | None = None

    for base_name, use_handcrafted, vectorizers, c_values in _candidate_specs():
        for c_value in c_values:
            name = f"{base_name}_C{c_value:g}"
            print(f"\n=== {name} ===")
            fit_t0 = time.perf_counter()
            scorer = _fit_candidate(
                name,
                texts,
                y,
                use_handcrafted=use_handcrafted,
                vectorizers=vectorizers,
                c_value=c_value,
            )
            print(f"  fit_sec={time.perf_counter() - fit_t0:.1f}")

            y_train, train_scores = _peak_ema_scores(
                youtube_train,
                scorer,
                ema_alpha=args.ema_alpha,
            )
            threshold, train_metrics = _select_threshold(
                y_train,
                train_scores,
                grid=threshold_grid,
            )
            y_test, test_scores = _peak_ema_scores(
                youtube_test,
                scorer,
                ema_alpha=args.ema_alpha,
            )
            test_metrics = _metrics(y_test, test_scores, threshold)
            result = {
                "name": name,
                "c": c_value,
                "threshold_selected_on_youtube_train": threshold,
                "youtube_train_streaming": train_metrics,
                "youtube_test_streaming": test_metrics,
            }
            results[name] = result
            print(
                "  train F1={:.3f} @ thr={:.3f} | test F1={:.3f} "
                "prec={:.3f} rec={:.3f} AUROC={:.3f} conf={}".format(
                    train_metrics["f1"],
                    threshold,
                    test_metrics["f1"],
                    test_metrics["precision"],
                    test_metrics["recall"],
                    test_metrics["auroc"],
                    test_metrics["confusion"],
                )
            )

            # Model selection is based only on the YouTube train fold. The
            # held-out YouTube test fold is reported, never used to select.
            if best_result is None or (
                train_metrics["f1"],
                train_metrics["auroc"],
                train_metrics["precision"],
                -c_value,
            ) > (
                best_result["youtube_train_streaming"]["f1"],
                best_result["youtube_train_streaming"]["auroc"],
                best_result["youtube_train_streaming"]["precision"],
                -best_result["c"],
            ):
                best_name = name
                best_scorer = scorer
                best_result = result

    assert best_scorer is not None and best_result is not None
    out = {
        "config": vars(args),
        "training_rows": {
            "n": len(texts),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        },
        "youtube_split": {
            "train_calls": len(youtube_train),
            "train_pos": int(sum(r.label for r in youtube_train)),
            "train_neg": int(len(youtube_train) - sum(r.label for r in youtube_train)),
            "test_calls": len(youtube_test),
            "test_pos": int(sum(r.label for r in youtube_test)),
            "test_neg": int(len(youtube_test) - sum(r.label for r in youtube_test)),
            "test_call_ids": [r.call_id for r in youtube_test],
        },
        "candidates": results,
        "best_model": best_name,
        "headline": best_result,
        "elapsed_sec": time.perf_counter() - t0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as fh:
        pickle.dump(
            {
                "scorer": best_scorer,
                "threshold": best_result["threshold_selected_on_youtube_train"],
                "config": vars(args),
                "best_model": best_name,
            },
            fh,
        )
    print(f"[saved] {model_out}")

    test = best_result["youtube_test_streaming"]
    print("\n=== robust LR headline: held-out YouTube streaming ===")
    print(
        f"  model={best_name}\n"
        f"  F1={test['f1']:.3f}  precision={test['precision']:.3f}  "
        f"recall={test['recall']:.3f}  AUROC={test['auroc']:.3f}\n"
        f"  threshold={best_result['threshold_selected_on_youtube_train']:.3f}  "
        f"confusion={test['confusion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
