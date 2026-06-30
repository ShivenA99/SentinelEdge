#!/usr/bin/env python3
r"""bootstrap_cis.py — Bootstrap confidence intervals for SentinelEdge.

Addresses Reviewer #1's "no statistical treatment" weakness. Computes
95% bootstrap percentile intervals for F1, precision, recall, and
(when probability scores are available) AUROC on each classifier head,
per-source breakdown, and the channel-disjoint robustness experiment.

USAGE
-----
  # Default: run on results/baselines.json and results/channel_disjoint.json
  python bootstrap_cis.py

  # Custom paths
  python bootstrap_cis.py \
      --baselines results/baselines.json \
      --channel-disjoint results/channel_disjoint.json \
      --out-json results/bootstrap_cis.json \
      --out-tex  paper/_cis.tex

  # Sanity check on synthetic data (no JSON needed)
  python bootstrap_cis.py --dry-run

  # Tighter intervals (slower)
  python bootstrap_cis.py --n-iter 20000

INPUT SCHEMA (best case)
------------------------
The script first looks for raw per-call predictions and labels:

  baselines.json
    {
      "handcrafted_lr": {
        "per_call_streaming": {
          "y_true":  [0, 1, 1, 0, ...],         # per-call labels
          "y_pred":  [0, 1, 0, 0, ...],         # per-call binary
          "y_score": [0.12, 0.91, 0.44, ...]    # optional, enables AUROC
        },
        "per_sentence_independent": {...},
        ...
      },
      "tfidf_lr": {...},
      ...
    }

INPUT SCHEMA (fallback)
-----------------------
If only confusion-matrix counts are present, the script reconstructs a
synthetic per-call array from {tn, fp, fn, tp} that exactly reproduces
the point estimate. Bootstrap CIs from this reconstruction are valid
for F1/precision/recall but cannot produce AUROC CIs.

  "per_call_streaming": {
    "confusion": {"tn": 670, "fp": 4, "fn": 255, "tp": 494}
  }

OUTPUT
------
- Markdown table to stdout (paste into a rebuttal or response letter).
- results/bootstrap_cis.json — structured intervals for every cell.
- paper/_cis.tex — LaTeX macros like \HandLRFOneCI etc., ready to
  \input in the paper. Each macro renders as "0.792 [0.770, 0.812]".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------
# Metric primitives (binary classification)
# -----------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tn, fp, fn, tp


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, fn, tp = _confusion(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tp = _confusion(y_true, y_pred)
    denom = tp + fp
    return (tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = _confusion(y_true, y_pred)
    denom = tp + fn
    return (tp / denom) if denom > 0 else 0.0


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUROC. No sklearn dependency."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend (0, 0) so the trapezoid starts at the origin.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x rename
    return float(trapz(tpr, fpr))


# -----------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------

def bootstrap_ci(
    metric_fn,
    *arrays,
    n_iter: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Returns (point, lo, hi) for the percentile bootstrap.

    `metric_fn(*resampled_arrays)` must return a scalar.
    `arrays` are 1D and resampled jointly (same indices).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(arrays[0])
    point = float(metric_fn(*arrays))
    if n == 0:
        return point, float("nan"), float("nan")

    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        resampled = tuple(a[idx] for a in arrays)
        samples[i] = metric_fn(*resampled)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


# -----------------------------------------------------------------
# Schema handling
# -----------------------------------------------------------------

def reconstruct_from_confusion(
    tn: int, fp: int, fn: int, tp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (y_true, y_pred) arrays whose confusion matrix
    exactly matches the given counts. Sufficient for bootstrapping
    precision/recall/F1; cannot recover AUROC.
    """
    y_true, y_pred = [], []
    y_true.extend([0] * tn); y_pred.extend([0] * tn)
    y_true.extend([0] * fp); y_pred.extend([1] * fp)
    y_true.extend([1] * fn); y_pred.extend([0] * fn)
    y_true.extend([1] * tp); y_pred.extend([1] * tp)
    return np.array(y_true), np.array(y_pred)


def extract_arrays(block: dict) -> dict[str, np.ndarray] | None:
    """Pull (y_true, y_pred, [y_score]) from a block of the input JSON.

    Tries, in order:
      1. y_true + y_pred (+ optional y_score) keys
      2. labels + predictions (+ scores) keys
      3. confusion {tn, fp, fn, tp} -> reconstructed binary arrays
    Returns None if nothing usable is found.
    """
    # Mode 1: explicit arrays
    for true_key, pred_key, score_key in [
        ("y_true", "y_pred", "y_score"),
        ("labels", "predictions", "scores"),
        ("y_true", "y_pred", "probabilities"),
    ]:
        if true_key in block and pred_key in block:
            out = {
                "y_true": np.asarray(block[true_key]).astype(int),
                "y_pred": np.asarray(block[pred_key]).astype(int),
            }
            if score_key in block:
                out["y_score"] = np.asarray(block[score_key]).astype(float)
            return out

    # Mode 1b: labels + scores (no explicit predictions). Derive the binary
    # predictions from the block's decision threshold (default 0.5). This is
    # what the result JSONs store, and it is what enables AUROC CIs.
    true_key = next((k for k in ("y_true", "labels") if k in block), None)
    score_key = next(
        (k for k in ("y_score", "scores", "probabilities") if k in block), None
    )
    if true_key and score_key:
        y_true = np.asarray(block[true_key]).astype(int)
        y_score = np.asarray(block[score_key]).astype(float)
        thr = float(block.get("threshold", 0.5))
        y_pred = (y_score >= thr).astype(int)
        return {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}

    # Mode 2: confusion matrix
    conf = block.get("confusion") or block.get("cm")
    if conf and all(k in conf for k in ("tn", "fp", "fn", "tp")):
        y_true, y_pred = reconstruct_from_confusion(
            conf["tn"], conf["fp"], conf["fn"], conf["tp"]
        )
        return {"y_true": y_true, "y_pred": y_pred}

    return None


# -----------------------------------------------------------------
# Driver
# -----------------------------------------------------------------

def fmt_ci(point: float, lo: float, hi: float, prec: int = 3) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return f"{point:.{prec}f}"
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def fmt_ci_pm(point: float, lo: float, hi: float, prec: int = 3) -> str:
    """Symmetric ± form, using half the CI width."""
    if np.isnan(lo) or np.isnan(hi):
        return f"{point:.{prec}f}"
    half = (hi - lo) / 2
    return f"{point:.{prec}f} \\pm {half:.{prec}f}"


def latex_macro_name(head: str, mode: str, metric: str) -> str:
    """e.g. handcrafted_lr / per_call_streaming / f1 -> HandLRStreamFOneCI."""
    head_map = {
        "handcrafted_lr": "HandLR", "hand_lr": "HandLR",
        "handcrafted_svm": "HandSVM", "hand_svm": "HandSVM",
        "tfidf_lr": "TfidfLR",
        "combined_lr": "CombLR", "comb_lr": "CombLR",
        "xgboost": "Xgb", "xgb": "Xgb",
        "distilbert": "Distil",
    }
    mode_map = {
        "per_sentence_independent": "PerSent", "per_sentence": "PerSent",
        "per_call_mean": "Mean", "per_call_streaming": "Stream",
        "full_transcript": "Full", "streaming": "Stream",
    }
    metric_map = {"f1": "FOne", "precision": "Prec", "recall": "Rec", "auroc": "Auroc"}
    h = head_map.get(head, head.title().replace("_", ""))
    m = mode_map.get(mode, mode.title().replace("_", ""))
    met = metric_map[metric]
    return f"{h}{m}{met}CI"


def run_block(
    block: dict, *, n_iter: int, rng: np.random.Generator
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap CIs for all metrics on one (head, mode) block."""
    arrs = extract_arrays(block)
    if arrs is None:
        return {}
    y_true, y_pred = arrs["y_true"], arrs["y_pred"]
    out = {}
    out["f1"]        = bootstrap_ci(f1,        y_true, y_pred, n_iter=n_iter, rng=rng)
    out["precision"] = bootstrap_ci(precision, y_true, y_pred, n_iter=n_iter, rng=rng)
    out["recall"]    = bootstrap_ci(recall,    y_true, y_pred, n_iter=n_iter, rng=rng)
    if "y_score" in arrs:
        out["auroc"] = bootstrap_ci(auroc, y_true, arrs["y_score"],
                                    n_iter=n_iter, rng=rng)
    return out


def walk_baselines(
    data: dict, *, n_iter: int, rng: np.random.Generator
) -> dict[str, dict[str, dict]]:
    """Walk a nested {head: {mode: block}} JSON."""
    results: dict[str, dict[str, dict]] = {}
    for head, modes in data.items():
        if not isinstance(modes, dict):
            continue
        results[head] = {}
        for mode, block in modes.items():
            if not isinstance(block, dict):
                continue
            cis = run_block(block, n_iter=n_iter, rng=rng)
            if cis:
                results[head][mode] = cis
    return results


def walk_channel_disjoint(
    data: dict, *, n_iter: int, rng: np.random.Generator
) -> dict[str, dict]:
    """Channel-disjoint typically has {model_variant: block} structure."""
    results: dict[str, dict] = {}
    for key, block in data.items():
        if not isinstance(block, dict):
            continue
        cis = run_block(block, n_iter=n_iter, rng=rng)
        if cis:
            results[key] = cis
    return results


# -----------------------------------------------------------------
# Synthetic dry-run data
# -----------------------------------------------------------------

def synthetic_data() -> dict:
    rng = np.random.default_rng(7)
    # Replicate the documented LR numbers approximately.
    n_scam, n_legit = 749, 674
    p_lr_scam  = rng.beta(3.5, 1.0, n_scam)
    p_lr_legit = rng.beta(1.0, 4.5, n_legit)
    p_xgb_scam  = rng.beta(2.5, 1.5, n_scam)
    p_xgb_legit = rng.beta(0.7, 5.0, n_legit)
    def make_head(p_scam, p_legit, thr=0.5):
        y_score = np.concatenate([p_scam, p_legit])
        y_true  = np.concatenate([np.ones(n_scam), np.zeros(n_legit)]).astype(int)
        y_pred  = (y_score >= thr).astype(int)
        return {"y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_score": y_score.tolist()}
    return {
        "handcrafted_lr": {"per_call_streaming": make_head(p_lr_scam, p_lr_legit, thr=0.62)},
        "xgboost":        {"per_call_streaming": make_head(p_xgb_scam, p_xgb_legit, thr=0.55)},
    }


# -----------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------

def print_markdown(results_baselines, results_channel) -> None:
    print("\n## Bootstrap 95% CIs — baselines.json")
    print("| Head | Mode | F1 | Precision | Recall | AUROC |")
    print("|---|---|---|---|---|---|")
    for head, modes in results_baselines.items():
        for mode, cis in modes.items():
            row = [head, mode]
            for k in ("f1", "precision", "recall", "auroc"):
                if k in cis:
                    row.append(fmt_ci(*cis[k]))
                else:
                    row.append("—")
            print("| " + " | ".join(row) + " |")
    if results_channel:
        print("\n## Bootstrap 95% CIs — channel_disjoint.json")
        print("| Variant | F1 | Precision | Recall | AUROC |")
        print("|---|---|---|---|---|")
        for variant, cis in results_channel.items():
            row = [variant]
            for k in ("f1", "precision", "recall", "auroc"):
                if k in cis:
                    row.append(fmt_ci(*cis[k]))
                else:
                    row.append("—")
            print("| " + " | ".join(row) + " |")


def write_json(path: Path, results_baselines, results_channel) -> None:
    serial = {"baselines": {}, "channel_disjoint": {}}
    for head, modes in results_baselines.items():
        serial["baselines"][head] = {}
        for mode, cis in modes.items():
            serial["baselines"][head][mode] = {
                k: {"point": p, "lo": lo, "hi": hi}
                for k, (p, lo, hi) in cis.items()
            }
    for variant, cis in results_channel.items():
        serial["channel_disjoint"][variant] = {
            k: {"point": p, "lo": lo, "hi": hi}
            for k, (p, lo, hi) in cis.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial, indent=2))
    print(f"\nwrote {path}")


def write_tex_macros(path: Path, results_baselines, results_channel) -> None:
    r"""Emit \newcommand definitions: \HandLRStreamFOneCI etc.

    Each macro renders as "0.792 [0.770, 0.812]" in math mode.
    Use them inline like:  $F_1 = \HandLRStreamFOneCI$.
    """
    lines = [
        "% Auto-generated by experiments/bootstrap_cis.py",
        "% Bootstrap 95% percentile intervals over per-call resamples.",
        "% Render inline as e.g.  $F_1 = \\HandLRStreamFOneCI$.",
        "",
    ]
    # LaTeX command names may contain letters only (no digits), so metric
    # tokens must be spelled out (f1 -> FOne) for the disjoint fallback too.
    metric_token = {"f1": "FOne", "precision": "Prec",
                    "recall": "Rec", "auroc": "Auroc"}

    def emit(scope: str, key: str, mode: str | None, cis: dict):
        for metric, (p, lo, hi) in cis.items():
            if mode:
                name = latex_macro_name(key, mode, metric)
            else:
                met = metric_token.get(metric, metric.title())
                name = f"{key.title().replace('_', '')}{met}CI"
            body = fmt_ci(p, lo, hi)
            lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")
        lines.append("")
    for head, modes in results_baselines.items():
        for mode, cis in modes.items():
            emit("baseline", head, mode, cis)
    for variant, cis in results_channel.items():
        emit("disjoint", variant, None, cis)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines", type=Path, default=Path("results/baselines.json"))
    ap.add_argument("--channel-disjoint", type=Path,
                    default=Path("results/channel_disjoint.json"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/bootstrap_cis.json"))
    ap.add_argument("--out-tex", type=Path,
                    default=Path("paper/_cis.tex"))
    ap.add_argument("--n-iter", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Use synthetic data; do not read input files.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        print("DRY RUN: using synthetic per-call data.")
        baselines_data = synthetic_data()
        channel_data: dict = {}
    else:
        if not args.baselines.exists():
            print(f"error: {args.baselines} not found "
                  f"(use --dry-run to test on synthetic data)", file=sys.stderr)
            return 1
        baselines_data = json.loads(args.baselines.read_text())
        # Real result files wrap the per-head blocks under a top-level
        # "results" key (alongside "config"); unwrap so walk_baselines sees
        # the {head: {mode: block}} structure documented above.
        if isinstance(baselines_data, dict) and "results" in baselines_data \
                and isinstance(baselines_data["results"], dict):
            baselines_data = baselines_data["results"]
        channel_data = {}
        if args.channel_disjoint.exists():
            try:
                txt = args.channel_disjoint.read_text().strip()
                if txt:
                    channel_data = json.loads(txt)
            except json.JSONDecodeError as e:
                print(f"warning: could not parse {args.channel_disjoint}: {e}",
                      file=sys.stderr)

    print(f"bootstrapping with n_iter={args.n_iter}, seed={args.seed}...")
    results_baselines = walk_baselines(baselines_data, n_iter=args.n_iter, rng=rng)
    results_channel   = walk_channel_disjoint(channel_data, n_iter=args.n_iter, rng=rng)

    if not results_baselines and not results_channel:
        print("error: no usable blocks found in inputs. Either the schema "
              "differs from what this script expects, or the input files "
              "contain only summary stats with no per-call arrays or "
              "confusion matrices. Inspect the JSON and adjust "
              "extract_arrays() if needed.", file=sys.stderr)
        return 2

    print_markdown(results_baselines, results_channel)
    write_json(args.out_json, results_baselines, results_channel)
    write_tex_macros(args.out_tex, results_baselines, results_channel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
