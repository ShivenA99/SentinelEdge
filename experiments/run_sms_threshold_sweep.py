"""SMS threshold sweep: test the calibration-shift hypothesis.

The cross-channel paragraph hypothesised that zero-shot transfer to SMS
*preserves AUROC* (ranking intact) but collapses $F_1$ purely because the
*decision threshold* tuned on calls is mis-calibrated for SMS. This script
tests the second half by sweeping the decision threshold over the SMS
predictions. On the real SMS Spam Collection the hypothesis does NOT hold:
re-thresholding recovers almost no $F_1$ (gain is sub-point), because the
channel's class imbalance caps precision (low AUPRC). The transfer failure
is therefore more fundamental than a calibration offset; the script reports
both the (near-flat) sweep and the AUROC/AUPRC that explain it.

It reports, over the SMS Spam Collection predictions:

  * $F_1$ at the call-tuned threshold (the value already in the paper),
  * the SMS-optimal threshold $\\tau^\\star$ and the $F_1$ there,
  * the full $F_1(\\tau)$ curve for $\\tau \\in [0, 1]$ at 0.01 steps,
  * precision / recall at $\\tau^\\star$.

Two ways to obtain the per-item scores:

  1. ``--from-json results/cross_channel.json`` (default) -- reads
     ``results.sms.y_score`` + ``results.sms.y_true`` if present. These
     arrays appear once ``run_cross_channel.metrics_block`` is patched to
     persist them (the same per-item-score patch used for the AUROC CIs).
  2. ``--rescore`` -- loads the SMS corpus and the call-trained classifier
     and computes the scores directly (needs xgboost + the model files;
     runs on the full environment).

If neither source yields scores, the script prints the exact JSON shape
it expects so the upstream patch is mechanical.

Outputs
-------
  * markdown table to stdout
  * ``results/sms_threshold_sweep.json`` (full curve + summary)
  * ``paper/_sms_sweep.tex`` (macros: ``\\SmsCallTau``, ``\\SmsTauStar``,
    ``\\SmsFOneAtCallTau``, ``\\SmsFOneAtSmsTau``, ``\\SmsFOneGainPts``,
    ``\\SmsPrecAtSmsTau``, ``\\SmsRecAtSmsTau``)
  * ``paper/figures/fig_sms_sweep.pdf`` (only if matplotlib is installed)

Add ``\\input{_sms_sweep}`` next to ``\\input{_numbers}`` once generated.

Usage
-----
    python experiments/run_sms_threshold_sweep.py
    python experiments/run_sms_threshold_sweep.py --rescore
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_TRUE_KEYS = ("y_true", "labels", "y")
_SCORE_KEYS = ("y_score", "scores", "probabilities", "proba")


# ---------------------------------------------------------------------------
# Score acquisition
# ---------------------------------------------------------------------------

def _find_arrays(block: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull (y_true, y_score) out of a results block if both are present."""
    if not isinstance(block, dict):
        return None
    true_key = next((k for k in _TRUE_KEYS if k in block), None)
    score_key = next((k for k in _SCORE_KEYS if k in block), None)
    if true_key is None or score_key is None:
        return None
    y_true = np.asarray(block[true_key]).astype(int)
    y_score = np.asarray(block[score_key], dtype=float)
    if y_true.shape != y_score.shape or y_true.size == 0:
        return None
    return y_true, y_score


def scores_from_json(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    results = data.get("results", data) if isinstance(data, dict) else {}
    sms = results.get("sms", {}) if isinstance(results, dict) else {}
    return _find_arrays(sms)


def scores_from_rescore(model_path: str, tfidf_path: str
                        ) -> tuple[np.ndarray, np.ndarray] | None:
    """Score the SMS corpus directly with the call-trained classifier."""
    try:
        from experiments.run_cross_channel import load_sms, score_batch
        from sentinel_edge.classifier.xgb_classifier import FraudClassifier
        from sentinel_edge.features.feature_pipeline import FeaturePipeline
    except Exception as exc:
        print(f"[error] cannot import scoring deps ({type(exc).__name__}: {exc}). "
              "xgboost is required for the call_fraud_xgb.json head.",
              file=sys.stderr)
        return None
    texts, y = load_sms()
    if len(texts) == 0:
        print("[error] SMS corpus not found at "
              "data/real/sms_spam/SMSSpamCollection.tsv", file=sys.stderr)
        return None
    tp = tfidf_path if Path(tfidf_path).exists() else None
    pipeline = FeaturePipeline(tp)
    classifier = FraudClassifier(model_path)
    scores = score_batch(texts, pipeline, classifier)
    return np.asarray(y).astype(int), np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(y_true: np.ndarray, y_score: np.ndarray, step: float
          ) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, f1_at_each_threshold)."""
    taus = np.round(np.arange(0.0, 1.0 + step / 2, step), 4)
    f1s = np.array([
        f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        for t in taus
    ])
    return taus, f1s


def _metrics_at(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> dict:
    y_pred = (y_score >= tau).astype(int)
    return {
        "threshold": float(tau),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_markdown(call: dict, best: dict, n: int, n_pos: int) -> None:
    gain = (best["f1"] - call["f1"]) * 100
    print(f"\n## SMS threshold sweep  (n={n}, n_pos={n_pos})")
    print("| Threshold | F1 | Precision | Recall |")
    print("|---|---|---|---|")
    print(f"| call-tuned τ={call['threshold']:.2f} | {call['f1']:.3f} | "
          f"{call['precision']:.3f} | {call['recall']:.3f} |")
    print(f"| SMS-optimal τ*={best['threshold']:.2f} | {best['f1']:.3f} | "
          f"{best['precision']:.3f} | {best['recall']:.3f} |")
    print(f"\nF1 gain from recalibration: +{gain:.1f} points "
          f"({call['f1']:.3f} → {best['f1']:.3f})")


def write_json(path: Path, taus: np.ndarray, f1s: np.ndarray,
               call: dict, best: dict, n: int, n_pos: int) -> None:
    payload = {
        "n": n,
        "n_pos": n_pos,
        "call_tuned": call,
        "sms_optimal": best,
        "f1_gain_points": round((best["f1"] - call["f1"]) * 100, 2),
        "curve": {"thresholds": taus.tolist(), "f1": [round(v, 6) for v in f1s]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


def write_tex(path: Path, call: dict, best: dict,
              auroc: float, auprc: float) -> None:
    gain = (best["f1"] - call["f1"]) * 100
    lines = [
        "% Auto-generated by experiments/run_sms_threshold_sweep.py",
        "% SMS decision-threshold sweep: the F1 collapse is NOT recovered by",
        "% re-thresholding (gain is near-zero); precision is capped by class",
        "% imbalance (low AUPRC), so transfer failure is more than calibration.",
        "",
        f"\\newcommand{{\\SmsCallTau}}{{{call['threshold']:.2f}}}",
        f"\\newcommand{{\\SmsTauStar}}{{{best['threshold']:.2f}}}",
        f"\\newcommand{{\\SmsFOneAtCallTau}}{{{call['f1']:.3f}}}",
        f"\\newcommand{{\\SmsFOneAtSmsTau}}{{{best['f1']:.3f}}}",
        f"\\newcommand{{\\SmsFOneGainPts}}{{{gain:.1f}}}",
        f"\\newcommand{{\\SmsPrecAtSmsTau}}{{{best['precision']:.3f}}}",
        f"\\newcommand{{\\SmsRecAtSmsTau}}{{{best['recall']:.3f}}}",
        f"\\newcommand{{\\SmsSweepAuroc}}{{{auroc:.3f}}}",
        f"\\newcommand{{\\SmsAuprc}}{{{auprc:.3f}}}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def maybe_write_figure(path: Path, taus: np.ndarray, f1s: np.ndarray,
                       call: dict, best: dict, y_true: np.ndarray,
                       y_score: np.ndarray, auprc: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except Exception:
        print("[info] matplotlib not installed; skipping figure", file=sys.stderr)
        return

    prevalence = float((y_true == 1).mean())
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 2.3))

    # Left: F1 is flat across the decision threshold.
    axl.plot(taus, f1s, color="#1f77b4", lw=1.5)
    axl.axvline(call["threshold"], color="#888", ls="--", lw=1,
                label=f"call τ={call['threshold']:.2f}")
    axl.axvline(best["threshold"], color="#d62728", ls=":", lw=1.2,
                label=f"SMS τ*={best['threshold']:.2f}")
    axl.scatter([call["threshold"]], [call["f1"]], color="#888", zorder=5)
    axl.scatter([best["threshold"]], [best["f1"]], color="#d62728", zorder=5)
    axl.set_xlabel("Decision threshold τ")
    axl.set_ylabel("SMS $F_1$")
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_title("$F_1$ is flat across τ", fontsize=8)
    axl.legend(fontsize=6, loc="upper right")

    # Right: PR envelope caps precision (low AUPRC).
    axr.plot(rec, prec, color="#2ca02c", lw=1.5,
             label=f"AUPRC={auprc:.3f}")
    axr.axhline(prevalence, color="#888", ls="--", lw=1,
                label=f"no-skill={prevalence:.3f}")
    axr.set_xlabel("Recall")
    axr.set_ylabel("Precision")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.set_title("PR envelope caps precision", fontsize=8)
    axr.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-json", type=Path,
                    default=_PROJECT_ROOT / "results" / "cross_channel.json")
    ap.add_argument("--rescore", action="store_true",
                    help="Score the SMS corpus directly (needs xgboost).")
    ap.add_argument("--model",
                    default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"))
    ap.add_argument("--tfidf",
                    default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"))
    ap.add_argument("--call-tau", type=float, default=0.5,
                    help="Threshold used for the cross-channel SMS row.")
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out-json", type=Path,
                    default=_PROJECT_ROOT / "results" / "sms_threshold_sweep.json")
    ap.add_argument("--out-tex", type=Path,
                    default=_PROJECT_ROOT / "paper" / "_sms_sweep.tex")
    ap.add_argument("--figure", type=Path,
                    default=_PROJECT_ROOT / "paper" / "figures" / "fig_sms_sweep.pdf")
    args = ap.parse_args()

    arrays = None
    if args.rescore:
        arrays = scores_from_rescore(args.model, args.tfidf)
    else:
        arrays = scores_from_json(args.from_json)

    if arrays is None:
        print(
            "\n[error] no SMS per-item scores available.\n"
            "Provide them one of two ways:\n"
            "  1. Re-run with --rescore (needs xgboost + model files), or\n"
            "  2. Patch experiments/run_cross_channel.py metrics_block to add,\n"
            "     for the SMS block, the arrays:\n"
            '         \"y_true\":  [0, 1, 0, ...]   # one int per message\n'
            '         \"y_score\": [0.02, 0.91, ...]  # predict_proba per message\n'
            "     then re-run run_cross_channel.py and this script.\n"
            "Expected JSON path: results.sms.{y_true,y_score}.",
            file=sys.stderr,
        )
        return 1

    y_true, y_score = arrays
    n, n_pos = int(y_true.size), int((y_true == 1).sum())
    if len(np.unique(y_true)) < 2:
        print("[error] SMS labels are single-class; cannot sweep.", file=sys.stderr)
        return 1

    taus, f1s = sweep(y_true, y_score, args.step)
    best_idx = int(np.argmax(f1s))
    best = _metrics_at(y_true, y_score, float(taus[best_idx]))
    call = _metrics_at(y_true, y_score, args.call_tau)
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    print_markdown(call, best, n, n_pos)
    print(f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
          f"(max F1 over all thresholds = {best['f1']:.3f})")
    write_json(args.out_json, taus, f1s, call, best, n, n_pos)
    write_tex(args.out_tex, call, best, auroc, auprc)
    maybe_write_figure(args.figure, taus, f1s, call, best,
                       y_true, y_score, auprc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
