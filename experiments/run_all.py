"""Reproduce the full SentinelEdge paper evaluation in one command.

Runs every experiment in dependency order, then assembles a single
``results/paper_tables.json`` that ``make_figures.py`` and the paper
prose pull numbers from.

Stages:
  1. Regenerate synthetic train/test CSV (idempotent).
  2. Run evaluation, time-to-detection, latency, baselines, ASR, adversarial.
  3. Build figures.
  4. Build a summary JSON keyed by the paper's table/figure references.

By default, neural baselines are skipped (need HF model + Anthropic key).
Pass ``--with-neural`` to include them.

Usage
-----
    python experiments/run_all.py
    python experiments/run_all.py --sources repo_real teleantifraud_28k
    python experiments/run_all.py --with-neural
    python experiments/run_all.py --skip-stages latency,asr
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable


def _run(cmd: list[str], stage: str) -> tuple[int, float]:
    print(f"\n{'='*70}\n[{stage}] {' '.join(cmd)}\n{'='*70}")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(_PROJECT_ROOT))
    dt = time.perf_counter() - t0
    print(f"[{stage}] rc={rc}  elapsed={dt:.1f}s")
    return rc, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["repo_real"])
    ap.add_argument("--with-neural", action="store_true")
    ap.add_argument("--legacy-neural", action="store_true",
                    help="Use the bundled run_baselines_neural.py path "
                         "instead of train_distilbert.py + eval_distilbert.py.")
    ap.add_argument(
        "--skip-stages", default="",
        help="Comma-separated list of stage names to skip "
             "(data, evaluation, ttd, latency, baselines, asr, "
             "adversarial, neural, figures, summary)",
    )
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}

    timings = {}

    if "data" not in skip:
        # Make sure synthetic CSVs exist.
        train_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
        if not train_csv.exists():
            _run([_PY, "training/generate_synthetic_data.py"], "data:syn")
            _run([_PY, "training/prepare_datasets.py"], "data:prep")
        adv_csv = (_PROJECT_ROOT / "data" / "raw" / "synthetic_transcripts" /
                   "adversarial_calls.csv")
        if not adv_csv.exists():
            _run([_PY, "training/generate_adversarial_data.py"], "data:adv")

    if "evaluation" not in skip:
        rc, dt = _run([
            _PY, "experiments/run_evaluation.py",
            "--sources", *args.sources,
        ], "evaluation")
        timings["evaluation"] = dt

    if "ttd" not in skip:
        rc, dt = _run([
            _PY, "experiments/run_time_to_detection.py",
            "--sources", *args.sources,
        ], "ttd")
        timings["ttd"] = dt

    if "latency" not in skip:
        rc, dt = _run([_PY, "experiments/run_latency.py"], "latency")
        timings["latency"] = dt

    if "baselines" not in skip:
        rc, dt = _run([
            _PY, "experiments/run_baselines.py",
            "--eval-sources", *args.sources,
        ], "baselines")
        timings["baselines"] = dt

    if "asr" not in skip:
        rc, dt = _run([
            _PY, "experiments/run_asr_robustness.py",
            "--sources", *args.sources,
        ], "asr")
        timings["asr"] = dt

    if "adversarial" not in skip:
        rc, dt = _run([_PY, "experiments/run_adversarial.py"], "adversarial")
        timings["adversarial"] = dt

    if args.with_neural and "neural" not in skip:
        # Modern path: dedicated train + eval scripts. Falls back to
        # the older bundled run_baselines_neural.py if --legacy-neural.
        if args.legacy_neural:
            rc, dt = _run([
                _PY, "experiments/run_baselines_neural.py",
                "--eval-sources", *args.sources,
            ], "neural")
            timings["neural"] = dt
        else:
            # Train if no checkpoint
            ckpt = _PROJECT_ROOT / "models" / "distilbert_scam" / "config.json"
            if not ckpt.exists():
                rc, dt = _run([_PY, "experiments/train_distilbert.py"], "neural:train")
                timings["neural_train"] = dt
            else:
                print("[neural] DistilBERT checkpoint exists, skipping train")
            rc, dt = _run([
                _PY, "experiments/eval_distilbert.py",
                "--eval-sources", *args.sources,
            ], "neural:eval")
            timings["neural_eval"] = dt

    if "figures" not in skip:
        rc, dt = _run([_PY, "experiments/make_figures.py"], "figures")
        timings["figures"] = dt

    if "summary" not in skip:
        # Stitch a single paper_tables.json from the result files
        results_dir = _PROJECT_ROOT / "results"
        summary: dict = {"timings_sec": timings, "sources": args.sources}
        for f in [
            "eval_xgb", "ttd", "latency", "baselines",
            "asr_robustness", "adversarial", "baselines_neural",
        ]:
            p = results_dir / f"{f}.json"
            if p.exists():
                summary[f] = json.loads(p.read_text())
        out = results_dir / "paper_tables.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"\n[summary] wrote {out} ({out.stat().st_size // 1024} KB)")

    print(f"\n{'='*70}\nALL DONE\n{'='*70}")
    for k, v in timings.items():
        print(f"  {k:<14} {v:>6.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
