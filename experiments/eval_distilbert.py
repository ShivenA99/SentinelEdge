"""Evaluate a fine-tuned DistilBERT checkpoint with the SentinelEdge harness.

Same three evaluation modes as the lightweight baselines:
  per-sentence, per-call mean, per-call streaming EMA.

Also measures real per-sentence latency (single CPU thread, 50 warmup +
500 timed samples, 3 reps) so the result drops into the same Pareto
plot as the other baselines.

Usage
-----
    python experiments/eval_distilbert.py
    python experiments/eval_distilbert.py --ckpt models/distilbert_scam \\
        --eval-sources repo_real teleantifraud_28k
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord, load_repo_real, load_teleantifraud, load_better30,
)
from experiments.run_baselines import _eval_record_list  # noqa: E402

_LOADERS = {
    "repo_real": load_repo_real,
    "teleantifraud_28k": load_teleantifraud,
    "better30": load_better30,
}


def _import_torch():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer,
        )
        return torch, AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print(f"[fatal] torch / transformers not installed: {e}", file=sys.stderr)
        sys.exit(2)


def make_scorer(ckpt: Path, device: str, threads: int):
    """Return a (score_fn, info) pair for a fine-tuned DistilBERT."""
    torch, AutoTokenizer, AutoModelForSequenceClassification = _import_torch()
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(
            f"No DistilBERT checkpoint at {ckpt}. "
            "Run `python experiments/train_distilbert.py` first."
        )
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    if device == "cpu":
        torch.set_num_threads(threads)
    model.to(device)

    def score(text: str) -> float:
        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            return float(torch.softmax(logits, dim=-1)[0, 1].item())

    nparams = sum(p.numel() for p in model.parameters())
    info = {
        "model": "distilbert_finetuned",
        "ckpt": str(ckpt),
        "n_params_M": nparams / 1e6,
        "disk_size_mb_estimate": nparams * 4 / (1024 * 1024),
        "device": device,
    }
    return score, info


def time_predictions(score_fn, sentences, n_warm: int, n_time: int,
                     reps: int) -> dict:
    """Measure per-sentence latency the same way run_latency.py does."""
    def gen(n):
        for i in range(n):
            yield sentences[i % len(sentences)]
    for s in gen(n_warm):
        score_fn(s)
    p50s, p95s, p99s, throughputs = [], [], [], []
    for _ in range(reps):
        t = []
        t0 = time.perf_counter()
        for s in gen(n_time):
            ts = time.perf_counter()
            score_fn(s)
            t.append((time.perf_counter() - ts) * 1000.0)
        t1 = time.perf_counter()
        t.sort()
        p50s.append(t[len(t) // 2])
        p95s.append(t[int(len(t) * 0.95)])
        p99s.append(t[int(len(t) * 0.99)])
        throughputs.append(n_time / (t1 - t0))
    return {
        "p50_ms": statistics.median(p50s),
        "p95_ms": statistics.median(p95s),
        "p99_ms": statistics.median(p99s),
        "throughput_sent_per_sec": statistics.median(throughputs),
        "reps": reps, "n_time": n_time, "n_warm": n_warm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_PROJECT_ROOT / "models" / "distilbert_scam"))
    ap.add_argument("--eval-sources", nargs="+", default=["repo_real"])
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU threads. Default 1 to match the lightweight latency benchmark.")
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)
    ap.add_argument("--n-warm", type=int, default=50)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "distilbert.json"),
    )
    args = ap.parse_args()

    torch, _, _ = _import_torch()
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else
        "cpu"
    ) if args.device == "auto" else args.device

    print(f"[device] {device}")
    score_fn, info = make_scorer(Path(args.ckpt), device, args.threads)
    print(f"[model]  {info}")

    # Quality on the same record list as everything else
    eval_records: list[CallRecord] = []
    for s in args.eval_sources:
        eval_records.extend(_LOADERS[s]())
    print(f"[load]   {len(eval_records)} eval call records from {args.eval_sources}")

    print(f"\n=== quality ===")
    quality = _eval_record_list(
        eval_records, score_fn,
        ema_alpha=args.ema_alpha, ema_threshold=args.ema_threshold,
    )
    print(f"  per-sentence    F1={quality['per_sentence']['f1']:.3f}  "
          f"AUROC={quality['per_sentence'].get('auroc', float('nan')):.3f}")
    print(f"  per-call mean   F1={quality['per_call_mean']['f1']:.3f}")
    print(f"  streaming EMA   F1={quality['per_call_streaming']['f1']:.3f}  "
          f"prec={quality['per_call_streaming']['precision']:.3f}  "
          f"rec={quality['per_call_streaming']['recall']:.3f}")

    # Latency
    print(f"\n=== latency ({args.reps} reps x {args.n_time} samples) ===")
    sentences = []
    for r in eval_records:
        sentences.extend(r.sentences)
    if not sentences:
        print("  no sentences available, skipping latency", file=sys.stderr)
        latency = {}
    else:
        latency = time_predictions(
            score_fn, sentences, args.n_warm, args.n_time, args.reps,
        )
        print(f"  p50={latency['p50_ms']:.1f} ms  p95={latency['p95_ms']:.1f} ms  "
              f"throughput={latency['throughput_sent_per_sec']:.1f} sent/s")

    out = {
        "config": vars(args),
        "info": info,
        "quality": quality,
        "latency": latency,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
