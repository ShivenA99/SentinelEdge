"""Neural baselines for SentinelEdge.

Two baselines, both runnable separately because they require model
downloads or API access that the main `run_baselines.py` deliberately
avoids:

  * ``distilbert_finetuned``  -- DistilBERT-base-uncased fine-tuned on the
    synthetic training data (one epoch, frozen embeddings except last
    layer). Saves checkpoint to ``models/distilbert_scam/`` on first
    run; reloads it on subsequent runs.

  * ``claude_zeroshot`` -- Calls the Anthropic API to score each
    sentence. Requires ``ANTHROPIC_API_KEY`` in the environment. Uses
    Claude Haiku by default for cost; switch with --model.

Both baselines write per-call results into ``results/baselines_neural.json``
in exactly the same schema as ``run_baselines.py``, so the paper-table
script can merge them.

Usage
-----
    # DistilBERT fine-tune + evaluate (5-10 min on CPU)
    python experiments/run_baselines_neural.py \\
        --baselines distilbert_finetuned \\
        --eval-sources repo_real teleantifraud_28k

    # Claude zero-shot evaluation (API key required)
    export ANTHROPIC_API_KEY=...
    python experiments/run_baselines_neural.py \\
        --baselines claude_zeroshot \\
        --claude-model claude-haiku-4-5 \\
        --eval-sources repo_real

    # Both at once
    python experiments/run_baselines_neural.py
"""
from __future__ import annotations

import argparse
import json
import os
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
from experiments.run_baselines import _eval_record_list  # noqa: E402

_LOADERS = {
    "repo_real": load_repo_real,
    "teleantifraud_28k": load_teleantifraud,
    "better30": load_better30,
}


# ===========================================================================
# DistilBERT fine-tuned baseline
# ===========================================================================

def _train_distilbert(train_csv: Path, ckpt_dir: Path,
                      epochs: int = 1, batch: int = 32) -> None:
    """Fine-tune DistilBERT on the synthetic training CSV."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, AdamW,
    )

    df = pd.read_csv(train_csv)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class _DS(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            return self.texts[i], self.labels[i]

    def collate(batch):
        ts = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        enc = tok(ts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return {k: v.to(device) for k, v in enc.items()}

    loader = DataLoader(_DS(texts, labels), batch_size=batch,
                        shuffle=True, collate_fn=collate)
    opt = AdamW(model.parameters(), lr=3e-5)
    model.train()
    for ep in range(epochs):
        running = 0.0
        for i, batch_d in enumerate(loader):
            opt.zero_grad()
            out = model(**batch_d)
            out.loss.backward()
            opt.step()
            running += out.loss.item()
            if (i + 1) % 50 == 0:
                print(f"  ep{ep} step{i+1} loss={running / (i+1):.4f}")
        print(f"  ep{ep} done, mean loss={running/len(loader):.4f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    print(f"[save] {ckpt_dir}")


def make_distilbert_scorer(args):
    """Return (score_fn, info) for the DistilBERT baseline.

    Trains the model on the synthetic CSV on first run, caches it, and
    reuses on subsequent runs.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt_dir = Path(args.distilbert_ckpt)
    train_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
    if not (ckpt_dir / "config.json").exists():
        print(f"[distilbert] no checkpoint at {ckpt_dir}, fine-tuning ...")
        _train_distilbert(train_csv, ckpt_dir,
                          epochs=args.distilbert_epochs,
                          batch=args.distilbert_batch)

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()
    torch.set_num_threads(1)
    device = "cpu"  # for fair on-device comparison
    model.to(device)

    def score(text: str) -> float:
        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            return float(torch.softmax(logits, dim=-1)[0, 1].item())

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "model": "distilbert_finetuned",
        "ckpt_dir": str(ckpt_dir),
        "n_params_M": n_params / 1e6,
        "disk_size_mb_estimate": n_params * 4 / (1024 * 1024),
    }
    return score, info


# ===========================================================================
# Anthropic / Claude zero-shot baseline
# ===========================================================================

_CLAUDE_PROMPT = (
    "You are a scam detection classifier. Given a single sentence from "
    "a phone call transcript, output ONLY a single number in [0,1] "
    "representing the probability that the *call* this sentence comes "
    "from is a scam. Do not output any explanation. The sentence:\n\n"
    "\"\"\"\n{sentence}\n\"\"\"\n\nProbability:"
)


def make_claude_scorer(args):
    import urllib.request
    import urllib.error

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to run claude_zeroshot.")
    model = args.claude_model

    def score(text: str) -> float:
        body = json.dumps({
            "model": model,
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": _CLAUDE_PROMPT.format(sentence=text)}
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  [claude] HTTP {e.code}: {e.reason}", file=sys.stderr)
            return 0.5
        # Take the first text block, strip, parse float
        out = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                out = block.get("text", "")
                break
        try:
            v = float(out.strip().split()[0])
            return max(0.0, min(1.0, v))
        except (ValueError, IndexError):
            return 0.5

    info = {"model": f"claude_zeroshot_{model}"}
    return score, info


# ===========================================================================
# Driver
# ===========================================================================

BASELINES = {
    "distilbert_finetuned": make_distilbert_scorer,
    "claude_zeroshot": make_claude_scorer,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baselines", nargs="+",
        default=list(BASELINES.keys()),
        choices=list(BASELINES.keys()),
    )
    ap.add_argument("--eval-sources", nargs="+", default=["repo_real"])
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)

    ap.add_argument(
        "--distilbert-ckpt",
        default=str(_PROJECT_ROOT / "models" / "distilbert_scam"),
    )
    ap.add_argument("--distilbert-epochs", type=int, default=1)
    ap.add_argument("--distilbert-batch", type=int, default=32)

    ap.add_argument("--claude-model", default="claude-haiku-4-5")

    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "baselines_neural.json"),
    )
    args = ap.parse_args()

    eval_records: list[CallRecord] = []
    for s in args.eval_sources:
        eval_records.extend(_LOADERS[s]())
    print(f"[load] {len(eval_records)} eval call records")

    results = {}
    for name in args.baselines:
        print(f"\n=== baseline: {name} ===")
        t0 = time.perf_counter()
        try:
            score_fn, info = BASELINES[name](args)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            results[name] = {"error": str(e)}
            continue
        load_sec = time.perf_counter() - t0
        print(f"  load: {load_sec:.1f}s -- {info}")

        m = _eval_record_list(
            eval_records, score_fn,
            ema_alpha=args.ema_alpha, ema_threshold=args.ema_threshold,
        )
        print(f"  per-call streaming F1={m['per_call_streaming']['f1']:.3f}  "
              f"prec={m['per_call_streaming']['precision']:.3f}  "
              f"rec={m['per_call_streaming']['recall']:.3f}")

        results[name] = {"load_sec": load_sec, "info": info, **m}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
