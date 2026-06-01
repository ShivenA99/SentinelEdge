"""Fine-tune DistilBERT on SentinelEdge synthetic call transcripts.

Trains a binary scam classifier on top of distilbert-base-uncased,
producing a HuggingFace-format checkpoint at ``models/distilbert_scam/``
that the evaluation scripts and ``run_baselines_neural.py`` can load.

Why a separate script
---------------------
Fine-tuning is slow (5-30 min CPU, 1-2 min GPU) and only needs to run
once per data revision. Pulling it out of ``run_baselines_neural.py``
means you can:

  - resume from a checkpoint without re-evaluating
  - inspect intermediate validation metrics
  - run on a different machine (GPU box) than the final eval

Outputs
-------
``models/distilbert_scam/``:
  config.json, model.safetensors, tokenizer.json, special_tokens_map.json
  vocab.txt, training_log.jsonl

``training_log.jsonl`` has one JSON object per training step::

  {"step": 0, "epoch": 0.0, "loss": 0.6932, "lr": 3e-5,
   "elapsed_sec": 0.4, "phase": "train"}

and one per validation pass::

  {"step": 500, "epoch": 0.6, "val_loss": 0.231, "val_acc": 0.912,
   "val_f1": 0.910, "elapsed_sec": 78.2, "phase": "val"}

Usage
-----
    python experiments/train_distilbert.py
    python experiments/train_distilbert.py --epochs 2 --batch 16
    python experiments/train_distilbert.py --device cuda --max-steps 1000
    python experiments/train_distilbert.py --resume models/distilbert_scam/
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


def _import_torch():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        # AdamW moved between transformers and torch.optim across versions;
        # try both.
        try:
            from torch.optim import AdamW
        except ImportError:
            from transformers import AdamW  # noqa: F401  -- older transformers
        return torch, AutoTokenizer, AutoModelForSequenceClassification, AdamW
    except Exception as e:
        print(f"[fatal] failed to import torch/transformers: {e}",
              file=sys.stderr)
        print("Install with:\n  pip install torch transformers",
              file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    """Load a (text, label) split from a SentinelEdge CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = (
        "label" if "label" in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else df.columns[-1])
    )
    return df[text_col].astype(str).tolist(), df[label_col].astype(int).tolist()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> int:
    torch, AutoTokenizer, AutoModelForSequenceClassification, AdamW = _import_torch()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv) if args.val_csv else None
    ckpt_dir = Path(args.output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pick device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] using {device}")
    if device == "cpu":
        torch.set_num_threads(args.threads)

    print(f"[data] train CSV: {train_csv}")
    train_texts, train_labels = load_split(train_csv)
    print(f"[data]   {len(train_texts)} samples, "
          f"scam fraction = {sum(train_labels)/len(train_labels):.3f}")

    if val_csv and val_csv.exists():
        print(f"[data] val CSV:   {val_csv}")
        val_texts, val_labels = load_split(val_csv)
        print(f"[data]   {len(val_texts)} samples")
    else:
        # 90/10 holdout
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(train_texts))
        rng.shuffle(idx)
        cut = int(0.9 * len(idx))
        val_texts = [train_texts[i] for i in idx[cut:]]
        val_labels = [train_labels[i] for i in idx[cut:]]
        train_texts = [train_texts[i] for i in idx[:cut]]
        train_labels = [train_labels[i] for i in idx[:cut]]
        print(f"[data] no val CSV -- using 90/10 split: "
              f"{len(train_texts)} train / {len(val_texts)} val")

    # Tokenizer / model
    print(f"[model] loading {args.base_model}")
    if args.resume and (Path(args.resume) / "config.json").exists():
        print(f"[model] resuming from {args.resume}")
        tok = AutoTokenizer.from_pretrained(args.resume)
        model = AutoModelForSequenceClassification.from_pretrained(args.resume)
    else:
        tok = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=2,
        )
    model.to(device)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] {nparams/1e6:.1f}M parameters")

    # DataLoader
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, t, y):
            self.t, self.y = t, y
        def __len__(self):
            return len(self.t)
        def __getitem__(self, i):
            return self.t[i], self.y[i]

    def collate(batch):
        ts = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        enc = tok(
            ts, padding=True, truncation=True, max_length=args.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return {k: v.to(device) for k, v in enc.items()}

    train_loader = DataLoader(
        _DS(train_texts, train_labels),
        batch_size=args.batch, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        _DS(val_texts, val_labels),
        batch_size=args.batch, shuffle=False, collate_fn=collate,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    log_path = ckpt_dir / "training_log.jsonl"
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(rec: dict) -> None:
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()

    def evaluate() -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        model.eval()
        all_logits, all_y = [], []
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(**batch)
                loss_sum += out.loss.item() * batch["labels"].size(0)
                n += batch["labels"].size(0)
                all_logits.append(out.logits.cpu().numpy())
                all_y.append(batch["labels"].cpu().numpy())
        model.train()
        y_pred = np.concatenate(all_logits).argmax(axis=1)
        y_true = np.concatenate(all_y)
        return {
            "val_loss": loss_sum / max(n, 1),
            "val_acc": float(accuracy_score(y_true, y_pred)),
            "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "val_n": int(n),
        }

    print(f"[train] epochs={args.epochs}  batch={args.batch}  "
          f"lr={args.lr}  max_steps={args.max_steps or 'all'}")
    t_start = time.perf_counter()
    step = 0
    model.train()
    for ep in range(args.epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            running += out.loss.item()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                rec = {
                    "step": step,
                    "epoch": ep + (i + 1) / len(train_loader),
                    "loss": running / args.log_every,
                    "lr": args.lr,
                    "elapsed_sec": elapsed,
                    "phase": "train",
                }
                print(f"  step {step:>5}  ep {rec['epoch']:.2f}  "
                      f"loss {rec['loss']:.4f}  ({elapsed:.0f}s)")
                log(rec)
                running = 0.0

            if step % args.eval_every == 0:
                vm = evaluate()
                vm.update({
                    "step": step,
                    "epoch": ep + (i + 1) / len(train_loader),
                    "elapsed_sec": time.perf_counter() - t_start,
                    "phase": "val",
                })
                print(f"  [val ] step {step:>5}  loss {vm['val_loss']:.4f}  "
                      f"acc {vm['val_acc']:.3f}  F1 {vm['val_f1']:.3f}")
                log(vm)

            if args.max_steps and step >= args.max_steps:
                print(f"[train] stopping at max_steps={args.max_steps}")
                break

        if args.max_steps and step >= args.max_steps:
            break

        # End-of-epoch eval
        vm = evaluate()
        vm.update({
            "step": step,
            "epoch": ep + 1.0,
            "elapsed_sec": time.perf_counter() - t_start,
            "phase": "val",
        })
        print(f"  [val ] end-of-epoch {ep+1}  loss {vm['val_loss']:.4f}  "
              f"acc {vm['val_acc']:.3f}  F1 {vm['val_f1']:.3f}")
        log(vm)

    # Save final
    print(f"[save] {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    log_fh.close()

    # Also write a single-shot summary metadata file
    metadata = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_length": args.max_length,
        "n_params": nparams,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "device": device,
        "total_train_sec": time.perf_counter() - t_start,
    }
    (ckpt_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"[done] {(time.perf_counter() - t_start):.0f}s total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument(
        "--train-csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"),
    )
    ap.add_argument(
        "--val-csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "call_fraud_test.csv"),
    )
    ap.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "models" / "distilbert_scam"),
    )
    ap.add_argument("--resume", default=None,
                    help="Resume from a checkpoint directory.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Hard cap on training steps (useful for smoke tests).")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads (only used if device=cpu).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
