"""ASR robustness experiment.

The deployed system uses Whisper transcripts, which contain errors. This
experiment measures how much the trained classifier degrades when its
input has realistic ASR noise.

Three perturbations, each parameterised by an error rate p:

  * **word_swap(p)**  -- with probability p, replace each token with a
                         homophone or random nearby English word
  * **word_delete(p)** -- with probability p, drop each token
  * **char_noise(p)** -- with probability p, apply a random
                         character-level perturbation (substitute,
                         delete, swap adjacent)

The first two approximate what whisper-tiny does on noisy 16 kHz audio:
substitutions and deletions are the dominant Whisper error modes.

We sweep p in {0.0, 0.05, 0.10, 0.20, 0.30, 0.50} and report streaming
F1 / precision / recall and Time-to-Detection at each level.

This isn't a replacement for true Whisper-on-audio evaluation. That is
the right experiment if you have audio and time (see PREPARE_DATA.md
for TeleAntiFraud-28k which ships both). But this lower-bounds the
ASR-error problem and is fully reproducible without any audio.

Usage
-----
    python experiments/run_asr_robustness.py
    python experiments/run_asr_robustness.py --rates 0.0 0.1 0.3
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.dataset_loader import (  # noqa: E402
    CallRecord, load_repo_real, load_better30,
    load_wu2024, load_bothbosu, load_youtube_baiters, load_teleantifraud,
)
from experiments.run_evaluation import (  # noqa: E402
    evaluate_per_call_streaming, evaluate_per_sentence,
)
from sentinel_edge.classifier.xgb_classifier import FraudClassifier  # noqa: E402
from sentinel_edge.features.feature_pipeline import FeaturePipeline  # noqa: E402


_LOADERS = {
    "repo_real": load_repo_real,
    "better30": load_better30,
    "wu2024_corpus": load_wu2024,
    "bothbosu": load_bothbosu,
    "youtube_baiters": load_youtube_baiters,
    "teleantifraud_28k": load_teleantifraud,
}


# A tiny phonetic-confusion list for word_swap. Not exhaustive --
# just enough to model realistic ASR slips for fraud-relevant tokens.
_CONFUSIONS = {
    "irs": ["iris", "i-r-s", "i.r.s.", "yrs", "eyes"],
    "ssa": ["sa", "essay", "s-s-a"],
    "account": ["a-count", "accent", "accost", "account."],
    "password": ["past word", "pass-word", "pass word"],
    "verify": ["verafy", "berify", "very", "verified"],
    "social": ["sosual", "soshull", "social"],
    "security": ["securitee", "security.", "securty"],
    "amazon": ["amazong", "amazon.", "amazin"],
    "apple": ["app le", "apo", "ample"],
    "microsoft": ["micro soft", "microsave", "microsft"],
    "urgent": ["urgient", "earned", "argument"],
    "immediately": ["immediatley", "immediatly", "intimately"],
    "department": ["depart ment", "departement", "departement"],
    "officer": ["off-er-cer", "officert", "offser"],
    "warrant": ["warant", "warent", "warent"],
    "arrest": ["a-rest", "a rest", "arrested"],
    "settlement": ["settlment", "settle ment", "settlement"],
    "agent": ["age int", "agen", "agile"],
    "federal": ["fedearal", "fedral", "fed earl"],
    "tax": ["tex", "taxs", "tacks"],
    "lawsuit": ["lawsute", "law suit", "law suits"],
    "click": ["clik", "klick", "click."],
    "link": ["lync", "li nk", "linked"],
}


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def word_swap(text: str, rate: float, rng: random.Random) -> str:
    """With probability `rate`, replace each token with a confusion."""
    if rate <= 0:
        return text
    out = []
    for tok in text.split():
        key = tok.lower().strip(".,!?;:\"'")
        if rng.random() < rate:
            opts = _CONFUSIONS.get(key)
            if opts:
                out.append(rng.choice(opts))
            else:
                # Fallback: swap two letters or drop one letter
                if len(tok) > 3:
                    i = rng.randrange(1, len(tok) - 1)
                    out.append(tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:])
                else:
                    out.append(tok)
        else:
            out.append(tok)
    return " ".join(out)


def word_delete(text: str, rate: float, rng: random.Random) -> str:
    """With probability `rate`, drop each token."""
    if rate <= 0:
        return text
    toks = text.split()
    kept = [t for t in toks if rng.random() >= rate]
    return " ".join(kept) if kept else (toks[0] if toks else text)


def char_noise(text: str, rate: float, rng: random.Random) -> str:
    """Character-level noise: insert / delete / swap each char with prob `rate`."""
    if rate <= 0:
        return text
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        if rng.random() < rate:
            op = rng.choice(["sub", "del", "swap"])
            if op == "sub":
                out.append(rng.choice(string.ascii_lowercase))
                i += 1
            elif op == "del":
                i += 1  # skip this char
            else:  # swap with next
                if i + 1 < len(chars):
                    out.append(chars[i + 1])
                    out.append(chars[i])
                    i += 2
                else:
                    out.append(chars[i])
                    i += 1
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


_PERTURBATIONS = {
    "word_swap": word_swap,
    "word_delete": word_delete,
    "char_noise": char_noise,
}


# ---------------------------------------------------------------------------
# Record perturbation
# ---------------------------------------------------------------------------

def perturb_records(
    records: list[CallRecord],
    fn,
    rate: float,
    seed: int = 0,
) -> list[CallRecord]:
    """Return a new list of records with each sentence perturbed."""
    rng = random.Random(seed)
    out: list[CallRecord] = []
    for r in records:
        new_sents = [fn(s, rate, rng) for s in r.sentences]
        out.append(CallRecord(
            call_id=r.call_id, label=r.label, category=r.category,
            source=r.source, sentences=new_sents,
        ))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["repo_real"])
    ap.add_argument(
        "--model", default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"),
    )
    ap.add_argument(
        "--tfidf",
        default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"),
    )
    ap.add_argument(
        "--perturbations", nargs="+",
        default=list(_PERTURBATIONS.keys()),
        choices=list(_PERTURBATIONS.keys()),
    )
    ap.add_argument(
        "--rates", nargs="+", type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--ema-threshold", type=float, default=0.75)
    ap.add_argument(
        "--out",
        default=str(_PROJECT_ROOT / "results" / "asr_robustness.json"),
    )
    args = ap.parse_args()

    base_records: list[CallRecord] = []
    for s in args.sources:
        base_records.extend(_LOADERS[s]())
    print(f"[load] {len(base_records)} call records")
    if not base_records:
        return 1

    pipeline = FeaturePipeline(args.tfidf)
    classifier = FraudClassifier(args.model)

    table = []
    for pert_name in args.perturbations:
        pert_fn = _PERTURBATIONS[pert_name]
        for rate in args.rates:
            recs = perturb_records(base_records, pert_fn, rate, seed=args.seed)
            sm = evaluate_per_call_streaming(
                recs, pipeline, classifier,
                ema_alpha=args.ema_alpha, ema_threshold=args.ema_threshold,
            )
            psm = evaluate_per_sentence(recs, pipeline, classifier)
            row = {
                "perturbation": pert_name,
                "rate": rate,
                "stream_f1": sm["f1"],
                "stream_precision": sm["precision"],
                "stream_recall": sm["recall"],
                "stream_auroc": sm.get("auroc", float("nan")),
                "sent_f1": psm["f1"],
                "sent_auroc": psm.get("auroc", float("nan")),
            }
            table.append(row)
            print(f"  {pert_name:>12} p={rate:.2f}  "
                  f"streamF1={row['stream_f1']:.3f}  "
                  f"sentF1={row['sent_f1']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": vars(args), "table": table}, indent=2,
    ))
    print(f"\n[saved] {out_path}")

    # Headline degradation: streaming F1 at p=0 vs p=0.30
    print("\n=== Degradation at p=0.30 vs p=0.0 ===")
    base = {r["perturbation"]: r["stream_f1"] for r in table if r["rate"] == 0.0}
    high = {r["perturbation"]: r["stream_f1"] for r in table if r["rate"] == 0.30}
    for k in base:
        if k in high:
            print(f"  {k:>12}: F1 {base[k]:.3f} -> {high[k]:.3f}  "
                  f"(delta {high[k]-base[k]:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
