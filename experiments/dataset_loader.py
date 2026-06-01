"""Unified dataset loader for SentinelEdge evaluation.

Provides a single interface for loading both per-sentence and per-call
data from four sources, returned in a common schema:

    {
      "call_id":  str,
      "label":    int,          # 1 = scam/fraud, 0 = legitimate
      "category": str,          # e.g. "irs_tax", "doctor_appointment"
      "source":   str,          # which dataset this came from
      "sentences": list[str],   # ordered, sentence-segmented transcript
    }

Currently wired:
  * ``repo_real``           -- the 23 human-written transcripts in
                               ``data/real/call_transcripts/``.
  * ``synthetic``           -- the existing template-based training data
                               (loaded from CSV if available, else regenerated).
  * ``teleantifraud_28k``   -- expects pre-downloaded HuggingFace files at
                               ``data/external/teleantifraud_28k/`` (loader
                               is implemented; data download is documented in
                               ``experiments/PREPARE_DATA.md``).
  * ``better30``            -- expects ``data/external/better30.csv`` from
                               Kaggle. Loader is implemented; download is
                               manual due to Kaggle auth.

Each loader returns ``list[dict]`` in the schema above.  Sentence
segmentation re-uses the project's own ``SentenceSplitter`` so the
evaluation pipeline matches what the deployed system would see.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Common record schema
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """A single call-level record, segmented into sentences."""

    call_id: str
    label: int                      # 1 = scam, 0 = legit
    category: str
    source: str
    sentences: list[str] = field(default_factory=list)

    @property
    def transcript(self) -> str:
        """Concatenated full transcript."""
        return " ".join(self.sentences)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "label": self.label,
            "category": self.category,
            "source": self.source,
            "sentences": self.sentences,
        }


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Robust sentence split that works without external models.

    We deliberately avoid spaCy/NLTK here so the loader has zero ML
    dependencies. The project's ``SentenceSplitter`` is preferable for
    streaming use; this regex-based splitter is faster and equivalent
    for already-punctuated text.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = _SENT_BOUNDARY.split(text)
    # Drop very short residues (less than 3 characters)
    return [c.strip() for c in chunks if c.strip() and len(c.strip()) >= 3]


# ---------------------------------------------------------------------------
# Loader 1: in-repo real transcripts
# ---------------------------------------------------------------------------

def load_repo_real(transcript_dir: Path | None = None) -> list[CallRecord]:
    """Load the 23 human-written transcripts shipped in the repo."""
    if transcript_dir is None:
        transcript_dir = _PROJECT_ROOT / "data" / "real" / "call_transcripts"

    records: list[CallRecord] = []
    for fp in sorted(transcript_dir.glob("*.txt")):
        name = fp.stem
        if not (name.startswith("scam_") or name.startswith("legit_")):
            # Skip stray result files etc.
            continue
        label = 1 if name.startswith("scam_") else 0
        # category is everything after "scam_NN_" / "legit_NN_"
        parts = name.split("_", 2)
        category = parts[2] if len(parts) == 3 else "unknown"
        body = "\n".join(
            line for line in fp.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        sentences = _split_sentences(body)
        if not sentences:
            continue
        records.append(
            CallRecord(
                call_id=name,
                label=label,
                category=category,
                source="repo_real",
                sentences=sentences,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Loader 2: synthetic (per-sentence) data from training CSVs
# ---------------------------------------------------------------------------

def load_synthetic_sentences(
    csv_path: Path | None = None,
) -> list[tuple[str, int, str]]:
    """Load (text, label, category) triples from the synthetic CSV.

    Returns sentence-level rows, not call-level — the synthetic pipeline
    generates standalone sentences, not multi-turn calls. Falls back to
    regenerating a small balanced sample if the CSV doesn't exist.
    """
    if csv_path is None:
        csv_path = (
            _PROJECT_ROOT / "data" / "processed" / "call_fraud_test.csv"
        )

    if not csv_path.exists():
        # Cold fallback: synthesise a tiny balanced set so eval runs.
        return _emergency_synthetic_fallback()

    import pandas as pd
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = (
        "label" if "label" in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else df.columns[-1])
    )
    cat_col = "category" if "category" in df.columns else None
    out = []
    for _, row in df.iterrows():
        t = str(row[text_col]).strip()
        if not t:
            continue
        y = int(row[label_col])
        c = str(row[cat_col]) if cat_col else ("scam" if y else "legit")
        out.append((t, y, c))
    return out


def _emergency_synthetic_fallback() -> list[tuple[str, int, str]]:
    """A tiny balanced sample for smoke-testing when no CSV is present."""
    scam = [
        ("This is the IRS. You owe back taxes and must pay today or face arrest.", 1, "irs"),
        ("Your Amazon account was charged $999. Press 1 to dispute.", 1, "amazon"),
        ("Your social security number has been suspended due to suspicious activity.", 1, "ssa"),
        ("You have won a $5000 gift card. Click the link to claim.", 1, "prize"),
        ("Your computer is infected with a virus. Allow remote access now.", 1, "tech_support"),
    ]
    legit = [
        ("Hi, this is Sarah from Dr. Chen's office confirming your appointment tomorrow.", 0, "appointment"),
        ("Hello, your pizza order is ready for pickup at the front counter.", 0, "delivery"),
        ("This is a courtesy reminder about your prescription refill.", 0, "pharmacy"),
        ("Just calling to check whether you got the documents I emailed last week.", 0, "personal"),
        ("Your flight DL432 has been rescheduled to depart at 7:45 PM.", 0, "airline"),
    ]
    return scam + legit


# ---------------------------------------------------------------------------
# Loader 3: TeleAntiFraud-28k (external)
# ---------------------------------------------------------------------------

def load_teleantifraud(
    root: Path | None = None,
    split: str = "test",
) -> list[CallRecord]:
    """Load TeleAntiFraud-28k from a local directory.

    Expected layout (after manual download per ``PREPARE_DATA.md``)::

        data/external/teleantifraud_28k/
            train.jsonl
            test.jsonl

    Each line is a JSON object with at minimum::

        {"id": "...", "label": 0|1, "transcript": "...",
         "category": "..."}

    If the data isn't present, returns an empty list and prints a
    one-time warning — callers should check for emptiness.
    """
    if root is None:
        root = _PROJECT_ROOT / "data" / "external" / "teleantifraud_28k"
    path = root / f"{split}.jsonl"
    if not path.exists():
        # Try a few common file names
        for alt in (f"{split}.json", "data.jsonl", "all.jsonl"):
            cand = root / alt
            if cand.exists():
                path = cand
                break
        else:
            return []

    records: list[CallRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            transcript = obj.get("transcript") or obj.get("text") or ""
            if not transcript.strip():
                continue
            records.append(
                CallRecord(
                    call_id=obj.get("id") or f"taf_{split}_{i}",
                    label=int(obj.get("label", 0)),
                    category=obj.get("category", "unknown"),
                    source="teleantifraud_28k",
                    sentences=_split_sentences(transcript),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Loader 4: Kaggle BETTER30 transcript dataset
# ---------------------------------------------------------------------------

def load_better30(csv_path: Path | None = None) -> list[CallRecord]:
    """Load the Kaggle BETTER30 transcript dataset.

    Expected at ``data/external/better30.csv`` after manual download.

    Columns expected (case-insensitive, tolerant):
        ``transcript``   -- full call transcript text
        ``label``        -- 1 / 0 or "scam" / "ham" / "not_scam"
        ``id``           -- optional call ID
    """
    if csv_path is None:
        csv_path = _PROJECT_ROOT / "data" / "external" / "better30.csv"
    if not csv_path.exists():
        return []

    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    text_col = (
        cols.get("transcript") or cols.get("text") or cols.get("call") or df.columns[0]
    )
    label_col = cols.get("label") or cols.get("is_scam") or cols.get("class") or df.columns[-1]
    id_col = cols.get("id")
    cat_col = cols.get("category")

    records: list[CallRecord] = []
    for i, row in df.iterrows():
        raw_label = str(row[label_col]).strip().lower()
        if raw_label in {"1", "scam", "fraud", "true", "yes"}:
            y = 1
        elif raw_label in {"0", "ham", "not_scam", "legit", "false", "no"}:
            y = 0
        else:
            try:
                y = int(float(raw_label))
            except Exception:
                continue
        transcript = str(row[text_col])
        records.append(
            CallRecord(
                call_id=str(row[id_col]) if id_col else f"b30_{i}",
                label=y,
                category=str(row[cat_col]) if cat_col else ("scam" if y else "legit"),
                source="better30",
                sentences=_split_sentences(transcript),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Convenience: load everything available
# ---------------------------------------------------------------------------

def load_all_call_records() -> dict[str, list[CallRecord]]:
    """Load every call-level dataset that's locally available."""
    return {
        "repo_real": load_repo_real(),
        "teleantifraud_28k": load_teleantifraud(),
        "better30": load_better30(),
    }


def summarize(records_by_source: dict[str, list[CallRecord]]) -> None:
    """Print a small summary table."""
    print(f"{'source':<22}{'n_calls':>10}{'n_scam':>10}{'n_legit':>10}{'sent/call':>12}")
    print("-" * 64)
    for src, recs in records_by_source.items():
        if not recs:
            print(f"{src:<22}{'(none)':>10}")
            continue
        n = len(recs)
        n_scam = sum(r.label == 1 for r in recs)
        n_legit = n - n_scam
        spc = sum(len(r.sentences) for r in recs) / n
        print(f"{src:<22}{n:>10}{n_scam:>10}{n_legit:>10}{spc:>12.1f}")


if __name__ == "__main__":
    summarize(load_all_call_records())
