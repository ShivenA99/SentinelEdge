"""Unified dataset loader for SentinelEdge evaluation.

Provides a single interface for loading both per-sentence and per-call
data from multiple sources, returned in a common schema:

    {
      "call_id":  str,
      "label":    int,          # 1 = scam/fraud, 0 = legitimate
      "category": str,          # e.g. "irs_tax", "doctor_appointment"
      "source":   str,          # which dataset this came from
      "sentences": list[str],   # ordered, sentence-segmented transcript
    }

Currently wired:
  * ``repo_real``         -- 23 human-written transcripts in
                             ``data/real/call_transcripts/``
  * ``better30``          -- Kaggle "Call Transcripts Scam
                             Determinations" (~30 real calls); place
                             at ``data/external/better30.csv``
  * ``wu2024_corpus``     -- aggregated corpus from Shen et al. 2024
                             (SC/SD/MASC/Our-Real/Our-Synt); requires
                             contacting authors -- see EMAIL_TEMPLATE.md
  * ``youtube_baiters``   -- self-collected YouTube scam-baiter corpus,
                             Whisper-transcribed and manually annotated;
                             see SCAMBAITER_PROTOCOL.md

The previously listed ``teleantifraud_28k`` loader is retained for
optional cross-lingual extensions but is **not part of the headline
evaluation** -- the corpus is Mandarin Chinese, not English.

Each loader returns ``list[dict]`` in the schema above. Loaders are
tolerant of missing data: if the external files aren't present the
loader returns an empty list and prints a one-line warning. This
makes ``load_all_call_records()`` safe to call in any environment.
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
# Loader 5: Wu et al. 2024 aggregated corpus
# ---------------------------------------------------------------------------

# Maps file stem -> (source_tag, default_label_if_subset_is_all_scam)
_WU2024_FILES = {
    "sc":       ("wu2024_sc", None),
    "sd":       ("wu2024_sd", None),
    "masc":     ("wu2024_masc", None),
    "our_real": ("wu2024_our_real", None),
    "our_synt": ("wu2024_our_synt", None),
}


def load_wu2024(root: Path | None = None) -> list[CallRecord]:
    """Load the aggregated corpus from Shen et al. 2024 (AAAI 2025).

    Expects (after Tier 2 acquisition per PREPARE_DATA.md) one CSV
    per subset at::

        data/external/wu2024_corpus/
            sc.csv
            sd.csv
            masc.csv
            our_real.csv
            our_synt.csv

    Each file should have a text column (any of ``text``, ``transcript``,
    ``dialogue``, ``content``) and a label column (``label``, ``is_scam``,
    ``class``). The loader is tolerant of missing files: only the
    subsets that are physically present get loaded.
    """
    if root is None:
        root = _PROJECT_ROOT / "data" / "external" / "wu2024_corpus"
    if not root.exists():
        return []

    import pandas as pd
    records: list[CallRecord] = []
    for stem, (src_tag, _) in _WU2024_FILES.items():
        candidates = [root / f"{stem}.csv", root / f"{stem}.tsv", root / f"{stem}.jsonl"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        cols = {c.lower(): c for c in df.columns}
        text_col = (cols.get("text") or cols.get("transcript")
                    or cols.get("dialogue") or cols.get("content")
                    or df.columns[0])
        label_col = (cols.get("label") or cols.get("is_scam")
                     or cols.get("class") or df.columns[-1])
        id_col = cols.get("id") or cols.get("call_id")
        for i, row in df.iterrows():
            raw_label = str(row[label_col]).strip().lower()
            if raw_label in {"1", "scam", "fraud", "true", "yes"}:
                y = 1
            elif raw_label in {"0", "ham", "not_scam", "legit", "normal", "false", "no"}:
                y = 0
            else:
                try:
                    y = int(float(raw_label))
                except Exception:
                    continue
            text = str(row[text_col])
            if not text.strip():
                continue
            records.append(CallRecord(
                call_id=str(row[id_col]) if id_col else f"{stem}_{i}",
                label=y,
                category=stem,
                source=src_tag,
                sentences=_split_sentences(text),
            ))
    return records


# ---------------------------------------------------------------------------
# Loader 5b: BothBosu (Gumphusiri 2024) HuggingFace datasets
# ---------------------------------------------------------------------------

import re as _re_bb
_TURN_PATTERN_BB = _re_bb.compile(
    r"(caller|receiver)\s*:\s*(.+?)(?=\s+(?:caller|receiver)\s*:|$)",
    flags=_re_bb.DOTALL | _re_bb.IGNORECASE,
)


def _parse_bothbosu_dialogue(dialogue: str, caller_only: bool = False) -> list[str]:
    if not dialogue or not isinstance(dialogue, str):
        return []
    turns = _TURN_PATTERN_BB.findall(dialogue)
    if not turns:
        return _split_sentences(dialogue)
    sentences: list[str] = []
    for speaker, utterance in turns:
        if caller_only and speaker.lower() != "caller":
            continue
        sentences.extend(_split_sentences(utterance.strip()))
    return sentences


def load_bothbosu(
    root: Path | None = None,
    subsets: list[str] | None = None,
    splits: list[str] | None = None,
    caller_only: bool = False,
) -> list[CallRecord]:
    """Load the BothBosu / Gumphusiri 2024 multi-turn scam dialogues.

    These are the publicly available SC/SD/MASC analogues that Shen et al.
    (2024) cite. Apache 2.0 license. Download via the snippet in
    EXPERIMENTS_TO_RUN.md, which places CSVs under
    ``data/external/bothbosu/``.

    Default behaviour evaluates on the *test* split of all four subsets
    and scores both speakers' utterances (matching the Gumphusiri /
    Shen et al. evaluation setup).
    """
    if root is None:
        root = _PROJECT_ROOT / "data" / "external" / "bothbosu"
    if subsets is None:
        subsets = [
            "scam_dialogue",
            "multi_agent_scam_conversation",
            "single_agent_scam_conversations",
            "Scammer_Conversation",
        ]
    if splits is None:
        splits = ["test"]

    import pandas as pd
    records: list[CallRecord] = []
    for subset in subsets:
        for split in splits:
            csv_path = root / f"{subset}_{split}.csv"
            if not csv_path.exists():
                alt = root / f"{subset}.csv"
                if alt.exists():
                    csv_path = alt
                else:
                    continue
            df = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in df.columns}
            text_col = (cols.get("dialogue") or cols.get("text")
                        or cols.get("conversation") or df.columns[0])
            label_col = (cols.get("label") or cols.get("is_scam")
                         or cols.get("class") or df.columns[-1])
            type_col = cols.get("type") or cols.get("category")
            id_col = cols.get("id") or cols.get("call_id")

            for i, row in df.iterrows():
                raw_label = str(row[label_col]).strip().lower()
                if raw_label in {"1", "scam", "fraud", "true", "yes"}:
                    y = 1
                elif raw_label in {"0", "non-scam", "non_scam", "legit",
                                   "not_scam", "false", "no", "ham"}:
                    y = 0
                else:
                    try:
                        y = int(float(raw_label))
                    except (ValueError, TypeError):
                        continue

                dialogue = str(row[text_col])
                sentences = _parse_bothbosu_dialogue(dialogue, caller_only=caller_only)
                if not sentences:
                    continue

                records.append(CallRecord(
                    call_id=str(row[id_col]) if id_col else f"bb_{subset}_{split}_{i}",
                    label=y,
                    category=str(row[type_col]) if type_col else ("scam" if y else "legit"),
                    source=f"bothbosu_{subset}",
                    sentences=sentences,
                ))
    return records


def load_youtube_baiters(root: Path | None = None) -> list[CallRecord]:
    """Load the YouTube scam-baiter corpus built per SCAMBAITER_PROTOCOL.md.

    Expects::

        data/external/youtube_baiters/
            annotations.tsv
            transcripts/{call_id}.json   # Whisper output per call

    Rows in annotations.tsv must have:
        call_id, label, category, transcript_file, ...

    Rows with an empty ``label`` are skipped (the script writes empty
    labels for un-annotated calls).
    """
    if root is None:
        root = _PROJECT_ROOT / "data" / "external" / "youtube_baiters"
    anno_path = root / "annotations.tsv"
    if not anno_path.exists():
        return []

    import csv
    records: list[CallRecord] = []
    with open(anno_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            label_raw = (row.get("label") or "").strip()
            if not label_raw:
                continue
            try:
                y = int(label_raw)
            except ValueError:
                continue
            tr_rel = row.get("transcript_file") or f"transcripts/{row['call_id']}.json"
            tr_path = root / tr_rel
            if not tr_path.exists():
                continue
            tr = json.loads(tr_path.read_text())
            # Whisper output has 'segments' with 'text' fields
            all_segments = tr.get("segments") if "segments" in tr else None
            seg_range = (row.get("segment_range") or "").strip()
            if all_segments and seg_range and ":" in seg_range:
                try:
                    a, _, b = seg_range.partition(":")
                    a = int(a) if a else 0
                    b = int(b) if b else len(all_segments)
                    all_segments = all_segments[a:b]
                except ValueError:
                    pass

            if all_segments:
                sentences = []
                for seg in all_segments:
                    txt = seg.get("text", "").strip()
                    if txt:
                        # Each Whisper segment is approximately a sentence
                        sentences.append(txt)
            else:
                # Fall back to 'text' field, segment manually
                sentences = _split_sentences(tr.get("text", ""))
            if not sentences:
                continue
            records.append(CallRecord(
                call_id=row["call_id"],
                label=y,
                category=row.get("category", "unknown"),
                source="youtube_baiters",
                sentences=sentences,
            ))
    return records


# ---------------------------------------------------------------------------
# Convenience: load everything available
# ---------------------------------------------------------------------------

def load_all_call_records() -> dict[str, list[CallRecord]]:
    """Load every call-level dataset that's locally available."""
    return {
        "repo_real": load_repo_real(),
        "better30": load_better30(),
        "wu2024_corpus": load_wu2024(),
        "bothbosu": load_bothbosu(),
        "youtube_baiters": load_youtube_baiters(),
        # Cross-lingual / not part of headline eval:
        "teleantifraud_28k": load_teleantifraud(),
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
