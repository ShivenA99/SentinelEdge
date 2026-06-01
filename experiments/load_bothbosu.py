"""Loader for the BothBosu HuggingFace datasets (Gumphusiri 2024).

Paste this function into ``experiments/dataset_loader.py`` and add
``load_bothbosu`` to the ``_LOADERS`` registry in each experiment
script (the same one-line addition pattern as ``load_better30``).

The BothBosu collection -- ``scam-dialogue``,
``multi-agent-scam-conversation``, ``single-agent-scam-conversations``,
``Scammer-Conversation`` -- is the publicly-available "SC / SD / MASC"
data that Shen et al. 2024 (arXiv:2409.11643) cite as their
synthesised evaluation set. Released under Apache 2.0.

CSV format expected (after running the download script in
EXPERIMENTS_TO_RUN.md):

    data/external/bothbosu/
        scam_dialogue_train.csv
        scam_dialogue_test.csv
        multi_agent_scam_conversation_train.csv
        ...

Each CSV has columns ``dialogue``, ``type``, ``label`` (some variants
also have ``conversation_id`` or similar). The loader is tolerant of
column-name variations.

Two important methodological points:

  * The ``dialogue`` field is a single string with explicit
    ``caller:`` and ``receiver:`` turn markers. We split on those
    markers to recover the per-utterance sequence. Each utterance
    becomes a "sentence" in the SentinelEdge pipeline.

  * The scammer is annotated as ``caller`` (Suspect) and the victim
    as ``receiver`` (Innocent). For a more realistic evaluation
    matching deployment (where the user receives a call), set
    ``caller_only=True`` to score only scammer utterances. Default
    (``False``) scores both speakers, matching the original
    Gumphusiri evaluation setup that Shen et al. compare against.
"""
from __future__ import annotations

import re
from pathlib import Path

# Patch into dataset_loader.py's namespace -- it provides CallRecord
# and _split_sentences. For a standalone import you can copy those
# definitions here.
from experiments.dataset_loader import CallRecord, _split_sentences


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _PROJECT_ROOT / "data" / "external" / "bothbosu"

_TURN_PATTERN = re.compile(
    r"(caller|receiver)\s*:\s*(.+?)(?=\s+(?:caller|receiver)\s*:|$)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_dialogue(dialogue: str, caller_only: bool = False) -> list[str]:
    """Parse the caller:/receiver: dialogue string into utterances.

    Returns a list of sentence-level strings (one per turn, further
    split on punctuation if a single turn contains multiple sentences).
    """
    if not dialogue or not isinstance(dialogue, str):
        return []

    turns = _TURN_PATTERN.findall(dialogue)
    if not turns:
        # Fallback: no caller:/receiver: markers -- just sentence-split
        return _split_sentences(dialogue)

    sentences: list[str] = []
    for speaker, utterance in turns:
        if caller_only and speaker.lower() != "caller":
            continue
        # Each utterance may contain multiple sentences; preserve order.
        sentences.extend(_split_sentences(utterance.strip()))
    return sentences


def load_bothbosu(
    root: Path | None = None,
    subsets: list[str] | None = None,
    splits: list[str] | None = None,
    caller_only: bool = False,
) -> list[CallRecord]:
    """Load any subset of the BothBosu HuggingFace datasets.

    Args:
        root:         Override the data directory.
        subsets:      Names like ``"scam_dialogue"``. Default = all four.
        splits:       ``["train", "test"]`` or a single split. Default = test only.
        caller_only:  If True, score only scammer utterances. Default
                      False matches Gumphusiri / Shen et al.

    Returns a list of ``CallRecord``s with ``source`` field set to
    ``bothbosu_<subset>`` for traceable per-subset breakdown.
    """
    import pandas as pd

    if root is None:
        root = _DEFAULT_ROOT
    if subsets is None:
        subsets = [
            "scam_dialogue",
            "multi_agent_scam_conversation",
            "single_agent_scam_conversations",
            "Scammer_Conversation",
        ]
    if splits is None:
        splits = ["test"]  # default: use test split only, not train

    records: list[CallRecord] = []
    for subset in subsets:
        for split in splits:
            csv_path = root / f"{subset}_{split}.csv"
            if not csv_path.exists():
                # Try alt: subset.csv without split suffix
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
                # Normalise label
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
                sentences = _parse_dialogue(dialogue, caller_only=caller_only)
                if not sentences:
                    continue

                call_id = (
                    str(row[id_col]) if id_col
                    else f"bothbosu_{subset}_{split}_{i}"
                )
                category = (
                    str(row[type_col]) if type_col else
                    ("scam" if y else "legit")
                )
                records.append(CallRecord(
                    call_id=call_id,
                    label=y,
                    category=category,
                    source=f"bothbosu_{subset}",
                    sentences=sentences,
                ))
    return records


# ---------------------------------------------------------------------------
# CLI: quick summary of what's available
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    recs = load_bothbosu()
    if not recs:
        print("No BothBosu CSVs found. Run the download script first; "
              "see EXPERIMENTS_TO_RUN.md for the snippet.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Total BothBosu records (test split, both speakers): {len(recs)}")
    sources = Counter(r.source for r in recs)
    print(f"\nBy source:")
    for s, n in sorted(sources.items()):
        scams = sum(1 for r in recs if r.source == s and r.label == 1)
        legit = n - scams
        sents = [len(r.sentences) for r in recs if r.source == s]
        median_sents = sorted(sents)[len(sents) // 2] if sents else 0
        print(f"  {s:<45} n={n:>5}  scam={scams:>5}  legit={legit:>5}  "
              f"median_sents/call={median_sents}")

    cats = Counter(r.category for r in recs)
    print(f"\nTop categories:")
    for c, n in cats.most_common(15):
        print(f"  {c:<30} {n}")
