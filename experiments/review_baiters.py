"""Interactive review tool for ``data/external/youtube_baiters/annotations.tsv``.

Walks through each row, shows the Whisper transcript paginated, and
lets you correct the ``label``, ``scammer_speaker``, and a new
``segment_range`` field (start:end indices) that tells the loader
which Whisper segments are the scam call (vs. baiter commentary,
intro music, outros, etc.).

Whisper-tiny does not diarize, so ``scammer_speaker`` is best left
blank and you instead mark the SEGMENT RANGE that contains the
scammer's speech. The loader scores only segments inside that range.

Run from the repo root::

    python experiments/review_baiters.py
    python experiments/review_baiters.py --start-at yt_005    # resume

Per-call commands (compound input OK, e.g. "1r"):

  1 / 0   set label = 1 (scam) / 0 (legit)
  c / r   set scammer_speaker = caller / receiver
  R       prompt for segment range (start:end indices)
  m       show more transcript segments (paginate)
  n       next call (advances cursor)
  b       previous call
  s       skip without changes
  q       save and quit
  ?       help

On first run a backup is written to ``annotations.tsv.bak``. Every
modification is saved immediately so a crash doesn't lose work.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _PROJECT_ROOT / "data" / "external" / "youtube_baiters"

# ANSI colours (best effort)
def _c(text, code): return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text
def bold(s):  return _c(s, "1")
def green(s): return _c(s, "32")
def red(s):   return _c(s, "31")
def cyan(s):  return _c(s, "36")
def dim(s):   return _c(s, "2")


# ---------------------------------------------------------------------------
# TSV I/O
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "call_id", "label", "category", "youtube_url",
    "start_sec", "end_sec", "scammer_speaker",
    "transcript_file", "notes",
]
# We append this column on first save if absent
EXTRA_COL = "segment_range"


def load_tsv(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)
    header, body = rows[0], rows[1:]
    if EXTRA_COL not in header:
        header.append(EXTRA_COL)
        for r in body:
            r.append("")
    # Pad short rows to header length
    body = [r + [""] * (len(header) - len(r)) for r in body]
    return header, body


def save_tsv(path: Path, header, body):
    with path.open("w", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for r in body:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Transcript display
# ---------------------------------------------------------------------------

def load_transcript(rel_path: str, root: Path) -> list[dict]:
    if not rel_path:
        return []
    p = (root / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data.get("segments", [])


def fmt_seg(i: int, seg: dict, max_w: int = 110) -> str:
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", 0.0))
    t = (seg.get("text") or "").strip()
    if len(t) > max_w:
        t = t[: max_w - 1] + "…"
    return f"{dim(f'{i:>3}')}  {dim(f'[{s:6.1f}-{e:6.1f}]')}  {t}"


def show_call(idx, total, row, header, segs, page_start, page_size):
    H = {h: i for i, h in enumerate(header)}
    print("\n" + "=" * 72)
    print(bold(f"[{idx+1}/{total}] {row[H['call_id']]}") +
          f"   category={row[H['category']]}")
    print(f"   url:        {row[H['youtube_url']]}")
    print(f"   transcript: {row[H['transcript_file']]} "
          f"({len(segs)} segments)")
    lbl = row[H['label']].strip()
    lbl_disp = green("1=scam") if lbl == "1" else (
        red("0=legit") if lbl == "0" else dim("(unlabeled)"))
    print(f"   label:      {lbl_disp}    "
          f"scammer_speaker: {row[H['scammer_speaker']] or dim('(none)')}    "
          f"range: {row[H[EXTRA_COL]] or dim('full')}")
    print("-" * 72)
    if not segs:
        print(dim("  (no transcript segments available)"))
        return
    page_end = min(page_start + page_size, len(segs))
    for i in range(page_start, page_end):
        print(fmt_seg(i, segs[i]))
    if page_end < len(segs):
        print(dim(f"   ... {len(segs) - page_end} more segments (press 'm')"))


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

HELP = """Commands (chain any combination, e.g. '1rn'):
  1 / 0   set label
  c / r   set scammer_speaker = caller / receiver
  R       set segment range (prompts for start:end)
  m       paginate transcript (show next chunk)
  n       next call
  b       previous call
  s       skip without changes
  q       save and quit
  ?       this help
"""


def apply(cmd: str, row, header, segs, state) -> tuple[bool, bool]:
    """Apply a command sequence. Returns (advance, quit)."""
    H = {h: i for i, h in enumerate(header)}
    advance = False
    for ch in cmd.strip():
        if ch == "1":
            row[H["label"]] = "1"; print(green("  -> label=1"))
        elif ch == "0":
            row[H["label"]] = "0"; print(red("  -> label=0"))
        elif ch == "c":
            row[H["scammer_speaker"]] = "caller"; print("  -> scammer=caller")
        elif ch == "r":
            row[H["scammer_speaker"]] = "receiver"; print("  -> scammer=receiver")
        elif ch == "R":
            try:
                rng = input(cyan(f"     range start:end (current {row[H[EXTRA_COL]] or 'full'}) > ")).strip()
                if rng:
                    a, _, b = rng.partition(":")
                    a, b = int(a or 0), int(b or len(segs))
                    row[H[EXTRA_COL]] = f"{a}:{b}"
                    print(f"  -> range={a}:{b}")
            except (ValueError, EOFError):
                print(red("  -> range invalid, unchanged"))
        elif ch == "m":
            state["page_start"] += state["page_size"]
            if state["page_start"] >= len(segs):
                state["page_start"] = 0
                print(dim("  (wrapping to top)"))
            state["redraw"] = True
        elif ch == "n":
            advance = True
        elif ch == "b":
            state["back"] = True; advance = True
        elif ch == "s":
            advance = True
        elif ch == "q":
            return False, True
        elif ch in (" ", "\t"):
            pass
        elif ch == "?":
            print(HELP)
        else:
            print(red(f"  ? unknown command '{ch}'"))
    return advance, False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_DEFAULT_ROOT))
    ap.add_argument("--page-size", type=int, default=12)
    ap.add_argument("--start-at", default=None,
                    help="Resume at this call_id (skip earlier rows).")
    args = ap.parse_args()

    root = Path(args.root)
    tsv = root / "annotations.tsv"
    if not tsv.exists():
        print(f"[fatal] {tsv} not found.", file=sys.stderr); return 1

    bak = tsv.with_suffix(".tsv.bak")
    if not bak.exists():
        shutil.copy2(tsv, bak); print(dim(f"[backup] {bak}"))

    header, body = load_tsv(tsv)
    print(dim(f"[load] {len(body)} rows from {tsv}"))
    print(dim("Type '?' at the prompt for help."))

    idx = 0
    if args.start_at:
        for i, r in enumerate(body):
            if r[0] == args.start_at:
                idx = i; break

    while 0 <= idx < len(body):
        row = body[idx]
        segs = load_transcript(row[header.index("transcript_file")], root)
        state = {"page_start": 0, "page_size": args.page_size,
                 "back": False, "redraw": True}
        while True:
            if state["redraw"]:
                show_call(idx, len(body), row, header, segs,
                          state["page_start"], state["page_size"])
                state["redraw"] = False
            try:
                cmd = input(bold("> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(); save_tsv(tsv, header, body); return 0
            if not cmd:
                continue
            advance, quit_ = apply(cmd, row, header, segs, state)
            save_tsv(tsv, header, body)   # incremental save
            if quit_:
                print(dim("[saved] " + str(tsv))); return 0
            if advance:
                idx = idx - 1 if state["back"] else idx + 1
                break

    print(dim("[done] reached end of TSV"))
    save_tsv(tsv, header, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
