"""Download YouTube scam-baiter videos and transcribe with Whisper.

This script automates Steps 3 of the collection protocol described
in ``experiments/SCAMBAITER_PROTOCOL.md``. It does NOT do annotation;
that step is necessarily manual.

Pipeline per video:
  1. Read video_list.tsv (URL + timing + category).
  2. Use yt-dlp to download audio as MP3, trim to [start_sec, end_sec].
  3. Run Whisper-tiny.en for per-sentence transcription with timestamps.
  4. Write one JSON file per call to transcripts/.
  5. Append a stub row to annotations.tsv (you fill in label/scammer_speaker).

The script is fully idempotent: re-running skips any video whose
transcript JSON already exists, so you can stop and resume.

Requirements
------------
    pip install yt-dlp openai-whisper       # (or faster-whisper)
    # ffmpeg available on PATH

Usage
-----
    # 1. Create video_list.tsv (see SCAMBAITER_PROTOCOL.md)
    # 2. Run:
    python experiments/collect_youtube_scambaiters.py \\
        --video-list data/external/youtube_baiters/video_list.tsv \\
        --output-dir data/external/youtube_baiters/ \\
        --whisper-model tiny.en

The script will print a per-video progress line and exit when every
row in video_list.tsv has been processed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Imports gated behind availability
# ---------------------------------------------------------------------------

def _import_whisper(model_name: str):
    """Return a callable that maps mp3 path -> Whisper result dict."""
    # Prefer faster-whisper (3x faster on CPU), fall back to openai-whisper.
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        def transcribe(audio_path: str) -> dict:
            segments, info = model.transcribe(
                audio_path, language="en", vad_filter=True,
            )
            seg_list = []
            for s in segments:
                seg_list.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text.strip(),
                })
            full_text = " ".join(s["text"] for s in seg_list)
            return {"text": full_text, "segments": seg_list,
                    "language": info.language, "backend": "faster-whisper"}
        return transcribe, "faster-whisper"
    except ImportError:
        pass

    try:
        import whisper
        model = whisper.load_model(model_name)
        def transcribe(audio_path: str) -> dict:
            r = model.transcribe(audio_path, language="en", verbose=False)
            return {
                "text": r["text"],
                "segments": [
                    {"start": float(s["start"]), "end": float(s["end"]),
                     "text": s["text"].strip()}
                    for s in r.get("segments", [])
                ],
                "language": r.get("language", "en"),
                "backend": "openai-whisper",
            }
        return transcribe, "openai-whisper"
    except ImportError:
        pass

    print("[fatal] need either 'faster-whisper' or 'openai-whisper' installed.",
          file=sys.stderr)
    print("Install with: pip install faster-whisper", file=sys.stderr)
    sys.exit(2)


def _have_yt_dlp() -> bool:
    return subprocess.call(
        ["yt-dlp", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


def _have_ffmpeg() -> bool:
    return subprocess.call(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def download_audio(url: str, out_path: Path) -> bool:
    """Download a video's audio as MP3 using yt-dlp.

    Returns True on success.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(out_path.with_suffix("")) + ".%(ext)s",
        url,
    ]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path.exists() and rc == 0


def trim_audio(src: Path, dst: Path, start_sec: float, end_sec: float) -> bool:
    """Trim an MP3 to a [start, end] range using ffmpeg."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(src),
        "-c", "copy",
        str(dst),
    ]
    rc = subprocess.call(cmd)
    return dst.exists() and rc == 0


def process_video(row: dict, out_dir: Path, transcribe_fn) -> dict | None:
    """Download + trim + transcribe a single row from video_list.tsv.

    Returns the transcript dict on success, None on failure.
    """
    vid = row["video_id"]
    url = row["youtube_url"]
    start_sec = float(row.get("start_sec") or 0.0)
    end_sec = float(row.get("end_sec") or 0.0)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = out_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    raw_audio = audio_dir / f"{vid}_full.mp3"
    clip_audio = audio_dir / f"{vid}.mp3"
    transcript_path = transcript_dir / f"{vid}.json"

    if transcript_path.exists():
        return json.loads(transcript_path.read_text())

    print(f"  [download] {vid}  {url}")
    if not download_audio(url, raw_audio):
        print(f"    [fail] download failed for {vid}", file=sys.stderr)
        return None

    if end_sec > 0:
        print(f"  [trim]     {vid}  {start_sec:.0f}-{end_sec:.0f}s")
        if not trim_audio(raw_audio, clip_audio, start_sec, end_sec):
            print(f"    [fail] trim failed for {vid}", file=sys.stderr)
            return None
    else:
        # No trim requested: use full audio
        clip_audio = raw_audio

    print(f"  [whisper]  {vid}")
    t0 = time.perf_counter()
    result = transcribe_fn(str(clip_audio))
    elapsed = time.perf_counter() - t0
    print(f"    transcribed in {elapsed:.1f}s; {len(result['segments'])} segments")

    # Attach metadata
    result["video_id"] = vid
    result["youtube_url"] = url
    result["clip_range_sec"] = [start_sec, end_sec]
    result["category"] = row.get("category", "unknown")
    result["transcribe_elapsed_sec"] = elapsed

    transcript_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# annotations.tsv stub management
# ---------------------------------------------------------------------------

ANNO_HEADER = [
    "call_id", "label", "category", "youtube_url",
    "start_sec", "end_sec", "scammer_speaker",
    "transcript_file", "notes",
]


def stub_annotation_row(row: dict, transcript_rel: str) -> list:
    return [
        row["video_id"],
        "",                # label -- to fill in
        row.get("category", ""),
        row["youtube_url"],
        str(row.get("start_sec", "")),
        str(row.get("end_sec", "")),
        "",                # scammer_speaker -- to fill in
        transcript_rel,
        "",                # notes
    ]


def update_annotations(out_dir: Path, row: dict) -> None:
    anno_path = out_dir / "annotations.tsv"
    transcript_rel = f"transcripts/{row['video_id']}.json"

    existing_ids: set[str] = set()
    if anno_path.exists():
        with open(anno_path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader, None)
            for r in reader:
                if r:
                    existing_ids.add(r[0])
    else:
        with open(anno_path, "w", encoding="utf-8") as fh:
            fh.write("\t".join(ANNO_HEADER) + "\n")

    if row["video_id"] in existing_ids:
        return
    with open(anno_path, "a", encoding="utf-8") as fh:
        fh.write("\t".join(stub_annotation_row(row, transcript_rel)) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video-list", required=True,
        help="TSV with columns: video_id, youtube_url, start_sec, end_sec, category, notes",
    )
    ap.add_argument(
        "--output-dir",
        default="data/external/youtube_baiters",
    )
    ap.add_argument("--whisper-model", default="tiny.en",
                    help="Whisper model name (tiny.en is recommended).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N videos (0 = all).")
    args = ap.parse_args()

    if not _have_yt_dlp():
        print("[fatal] yt-dlp not on PATH. Install: pip install yt-dlp",
              file=sys.stderr)
        return 2
    if not _have_ffmpeg():
        print("[fatal] ffmpeg not on PATH.", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[whisper] loading {args.whisper_model} ...")
    transcribe_fn, backend = _import_whisper(args.whisper_model)
    print(f"[whisper] backend = {backend}")

    rows = []
    with open(args.video_list, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            if r.get("video_id"):
                rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[plan] {len(rows)} videos to process")

    n_ok, n_fail, n_skip = 0, 0, 0
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] {row['video_id']}")
        try:
            result = process_video(row, out_dir, transcribe_fn)
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
            n_fail += 1
            continue
        if result is None:
            n_fail += 1
            continue
        update_annotations(out_dir, row)
        n_ok += 1

    print(f"\n[done] ok={n_ok}  fail={n_fail}  skipped={n_skip}")
    print(f"[done] next step: open {out_dir / 'annotations.tsv'} and fill in")
    print(f"       the 'label', 'scammer_speaker', and 'notes' columns for each row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
