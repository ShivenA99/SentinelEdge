"""Per-stage latency breakdown for the full on-device pipeline.

The headline classifier latency (~0.11 ms) is only the final stage of a
longer pipeline. Quoting it on its own reads as an end-to-end number,
which it is not. This harness times *each* stage independently with
``time.perf_counter()`` so the paper can show exactly where the
wall-clock goes:

    1. ASR        Whisper-tiny transcription of one audio window
    2. segment    SentenceSplitter over the window text
    3. features   FeaturePipeline.extract per sentence (518-dim)
    4. classify   classifier.predict_proba per sentence
    5. ema        ScoreAccumulator.update per sentence

Each stage is warmed up, then timed over many repetitions; the median
per-call latency is reported (CPU pinned to one thread to model the
on-device single-core case). Stages whose dependencies are missing
(Whisper not installed, no ``--audio`` file, xgboost absent) are
*skipped and clearly marked* so the emitted table shows precisely what
was and was not measured -- no silent substitution of a different model
for the classifier number.

The stage callables are deliberately small and self-contained; to swap
in a different ASR engine, sentence splitter, or feature extractor,
edit the corresponding ``_build_*_stage`` factory below -- the timing
core does not need to change.

Outputs
-------
  * a markdown table to stdout
  * ``results/latency_breakdown.json`` (full per-stage numbers)
  * ``paper/_latency.tex`` (macros: ``\\LatWhisperMs``, ``\\LatSegMs``,
    ``\\LatFeatMs``, ``\\LatClassMs``, ``\\LatEmaMs``, ``\\LatTotalMs``,
    plus ``\\LatWindowSec`` and ``\\LatClassModel``)

To use the macros, add ``\\input{_latency}`` next to the existing
``\\input{_numbers}`` line once this script has been run on an
environment that has Whisper + xgboost installed.

Usage
-----
    # Full breakdown (needs a ~30 s WAV and Whisper installed):
    python experiments/run_latency_breakdown.py --audio sample_30s.wav

    # Text-only stages (skips ASR when no audio / Whisper):
    python experiments/run_latency_breakdown.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pin thread counts BEFORE importing anything that touches BLAS, so the
# numbers reflect the single-core on-device scenario.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Sample text (fallback when the real corpus loader is unavailable)
# ---------------------------------------------------------------------------

_FALLBACK_SENTENCES: list[str] = [
    "Hello, this is the IRS calling about your unpaid tax balance.",
    "We need you to confirm your social security number to proceed.",
    "Your account has been suspended due to suspicious activity.",
    "Please purchase gift cards and read me the numbers on the back.",
    "Thank you for calling, your appointment is confirmed for Tuesday.",
    "I am calling to schedule the plumber for your kitchen repair.",
    "There is a warrant out for your arrest unless you pay immediately.",
    "This is your bank's fraud department, did you authorise this charge?",
]


def _load_sentences(n: int) -> list[str]:
    """Return *n* real sentences from the repo corpus, or fallbacks."""
    try:
        from experiments.dataset_loader import load_repo_real  # noqa: E402

        records = load_repo_real()
        texts: list[str] = []
        for rec in records:
            text = getattr(rec, "text", None) or getattr(rec, "transcript", "")
            for part in str(text).replace("\n", " ").split("."):
                part = part.strip()
                if len(part) > 12:
                    texts.append(part + ".")
        if texts:
            reps = (n // len(texts)) + 1
            return (texts * reps)[:n]
    except Exception as exc:  # pragma: no cover - corpus optional
        print(f"[info] repo corpus unavailable ({exc}); using fallback text",
              file=sys.stderr)
    reps = (n // len(_FALLBACK_SENTENCES)) + 1
    return (_FALLBACK_SENTENCES * reps)[:n]


# ---------------------------------------------------------------------------
# Timing core
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """One timed pipeline stage.

    ``fn`` is a zero-argument callable that performs exactly one unit of
    work (one sentence, or one audio window for ASR). ``unit`` documents
    what a single call represents in the emitted table.
    """

    key: str
    label: str
    macro: str
    fn: Callable[[], object] | None
    unit: str = "sentence"
    available: bool = True
    skip_reason: str = ""
    samples_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.samples_ms) if self.samples_ms else None


def _time_stage(stage: Stage, *, n_warm: int, n_time: int, reps: int) -> None:
    """Warm up, then time ``stage.fn`` and record median per-call ms/rep."""
    if stage.fn is None or not stage.available:
        return
    fn = stage.fn
    for _ in range(n_warm):
        fn()
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n_time):
            fn()
        elapsed = time.perf_counter() - t0
        stage.samples_ms.append((elapsed / n_time) * 1000.0)


# ---------------------------------------------------------------------------
# Stage factories -- swap implementations here, not in the timing core.
# ---------------------------------------------------------------------------

def _build_asr_stage(audio_path: str | None, window_sec: float,
                     model_name: str) -> Stage:
    """Whisper-tiny transcription of a single ``window_sec`` audio window."""
    stage = Stage(key="asr", label=f"ASR (Whisper {model_name})",
                  macro="LatWhisperMs", fn=None, unit=f"{window_sec:g}s window")
    if not audio_path:
        stage.available = False
        stage.skip_reason = "no --audio file provided"
        return stage
    if not Path(audio_path).exists():
        stage.available = False
        stage.skip_reason = f"audio file not found: {audio_path}"
        return stage
    try:
        import whisper  # type: ignore[import-untyped]

        from sentinel_edge.audio.transcriber import Transcriber

        audio = whisper.load_audio(audio_path)  # 16 kHz mono float32
        win = int(window_sec * 16_000)
        if audio.shape[0] < win:
            window = np.pad(audio, (0, win - audio.shape[0]))
        else:
            window = audio[:win]
        transcriber = Transcriber(model_name=model_name)
        transcriber.load()  # exclude one-time model load from the timed loop
        stage.fn = lambda: transcriber.transcribe(window, sample_rate=16_000)
    except Exception as exc:
        stage.available = False
        stage.skip_reason = f"Whisper unavailable ({type(exc).__name__}: {exc})"
    return stage


def _build_segment_stage(sentences: list[str]) -> Stage:
    """SentenceSplitter over one window's worth of transcript text."""
    stage = Stage(key="segment", label="Segmentation",
                  macro="LatSegMs", fn=None, unit="window")
    try:
        from sentinel_edge.audio.sentence_splitter import SentenceSplitter

        # A window of text ~= a few sentences fed at once.
        chunk = " ".join(sentences[:3]) + " "
        counter = {"i": 0}

        def _run() -> object:
            splitter = SentenceSplitter()
            out = splitter.feed(chunk)
            counter["i"] += 1
            return out

        stage.fn = _run
    except Exception as exc:
        stage.available = False
        stage.skip_reason = f"SentenceSplitter unavailable ({exc})"
    return stage


def _build_feature_stage(sentences: list[str], tfidf_path: str) -> Stage:
    """FeaturePipeline.extract for one sentence (518-dim TF-IDF mode)."""
    stage = Stage(key="features", label="Feature extraction",
                  macro="LatFeatMs", fn=None, unit="sentence")
    try:
        from sentinel_edge.features.feature_pipeline import FeaturePipeline

        tp = tfidf_path if Path(tfidf_path).exists() else None
        if tp is None:
            print(f"[info] TF-IDF vectorizer not found at {tfidf_path}; "
                  "feature timing still valid (zero TF-IDF block)",
                  file=sys.stderr)
        pipeline = FeaturePipeline(tfidf_path=tp, mode="tfidf")
        idx = {"i": 0}
        n = len(sentences)

        def _run() -> object:
            s = sentences[idx["i"] % n]
            idx["i"] += 1
            return pipeline.extract(s)

        stage.fn = _run
    except Exception as exc:
        stage.available = False
        stage.skip_reason = f"FeaturePipeline unavailable ({exc})"
    return stage


def _build_classify_stage(sentences: list[str], tfidf_path: str,
                          model_path: str) -> tuple[Stage, str]:
    """classifier.predict_proba for one 518-dim feature vector.

    Returns the stage and a human-readable model label. We deliberately
    use the paper's actual classifier (XGBoost JSON via FraudClassifier);
    if xgboost is unavailable the stage is skipped rather than silently
    timing a different model.
    """
    stage = Stage(key="classify", label="Classifier",
                  macro="LatClassMs", fn=None, unit="sentence")
    model_label = Path(model_path).name
    try:
        from sentinel_edge.classifier.xgb_classifier import FraudClassifier
        from sentinel_edge.features.feature_pipeline import FeaturePipeline

        if not Path(model_path).exists():
            stage.available = False
            stage.skip_reason = f"model not found: {model_path}"
            return stage, model_label

        tp = tfidf_path if Path(tfidf_path).exists() else None
        pipeline = FeaturePipeline(tfidf_path=tp, mode="tfidf")
        # Pre-compute feature vectors so only inference is timed.
        feats = [pipeline.extract(s) for s in sentences]
        classifier = FraudClassifier(model_path)
        idx = {"i": 0}
        n = len(feats)

        def _run() -> object:
            v = feats[idx["i"] % n]
            idx["i"] += 1
            return classifier.predict_proba(v)

        stage.fn = _run
    except Exception as exc:
        stage.available = False
        stage.skip_reason = f"classifier unavailable ({type(exc).__name__}: {exc})"
    return stage, model_label


def _build_ema_stage() -> Stage:
    """ScoreAccumulator.update for one per-sentence score."""
    stage = Stage(key="ema", label="EMA smoothing",
                  macro="LatEmaMs", fn=None, unit="sentence")
    try:
        from sentinel_edge.classifier.score_accumulator import ScoreAccumulator

        acc = ScoreAccumulator(alpha=0.3)
        rng = np.random.default_rng(0)
        scores = rng.random(512).tolist()
        idx = {"i": 0}

        def _run() -> object:
            s = scores[idx["i"] % len(scores)]
            idx["i"] += 1
            return acc.update(s)

        stage.fn = _run
    except Exception as exc:
        stage.available = False
        stage.skip_reason = f"ScoreAccumulator unavailable ({exc})"
    return stage


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    if x < 0.01:
        return f"{x:.4f}"
    if x < 1.0:
        return f"{x:.3f}"
    if x < 10.0:
        return f"{x:.2f}"
    return f"{x:.1f}"


def print_markdown(stages: list[Stage], window_sec: float) -> None:
    print("\n## Per-stage latency breakdown")
    print("| Stage | Unit | Median latency (ms) | Status |")
    print("|---|---|---|---|")
    for st in stages:
        status = "measured" if st.median_ms is not None else f"skipped — {st.skip_reason}"
        print(f"| {st.label} | {st.unit} | {_fmt_ms(st.median_ms)} | {status} |")


def write_json(path: Path, stages: list[Stage], window_sec: float,
               model_label: str) -> None:
    payload = {
        "window_sec": window_sec,
        "classifier_model": model_label,
        "stages": {
            st.key: {
                "label": st.label,
                "unit": st.unit,
                "median_ms": st.median_ms,
                "available": st.median_ms is not None,
                "skip_reason": st.skip_reason,
                "samples_ms": st.samples_ms,
            }
            for st in stages
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


def write_tex(path: Path, stages: list[Stage], window_sec: float,
              model_label: str) -> None:
    lines = [
        "% Auto-generated by experiments/run_latency_breakdown.py",
        "% Per-stage pipeline latency. '—' marks a stage not measured",
        "% in the environment that produced this file.",
        "",
        f"\\newcommand{{\\LatWindowSec}}{{{window_sec:g}}}",
        f"\\newcommand{{\\LatClassModel}}{{{model_label}}}",
    ]
    measured = [st.median_ms for st in stages if st.median_ms is not None]
    for st in stages:
        val = _fmt_ms(st.median_ms)
        lines.append(f"\\newcommand{{\\{st.macro}}}{{{val}}}")
    total = sum(measured) if measured else None
    lines.append(f"\\newcommand{{\\LatTotalMs}}{{{_fmt_ms(total)}}}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", default=None,
                    help="Path to a ~30 s WAV for the ASR stage "
                         "(skipped if omitted or Whisper unavailable).")
    ap.add_argument("--window-sec", type=float, default=5.0,
                    help="ASR window length in seconds.")
    ap.add_argument("--whisper-model", default="tiny.en")
    ap.add_argument("--model",
                    default=str(_PROJECT_ROOT / "models" / "call_fraud_xgb.json"),
                    help="Classifier model for the classify stage.")
    ap.add_argument("--tfidf",
                    default=str(_PROJECT_ROOT / "models" / "tfidf_call_vectorizer.pkl"))
    ap.add_argument("--n-warm", type=int, default=50)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--asr-reps", type=int, default=1,
                    help="Repetitions for the (slow) ASR stage.")
    ap.add_argument("--asr-n-time", type=int, default=3,
                    help="Timed windows per ASR repetition.")
    ap.add_argument("--out-json", type=Path,
                    default=_PROJECT_ROOT / "results" / "latency_breakdown.json")
    ap.add_argument("--out-tex", type=Path,
                    default=_PROJECT_ROOT / "paper" / "_latency.tex")
    args = ap.parse_args()

    sentences = _load_sentences(max(args.n_time, 64))

    asr_stage = _build_asr_stage(args.audio, args.window_sec, args.whisper_model)
    seg_stage = _build_segment_stage(sentences)
    feat_stage = _build_feature_stage(sentences, args.tfidf)
    clf_stage, model_label = _build_classify_stage(sentences, args.tfidf, args.model)
    ema_stage = _build_ema_stage()
    stages = [asr_stage, seg_stage, feat_stage, clf_stage, ema_stage]

    print(f"timing stages (n_warm={args.n_warm}, n_time={args.n_time}, "
          f"reps={args.reps})...")
    # ASR is far slower; time it with its own (smaller) budget.
    _time_stage(asr_stage, n_warm=1, n_time=args.asr_n_time, reps=args.asr_reps)
    for st in (seg_stage, feat_stage, clf_stage, ema_stage):
        _time_stage(st, n_warm=args.n_warm, n_time=args.n_time, reps=args.reps)

    print_markdown(stages, args.window_sec)
    write_json(args.out_json, stages, args.window_sec, model_label)
    write_tex(args.out_tex, stages, args.window_sec, model_label)

    skipped = [st.label for st in stages if st.median_ms is None]
    if skipped:
        print(f"\n[note] {len(skipped)} stage(s) skipped: {', '.join(skipped)}. "
              "Re-run on an environment with Whisper + xgboost installed "
              "for the complete table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
