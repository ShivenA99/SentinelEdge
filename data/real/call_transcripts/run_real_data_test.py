#!/usr/bin/env python3
"""Test the SentinelEdge fraud detection pipeline against real-world
scam call transcripts sourced from FTC, IRS, SSA, AARP, FCC, FBI,
and news organizations.

This script:
1. Loads each transcript from data/real/call_transcripts/
2. Parses the metadata header (source, type, label)
3. Runs the transcript through the SentinelEngine using both
   the heuristic pipeline and (if available) the XGBoost pipeline
4. Tracks per-sentence EMA scores and feature activations
5. Reports detailed per-call results and overall accuracy
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel_edge.engine import SentinelEngine
from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.classifier.score_accumulator import ScoreAccumulator
from sentinel_edge.classifier.alert_engine import AlertEngine
from sentinel_edge.audio.sentence_splitter import SentenceSplitter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptMeta:
    filename: str
    source: str
    scam_type: str
    category: str
    label: str  # "FRAUD" or "LEGITIMATE"
    notes: str


@dataclass
class SentenceResult:
    text: str
    raw_score: float
    ema_after: float
    features: dict[str, float]
    active_features: list[str]


@dataclass
class CallResult:
    meta: TranscriptMeta
    sentences: list[SentenceResult]
    final_ema: float
    peak_score: float
    mean_score: float
    risk_level: str
    is_fraud_verdict: bool
    reasons: list[str]
    correctly_classified: bool
    inference_ms: float
    backend: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_transcript_file(filepath: Path) -> tuple[TranscriptMeta, str]:
    """Parse a transcript file, extracting metadata header and body text."""
    lines = filepath.read_text(encoding="utf-8").splitlines()

    meta_fields: dict[str, str] = {}
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            # Parse key: value from comment lines
            content = stripped.lstrip("# ").strip()
            if ":" in content:
                key, _, value = content.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key in ("source", "type", "category", "label", "notes"):
                    meta_fields[key] = value
        elif stripped == "":
            continue
        else:
            body_start = i
            break

    transcript_text = "\n".join(lines[body_start:]).strip()

    meta = TranscriptMeta(
        filename=filepath.name,
        source=meta_fields.get("source", "unknown"),
        scam_type=meta_fields.get("type", "unknown"),
        category=meta_fields.get("category", "unknown"),
        label=meta_fields.get("label", "unknown").upper(),
        notes=meta_fields.get("notes", ""),
    )

    return meta, transcript_text


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_transcript(
    engine: SentinelEngine,
    meta: TranscriptMeta,
    transcript: str,
) -> CallResult:
    """Run a full transcript through the engine, sentence by sentence."""
    t0 = time.perf_counter()

    # Reset call state
    engine.reset_call_state()

    # Split into sentences
    splitter = SentenceSplitter()
    sentences = splitter.feed(transcript)
    leftover = splitter.flush()
    if leftover:
        sentences.append(leftover)

    # If the splitter doesn't produce enough sentences (e.g. no punctuation),
    # fall back to splitting on newlines
    if len(sentences) <= 1:
        sentences = [s.strip() for s in transcript.split("\n") if s.strip()]

    sentence_results: list[SentenceResult] = []
    all_features: dict[str, float] = {}

    for sentence in sentences:
        if not sentence.strip():
            continue

        score, features = engine.analyze_sentence(sentence)
        engine.accumulator.update(score)

        # Track which features are "active" (non-zero and meaningful)
        active = []
        for k, v in features.items():
            if k in ("char_count", "word_count", "avg_word_len"):
                continue  # structural, not indicative
            if k == "caps_ratio" and v < 0.3:
                continue
            if k == "exclamation_count" and v < 2:
                continue
            if v > 0:
                active.append(k)

        sentence_results.append(SentenceResult(
            text=sentence[:120] + ("..." if len(sentence) > 120 else ""),
            raw_score=score,
            ema_after=engine.accumulator.current_score,
            features=dict(features),
            active_features=active,
        ))

        # Merge features
        for k, v in features.items():
            all_features[k] = max(all_features.get(k, 0.0), v)

    final_ema = engine.accumulator.current_score
    peak = engine.accumulator.peak_score
    mean = engine.accumulator.mean_score

    alert = engine.alert_engine.evaluate(final_ema, all_features)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    is_fraud_verdict = final_ema >= engine.threshold
    expected_fraud = meta.label == "FRAUD"
    correctly_classified = is_fraud_verdict == expected_fraud

    return CallResult(
        meta=meta,
        sentences=sentence_results,
        final_ema=final_ema,
        peak_score=peak,
        mean_score=mean,
        risk_level=alert.risk_level.value,
        is_fraud_verdict=is_fraud_verdict,
        reasons=alert.reasons,
        correctly_classified=correctly_classified,
        inference_ms=elapsed_ms,
        backend=engine.active_backend,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_results(results: list[CallResult]) -> str:
    """Generate a comprehensive text report."""
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  SENTINELEDGE REAL-WORLD TRANSCRIPT EVALUATION REPORT")
    lines.append(sep)
    lines.append("")
    lines.append(f"Date: 2026-03-21")
    lines.append(f"Backend: {results[0].backend if results else 'N/A'}")
    lines.append(f"Total transcripts tested: {len(results)}")
    lines.append(f"  Scam transcripts: {sum(1 for r in results if r.meta.label == 'FRAUD')}")
    lines.append(f"  Legitimate transcripts: {sum(1 for r in results if r.meta.label == 'LEGITIMATE')}")
    lines.append("")

    # --- Overall accuracy ---
    correct = sum(1 for r in results if r.correctly_classified)
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    scam_results = [r for r in results if r.meta.label == "FRAUD"]
    legit_results = [r for r in results if r.meta.label == "LEGITIMATE"]

    tp = sum(1 for r in scam_results if r.is_fraud_verdict)  # true positive
    fn = sum(1 for r in scam_results if not r.is_fraud_verdict)  # false negative
    tn = sum(1 for r in legit_results if not r.is_fraud_verdict)  # true negative
    fp = sum(1 for r in legit_results if r.is_fraud_verdict)  # false positive

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    lines.append(sep)
    lines.append("  OVERALL RESULTS SUMMARY")
    lines.append(sep)
    lines.append(f"  Overall Accuracy:  {correct}/{total} ({accuracy:.1f}%)")
    lines.append(f"  True Positives:    {tp} (scams correctly flagged)")
    lines.append(f"  False Negatives:   {fn} (scams MISSED)")
    lines.append(f"  True Negatives:    {tn} (legit calls correctly cleared)")
    lines.append(f"  False Positives:   {fp} (legit calls wrongly flagged)")
    lines.append(f"  Precision:         {precision:.1f}%")
    lines.append(f"  Recall:            {recall:.1f}%")
    lines.append(f"  F1 Score:          {f1:.1f}%")
    lines.append("")

    # --- Per-call details ---
    lines.append(sep)
    lines.append("  DETAILED PER-CALL RESULTS")
    lines.append(sep)

    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"--- Call {i}: {r.meta.filename} ---")
        lines.append(f"  Type:       {r.meta.scam_type}")
        lines.append(f"  Category:   {r.meta.category}")
        lines.append(f"  Source:     {r.meta.source[:80]}...")
        lines.append(f"  True Label: {r.meta.label}")
        lines.append(f"  Verdict:    {'FRAUD' if r.is_fraud_verdict else 'LEGITIMATE'}")
        lines.append(f"  Correct:    {'YES' if r.correctly_classified else '*** NO ***'}")
        lines.append(f"  Final EMA:  {r.final_ema:.4f}")
        lines.append(f"  Peak Score: {r.peak_score:.4f}")
        lines.append(f"  Mean Score: {r.mean_score:.4f}")
        lines.append(f"  Risk Level: {r.risk_level}")
        lines.append(f"  Sentences:  {len(r.sentences)}")
        lines.append(f"  Latency:    {r.inference_ms:.1f} ms")
        lines.append(f"  Reasons:    {', '.join(r.reasons) if r.reasons else '(none)'}")

        # Show sentence-level EMA progression
        lines.append(f"  Score Progression:")
        for j, s in enumerate(r.sentences):
            marker = ""
            if s.raw_score >= 0.5:
                marker = " <<<< HIGH"
            elif s.raw_score >= 0.3:
                marker = " << MEDIUM"
            lines.append(
                f"    [{j+1:2d}] raw={s.raw_score:.4f}  ema={s.ema_after:.4f}"
                f"  features=[{', '.join(s.active_features)}]{marker}"
            )

        lines.append("")

    # --- Feature activation analysis ---
    lines.append(sep)
    lines.append("  FEATURE ACTIVATION ANALYSIS")
    lines.append(sep)
    lines.append("")

    # Count how often each feature fires across scam vs legit calls
    feature_names = [
        "urgency_count", "action_count", "financial_count",
        "impersonation_count", "has_url", "has_shortened_url",
        "has_verify_pattern", "has_threat", "has_prize",
        "has_account_ref", "dollar_sign", "has_phone_number",
        "exclamation_count", "caps_ratio",
    ]

    lines.append(f"  {'Feature':<25s} {'Scam Calls':>12s} {'Legit Calls':>12s} {'Discriminative?':>16s}")
    lines.append(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*16}")

    for feat in feature_names:
        scam_fires = 0
        legit_fires = 0
        for r in results:
            max_val = 0.0
            for s in r.sentences:
                val = s.features.get(feat, 0.0)
                if feat in ("caps_ratio",):
                    val = 1.0 if val >= 0.3 else 0.0
                elif feat == "exclamation_count":
                    val = 1.0 if val >= 2 else 0.0
                max_val = max(max_val, val)
            if max_val > 0:
                if r.meta.label == "FRAUD":
                    scam_fires += 1
                else:
                    legit_fires += 1

        scam_pct = scam_fires / len(scam_results) * 100 if scam_results else 0
        legit_pct = legit_fires / len(legit_results) * 100 if legit_results else 0
        diff = scam_pct - legit_pct
        disc = "YES" if diff > 30 else ("weak" if diff > 10 else "NO")
        lines.append(
            f"  {feat:<25s} {scam_fires:>4d} ({scam_pct:5.1f}%)"
            f" {legit_fires:>4d} ({legit_pct:5.1f}%) {disc:>14s}"
        )

    lines.append("")

    # --- Misclassified calls ---
    misclassified = [r for r in results if not r.correctly_classified]
    lines.append(sep)
    lines.append("  MISCLASSIFICATION ANALYSIS")
    lines.append(sep)
    lines.append("")

    if not misclassified:
        lines.append("  No misclassifications! All calls were correctly classified.")
    else:
        lines.append(f"  {len(misclassified)} call(s) misclassified:")
        lines.append("")
        for r in misclassified:
            lines.append(f"  FILE: {r.meta.filename}")
            lines.append(f"    True label: {r.meta.label}")
            lines.append(f"    Verdict:    {'FRAUD' if r.is_fraud_verdict else 'LEGITIMATE'}")
            lines.append(f"    Final EMA:  {r.final_ema:.4f}")
            lines.append(f"    Risk Level: {r.risk_level}")
            if r.meta.label == "FRAUD" and not r.is_fraud_verdict:
                lines.append(f"    ANALYSIS: Scam call was MISSED. The heuristic scoring")
                lines.append(f"      was too conservative. Key scam indicators may not have")
                lines.append(f"      matched the keyword lists or the EMA smoothing diluted")
                lines.append(f"      high-scoring sentences with benign ones.")
                # Check what features were present
                all_feats: set[str] = set()
                for s in r.sentences:
                    all_feats.update(s.active_features)
                lines.append(f"    Features that DID fire: {', '.join(sorted(all_feats))}")
                missing = set(feature_names) - all_feats
                lines.append(f"    Features that did NOT fire: {', '.join(sorted(missing))}")
            elif r.meta.label == "LEGITIMATE" and r.is_fraud_verdict:
                lines.append(f"    ANALYSIS: Legitimate call was wrongly flagged. False")
                lines.append(f"      positive caused by feature overlap with scam patterns.")
                all_feats = set()
                for s in r.sentences:
                    all_feats.update(s.active_features)
                lines.append(f"    Features that fired: {', '.join(sorted(all_feats))}")
            lines.append("")

    # --- Brutally honest assessment ---
    lines.append(sep)
    lines.append("  HONEST ASSESSMENT OF PIPELINE PERFORMANCE")
    lines.append(sep)
    lines.append("")

    lines.append("  STRENGTHS:")
    if recall >= 80:
        lines.append(f"  + High recall ({recall:.1f}%) - most scam calls are detected")
    if precision >= 80:
        lines.append(f"  + High precision ({precision:.1f}%) - few false alarms on legit calls")
    if fn == 0:
        lines.append(f"  + Zero false negatives - no scam calls slipped through")
    if fp == 0:
        lines.append(f"  + Zero false positives - no legitimate calls wrongly flagged")

    # Check if threat/urgency features work well
    threat_tp = sum(1 for r in scam_results
                    if any("has_threat" in s.active_features for s in r.sentences)
                    and r.is_fraud_verdict)
    if threat_tp > 0:
        lines.append(f"  + Threat detection feature fired on {threat_tp}/{len(scam_results)} scam calls")

    imp_tp = sum(1 for r in scam_results
                 if any("impersonation_count" in s.active_features for s in r.sentences))
    if imp_tp > 0:
        lines.append(f"  + Impersonation detection fired on {imp_tp}/{len(scam_results)} scam calls")

    lines.append("")
    lines.append("  WEAKNESSES:")

    if fn > 0:
        lines.append(f"  - {fn} scam call(s) MISSED (false negatives)")
        lines.append(f"    This is the most dangerous failure mode for a fraud detector.")
    if fp > 0:
        lines.append(f"  - {fp} legitimate call(s) wrongly flagged (false positives)")
        lines.append(f"    Users will learn to ignore alerts if too many are false.")

    # Check if heuristic scores are generally too low
    scam_emas = [r.final_ema for r in scam_results]
    legit_emas = [r.final_ema for r in legit_results]
    avg_scam_ema = sum(scam_emas) / len(scam_emas) if scam_emas else 0
    avg_legit_ema = sum(legit_emas) / len(legit_emas) if legit_emas else 0

    lines.append(f"  - Average scam EMA: {avg_scam_ema:.4f} (want > 0.5)")
    lines.append(f"  - Average legit EMA: {avg_legit_ema:.4f} (want < 0.3)")

    separation = avg_scam_ema - avg_legit_ema
    lines.append(f"  - Score separation (scam avg - legit avg): {separation:.4f}")
    if separation < 0.3:
        lines.append(f"    WARNING: Low score separation means the threshold cannot cleanly")
        lines.append(f"    separate scams from legitimate calls. Need better features or ML model.")

    if results[0].backend == "heuristic":
        lines.append(f"  - Running in HEURISTIC-ONLY mode (no trained model loaded).")
        lines.append(f"    The heuristic scorer uses simple keyword matching with capped weights.")
        lines.append(f"    A trained XGBoost or MLP classifier would likely perform much better.")
        lines.append(f"    The engine looked for 'xgb_model.json' and 'tfidf.joblib' but the")
        lines.append(f"    actual model files are named 'call_fraud_xgb.json' and")
        lines.append(f"    'tfidf_call_vectorizer.pkl'. Fix the file naming to enable ML scoring.")

    # EMA smoothing commentary
    lines.append("")
    lines.append("  EMA SMOOTHING IMPACT:")
    for r in scam_results:
        if r.peak_score >= 0.5 and r.final_ema < 0.5:
            lines.append(
                f"  - {r.meta.filename}: Peak={r.peak_score:.4f} but "
                f"EMA={r.final_ema:.4f}. EMA smoothing DILUTED the signal."
            )

    lines.append("")
    lines.append("  RECOMMENDATIONS:")
    lines.append("  1. Fix model file naming (xgb_model.json -> call_fraud_xgb.json)")
    lines.append("     so the XGBoost/MLP classifiers actually load.")
    lines.append("  2. The heuristic _heuristic_text_score() caps each feature contribution")
    lines.append("     too aggressively. Consider raising the caps or using multiplicative")
    lines.append("     scoring for compounding indicators.")
    lines.append("  3. Add more impersonation keywords: 'badge', 'warrant', 'officer',")
    lines.append("     'department', 'investigation', 'court', 'judge', 'legal'.")
    lines.append("  4. Add payment-method scam keywords: 'gift card', 'prepaid', 'MoneyPak',")
    lines.append("     'wire transfer', 'Western Union', 'Bitcoin', 'Zelle'.")
    lines.append("  5. Consider a 'secrecy' feature for phrases like 'don't tell anyone',")
    lines.append("     'gag order', 'confidential', 'do not contact'.")
    lines.append("  6. The EMA alpha=0.3 may be too aggressive for long transcripts where")
    lines.append("     early benign sentences dilute later scam content.")
    lines.append("  7. Train on real scam transcripts like these for better generalization.")
    lines.append("")

    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    transcript_dir = Path(__file__).parent
    output_path = transcript_dir / "real_data_results.txt"

    # Collect transcript files
    transcript_files = sorted(
        p for p in transcript_dir.glob("*.txt")
        if p.name.startswith(("scam_", "legit_"))
    )

    if not transcript_files:
        print("ERROR: No transcript files found!")
        sys.exit(1)

    print(f"Found {len(transcript_files)} transcript files")

    # Try multiple pipelines and report on each.
    # We test heuristic (always available) and xgboost (if models load).
    # The MLP pipeline has a known numpy 2.x compatibility issue in
    # mlp_classifier.py line 86 so we skip it for now.
    pipelines_to_test = []

    # 1. Heuristic (always works)
    engine_heuristic = SentinelEngine(
        models_dir=str(PROJECT_ROOT / "models"),
        threshold=0.5,
        alpha=0.3,
        pipeline="xgboost",  # will fall back to heuristic if xgb model name doesn't match
    )
    pipelines_to_test.append(("heuristic", engine_heuristic))
    print(f"Heuristic engine backend: {engine_heuristic.active_backend}")

    # 2. Try XGBoost with correct model path (symlink or direct load)
    # The engine looks for xgb_model.json but files are call_fraud_xgb.json
    # Let's create symlinks to make the xgboost pipeline load
    import shutil
    models_dir = PROJECT_ROOT / "models"
    xgb_link = models_dir / "xgb_model.json"
    tfidf_link = models_dir / "tfidf.joblib"

    created_links = []
    if not xgb_link.exists() and (models_dir / "call_fraud_xgb.json").exists():
        os.symlink(models_dir / "call_fraud_xgb.json", xgb_link)
        created_links.append(xgb_link)
    if not tfidf_link.exists() and (models_dir / "tfidf_call_vectorizer.pkl").exists():
        os.symlink(models_dir / "tfidf_call_vectorizer.pkl", tfidf_link)
        created_links.append(tfidf_link)

    try:
        engine_xgb = SentinelEngine(
            models_dir=str(PROJECT_ROOT / "models"),
            threshold=0.5,
            alpha=0.3,
            pipeline="xgboost",
        )
        if engine_xgb.active_backend == "xgboost":
            pipelines_to_test.append(("xgboost", engine_xgb))
            print(f"XGBoost engine loaded successfully!")
        else:
            print(f"XGBoost engine fell back to: {engine_xgb.active_backend}")
    except Exception as e:
        print(f"XGBoost engine failed to load: {e}")

    # Use the best available engine for primary results
    engine = pipelines_to_test[-1][1]
    print(f"\nPrimary engine for report: {engine.active_backend}")

    # Run analysis with ALL backends
    all_backend_results: dict[str, list[CallResult]] = {}

    for backend_name, eng in pipelines_to_test:
        print(f"\n--- Running with {backend_name} backend ---")
        results: list[CallResult] = []
        for filepath in transcript_files:
            meta, transcript = parse_transcript_file(filepath)
            print(f"  Analyzing: {filepath.name} ({meta.label})...", end=" ")

            result = analyze_transcript(eng, meta, transcript)
            results.append(result)

            verdict = "FRAUD" if result.is_fraud_verdict else "LEGIT"
            status = "OK" if result.correctly_classified else "WRONG"
            print(
                f"EMA={result.final_ema:.4f} "
                f"verdict={verdict} "
                f"[{status}]"
            )

        all_backend_results[backend_name] = results

        correct = sum(1 for r in results if r.correctly_classified)
        print(f"  {backend_name}: {correct}/{len(results)} correctly classified "
              f"({correct/len(results)*100:.1f}%)")

    # Use the best backend for the detailed report
    primary_backend = list(all_backend_results.keys())[-1]
    primary_results = all_backend_results[primary_backend]

    # Generate report
    report_parts = []
    report_parts.append(format_results(primary_results))

    # Add multi-backend comparison if we have more than one
    if len(all_backend_results) > 1:
        report_parts.append("")
        report_parts.append("=" * 80)
        report_parts.append("  MULTI-BACKEND COMPARISON")
        report_parts.append("=" * 80)
        report_parts.append("")

        for bname, bresults in all_backend_results.items():
            scam_r = [r for r in bresults if r.meta.label == "FRAUD"]
            legit_r = [r for r in bresults if r.meta.label == "LEGITIMATE"]
            tp = sum(1 for r in scam_r if r.is_fraud_verdict)
            fn = sum(1 for r in scam_r if not r.is_fraud_verdict)
            tn = sum(1 for r in legit_r if not r.is_fraud_verdict)
            fp = sum(1 for r in legit_r if r.is_fraud_verdict)
            correct = tp + tn
            total = len(bresults)
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            avg_scam = sum(r.final_ema for r in scam_r) / len(scam_r) if scam_r else 0
            avg_legit = sum(r.final_ema for r in legit_r) / len(legit_r) if legit_r else 0

            report_parts.append(f"  Backend: {bname}")
            report_parts.append(f"    Accuracy:    {correct}/{total} ({correct/total*100:.1f}%)")
            report_parts.append(f"    TP={tp}  FN={fn}  TN={tn}  FP={fp}")
            report_parts.append(f"    Precision={prec:.1f}%  Recall={rec:.1f}%  F1={f1:.1f}%")
            report_parts.append(f"    Avg scam EMA:  {avg_scam:.4f}")
            report_parts.append(f"    Avg legit EMA: {avg_legit:.4f}")
            report_parts.append("")

            # Per-call comparison for this backend
            for r in bresults:
                v = "FRAUD" if r.is_fraud_verdict else "LEGIT"
                c = "OK" if r.correctly_classified else "MISS"
                report_parts.append(
                    f"    {r.meta.filename:<40s}  EMA={r.final_ema:.4f}  "
                    f"verdict={v:5s}  [{c}]  risk={r.risk_level}"
                )
            report_parts.append("")

    report = "\n".join(report_parts)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    # Clean up symlinks we created
    for link in created_links:
        if link.is_symlink():
            link.unlink()
            print(f"Cleaned up symlink: {link}")

    # Print final summary
    correct = sum(1 for r in primary_results if r.correctly_classified)
    print(f"\nFinal ({primary_backend}): {correct}/{len(primary_results)} correctly classified "
          f"({correct/len(primary_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
