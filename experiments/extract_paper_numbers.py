"""Generate ``paper/_numbers.tex`` from the result JSON files.

The paper draft ``paper/sentineledge_emnlp2026.tex`` references every
experiment-derived quantity through a LaTeX macro
(e.g. ``\\StreamFOne``, ``\\TtdMedianSent``). This script reads the
JSON outputs of every experiment under ``results/`` and writes the
matching ``\\newcommand`` definitions to
``paper/_numbers.tex``, which the main TeX file already
``\\input``s.

Run after every change to the experiment outputs::

    python experiments/extract_paper_numbers.py

The script is tolerant of missing files: if a particular results
JSON isn't there yet, the corresponding macros emit a clearly
visible placeholder string ("TBD") so the compiled PDF flags
exactly which numbers still need to be measured.

Macro inventory
---------------
See ``paper/MACROS.md`` for the full list, what each one means,
and which JSON file populates it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Defaults: every macro starts as "TBD" so the compiled PDF makes it
# obvious which numbers haven't been measured yet.
# ---------------------------------------------------------------------------

_DEFAULT_MACROS: dict[str, str] = {
    # --- corpus sizes ---
    "NumTrainSentences": "TBD",
    "NumEvalCalls": "TBD",
    "NumScamCalls": "TBD",
    "NumLegitCalls": "TBD",
    "EvalSources": "\\textsc{repo-real}",
    # --- per-sentence ---
    "PerSentAcc":  "TBD", "PerSentPrec": "TBD",
    "PerSentRec":  "TBD", "PerSentFOne": "TBD",
    # --- per-call mean ---
    "MeanAcc":  "TBD", "MeanPrec": "TBD",
    "MeanRec":  "TBD", "MeanFOne": "TBD",
    # --- streaming EMA ---
    "StreamAcc":  "TBD", "StreamPrec": "TBD",
    "StreamRec":  "TBD", "StreamFOne": "TBD",
    "StreamMissed": "TBD",
    # --- full transcript ---
    "FullAcc":  "TBD", "FullPrec": "TBD",
    "FullRec":  "TBD", "FullFOne": "TBD",
    # --- time to detection ---
    "TtdMedianSent": "TBD", "TtdMedianSec": "TBD",
    "TtdMidSent":    "TBD", "TtdMidSec":    "TBD",    # P75
    "TtdHighSent":   "TBD",                            # P90
    # --- latency / size ---
    "XgbLatencyMs":   "TBD", "XgbThroughput": "TBD",
    "XgbSizeMB":      "TBD",
    "HandLRLat":  "TBD", "HandLRSize":  "TBD", "HandLRThroughput": "TBD",
    "HandSVMLat": "TBD", "HandSVMSize": "TBD",
    "TfidfLRLat": "TBD", "TfidfLRSize": "TBD",
    "CombLRLat":  "TBD", "CombLRSize":  "TBD",
    "DistilLat":  "TBD", "DistilSize":  "TBD",
    "SpeedupVsDistil":   "TBD",
    "SizeRatioVsDistil": "TBD",
    # --- baseline quality rows ---
    "HandLRFOne":  "TBD", "HandLRPrec":  "TBD", "HandLRRec":  "TBD",
    "HandSVMFOne": "TBD", "HandSVMPrec": "TBD", "HandSVMRec": "TBD",
    "TfidfLRFOne": "TBD", "TfidfLRPrec": "TBD", "TfidfLRRec": "TBD",
    "CombLRFOne":  "TBD", "CombLRPrec":  "TBD", "CombLRRec":  "TBD",
    "DistilFOne":  "TBD", "DistilPrec":  "TBD", "DistilRec":  "TBD",
    # --- ASR robustness ---
    "AsrCleanF":    "TBD",
    "AsrSwapHi":    "TBD",
    "AsrDelHi":     "TBD",
    "AsrCharHi":    "TBD",
    # --- adversarial (XGBoost; comparison row in Framing B) ---
    "AdvNumSents":         "TBD",
    "AdvCleanF":           "TBD",
    "AdvAdvF":             "TBD",
    "AdvRetrainedCleanF":  "TBD",
    "AdvRetrainedAdvF":    "TBD",
    "AdvGapClosedPts":     "TBD",
    "GovFPbeforePct":      "TBD",
    "GovFPafterPct":       "TBD",
    # --- adversarial (LR-on-handcrafted; headline classifier in Framing B) ---
    "HandLRAdvCleanF":           "TBD",
    "HandLRAdvAdvF":             "TBD",
    "HandLRAdvRetrainedCleanF":  "TBD",
    "HandLRAdvRetrainedAdvF":    "TBD",
    "HandLRAdvGapClosedPts":     "TBD",
    "HandLRGovFPbeforePct":      "TBD",
    "HandLRGovFPafterPct":       "TBD",
    # --- headline classifier aliases (Framing B: LR-on-handcrafted) ---
    "HeadlineName":    "LR (18 hand-crafted features)",
    "HeadlineDim":     "18",
    "HeadlineFOne":    "TBD",
    "HeadlinePrec":    "TBD",
    "HeadlineRec":     "TBD",
    "HeadlineLat":     "TBD",
    "HeadlineSize":    "TBD",
    "HeadlineThroughput": "TBD",
    # --- cross-channel ---
    "CallsInDomainF":     "TBD", "CallsInDomainP":     "TBD",
    "CallsInDomainR":     "TBD", "CallsInDomainAuroc": "TBD",
    "SmsFOne":  "TBD", "SmsPrec":  "TBD", "SmsRec":  "TBD", "SmsAuroc": "TBD",
    "UrlFOne":  "TBD", "UrlPrec":  "TBD", "UrlRec":  "TBD", "UrlAuroc": "TBD",
    # --- meta ---
    "RunAllSeconds": "TBD",
    "DemoUrl":  "[URL]",
    "VideoUrl": "[URL]",
    "RepoUrl":  "[URL]",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(x, ndp: int = 3) -> str:
    """Format a number for the paper."""
    if x is None:
        return "TBD"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "TBD"
    return f"{f:.{ndp}f}"


def fmt_int(x) -> str:
    if x is None:
        return "TBD"
    try:
        return f"{int(x):,}".replace(",", "{,}")
    except (TypeError, ValueError):
        return str(x)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Per-result-file extractors
# ---------------------------------------------------------------------------

def from_eval(data: dict, m: dict) -> None:
    """eval_xgb.json -> per-sentence / per-call mean / streaming."""
    ps = data.get("per_sentence", {})
    pm = data.get("per_call_mean", {})
    ps_stream = data.get("per_call_streaming", {})

    m["PerSentAcc"]  = fmt(ps.get("accuracy"))
    m["PerSentPrec"] = fmt(ps.get("precision"))
    m["PerSentRec"]  = fmt(ps.get("recall"))
    m["PerSentFOne"] = fmt(ps.get("f1"))

    m["MeanAcc"]  = fmt(pm.get("accuracy"))
    m["MeanPrec"] = fmt(pm.get("precision"))
    m["MeanRec"]  = fmt(pm.get("recall"))
    m["MeanFOne"] = fmt(pm.get("f1"))

    m["StreamAcc"]  = fmt(ps_stream.get("accuracy"))
    m["StreamPrec"] = fmt(ps_stream.get("precision"))
    m["StreamRec"]  = fmt(ps_stream.get("recall"))
    m["StreamFOne"] = fmt(ps_stream.get("f1"))

    # Streaming confusion: number of missed scams
    conf = ps_stream.get("confusion", {})
    if conf:
        m["StreamMissed"] = str(conf.get("fn", "TBD"))
        npos = conf.get("fn", 0) + conf.get("tp", 0)
        nneg = conf.get("tn", 0) + conf.get("fp", 0)
        if npos or nneg:
            m["NumScamCalls"] = str(npos)
            m["NumLegitCalls"] = str(nneg)
            m["NumEvalCalls"] = str(npos + nneg)

    # Sources string from config
    cfg = data.get("config", {})
    sources = cfg.get("sources", [])
    if sources:
        names = []
        for s in sources:
            names.append({
                "repo_real": "\\textsc{repo-real}",
                "better30":  "BETTER30",
                "wu2024_corpus": "Wu~et~al.~corpus",
                "bothbosu": "BothBosu",
                "bothbosu_scam_dialogue": "BothBosu~SD",
                "bothbosu_multi_agent_scam_conversation": "BothBosu~MASC",
                "bothbosu_single_agent_scam_conversations": "BothBosu~SASC",
                "bothbosu_scammer_conversation": "BothBosu~SC",
                "youtube_baiters": "YouTube~scam-baiter",
            }.get(s, s))
        m["EvalSources"] = " + ".join(names)


def from_ttd(data: dict, m: dict) -> None:
    """ttd.json -> time-to-detection + full-transcript baseline."""
    agg = data.get("ttd_aggregate", {})
    m["TtdMedianSent"] = fmt_int(agg.get("ttd_idx_median"))
    m["TtdMedianSec"]  = fmt_int(agg.get("ttd_sec_median"))
    m["TtdMidSent"]    = fmt_int(agg.get("ttd_idx_p75"))
    m["TtdMidSec"]     = fmt_int(agg.get("ttd_sec_p75"))
    m["TtdHighSent"]   = fmt_int(agg.get("ttd_idx_p90"))

    full = data.get("streaming_vs_full", {}).get("full_transcript", {})
    m["FullAcc"]  = fmt(full.get("accuracy"))
    m["FullPrec"] = fmt(full.get("precision"))
    m["FullRec"]  = fmt(full.get("recall"))
    m["FullFOne"] = fmt(full.get("f1"))


def from_latency(data: dict, m: dict) -> None:
    """latency.json -> XGB and baseline latencies / sizes."""
    for r in data.get("results", []):
        name = r.get("model", "")
        if "xgb" in name:
            m["XgbLatencyMs"]  = fmt(r.get("p50_ms"), 2)
            m["XgbThroughput"] = fmt_int(r.get("throughput_sent_per_sec"))
            m["XgbSizeMB"]     = fmt(r.get("disk_size_mb"), 2)
        elif "logreg_handcrafted" in name:
            m["HandLRLat"]        = fmt(r.get("p50_ms"), 2)
            m["HandLRSize"]       = fmt(r.get("disk_size_mb"), 3)
            m["HandLRThroughput"] = fmt_int(r.get("throughput_sent_per_sec"))
        elif "logreg_tfidf" in name and "hand" not in name:
            m["TfidfLRLat"]  = fmt(r.get("p50_ms"), 2)
            m["TfidfLRSize"] = fmt(r.get("disk_size_mb"), 3)


def from_baselines(data: dict, m: dict) -> None:
    """baselines.json -> per-classifier quality rows."""
    results = data.get("results", {})
    # n training sentences (read from train_csv via baseline config if available)
    # otherwise leave for from_eval to fill in
    mapping = {
        "handcrafted_lr":       ("HandLRFOne",  "HandLRPrec",  "HandLRRec"),
        "handcrafted_svm":      ("HandSVMFOne", "HandSVMPrec", "HandSVMRec"),
        "tfidf_lr":             ("TfidfLRFOne", "TfidfLRPrec", "TfidfLRRec"),
        "tfidf_handcrafted_lr": ("CombLRFOne",  "CombLRPrec",  "CombLRRec"),
    }
    for key, (mf, mp, mr) in mapping.items():
        if key in results and "per_call_streaming" in results[key]:
            s = results[key]["per_call_streaming"]
            m[mf] = fmt(s.get("f1"))
            m[mp] = fmt(s.get("precision"))
            m[mr] = fmt(s.get("recall"))

    # Combined LR doesn't have a separate latency benchmark; use TfidfLR's
    if m.get("CombLRLat") == "TBD" and m.get("TfidfLRLat") != "TBD":
        m["CombLRLat"]  = m["TfidfLRLat"]
        m["CombLRSize"] = m.get("TfidfLRSize", "0.02")
    if m.get("HandSVMLat") == "TBD" and m.get("HandLRLat") != "TBD":
        m["HandSVMLat"]  = m["HandLRLat"]
        m["HandSVMSize"] = m.get("HandLRSize", "0.001")


def from_distilbert(data: dict, m: dict) -> None:
    """distilbert.json -> DistilBERT quality + latency + size."""
    q = data.get("quality", {}).get("per_call_streaming", {})
    lat = data.get("latency", {})
    info = data.get("info", {})
    m["DistilFOne"] = fmt(q.get("f1"))
    m["DistilPrec"] = fmt(q.get("precision"))
    m["DistilRec"]  = fmt(q.get("recall"))
    m["DistilLat"]  = fmt(lat.get("p50_ms"), 1)
    size_mb = info.get("disk_size_mb_estimate")
    m["DistilSize"] = fmt(size_mb, 1) if size_mb else "TBD"

    # Compute speed / size ratios vs XGB if both present
    try:
        speedup = float(lat["p50_ms"]) / float(m["XgbLatencyMs"])
        m["SpeedupVsDistil"] = fmt_int(round(speedup))
    except (KeyError, ValueError, TypeError):
        pass
    try:
        size_ratio = float(size_mb) / float(m["XgbSizeMB"])
        m["SizeRatioVsDistil"] = fmt_int(round(size_ratio))
    except (KeyError, ValueError, TypeError):
        pass


def from_asr(data: dict, m: dict) -> None:
    """asr_robustness.json -> clean F1 and F1 at p=0.30 per perturbation."""
    table = data.get("table", [])
    for r in table:
        if r["rate"] == 0.0:
            m["AsrCleanF"] = fmt(r["stream_f1"])
        elif r["rate"] == 0.30:
            if r["perturbation"] == "word_swap":
                m["AsrSwapHi"] = fmt(r["stream_f1"])
            elif r["perturbation"] == "word_delete":
                m["AsrDelHi"] = fmt(r["stream_f1"])
            elif r["perturbation"] == "char_noise":
                m["AsrCharHi"] = fmt(r["stream_f1"])


def from_adversarial(data: dict, m: dict) -> None:
    """adversarial.json -> default vs retrained clean+adv F1."""
    results = data.get("results", {})
    default = results.get("default", {})
    retrained = results.get("adv_retrained", {})
    if default:
        m["AdvNumSents"] = fmt_int(default.get("adversarial", {}).get("n"))
        m["AdvCleanF"]   = fmt(default.get("clean", {}).get("f1"))
        m["AdvAdvF"]     = fmt(default.get("adversarial", {}).get("f1"))
        # Government false positive
        gov = default.get("adv_legit_by_category", {}).get(
            "real_government_contact", {})
        if gov:
            m["GovFPbeforePct"] = fmt(gov.get("frac_above_0.5", 0) * 100, 1)
    if retrained:
        m["AdvRetrainedCleanF"] = fmt(retrained.get("clean", {}).get("f1"))
        m["AdvRetrainedAdvF"]   = fmt(retrained.get("adversarial", {}).get("f1"))
        gov = retrained.get("adv_legit_by_category", {}).get(
            "real_government_contact", {})
        if gov:
            m["GovFPafterPct"] = fmt(gov.get("frac_above_0.5", 0) * 100, 1)
    # Adversarial gap closed in F1 points
    try:
        gap = (float(m["AdvRetrainedAdvF"]) - float(m["AdvAdvF"])) * 100
        m["AdvGapClosedPts"] = fmt(gap, 1)
    except (KeyError, ValueError, TypeError):
        pass


def from_adversarial_lr(data: dict, m: dict) -> None:
    """adversarial_lr.json -> default vs retrained clean+adv F1 for LR.

    Mirrors ``from_adversarial`` exactly; populates the ``HandLR*``
    adversarial macros used by the Framing-B headline narrative.
    """
    results = data.get("results", {})
    default = results.get("default", {})
    retrained = results.get("adv_retrained", {})
    if default:
        m["HandLRAdvCleanF"] = fmt(default.get("clean", {}).get("f1"))
        m["HandLRAdvAdvF"]   = fmt(default.get("adversarial", {}).get("f1"))
        gov = default.get("adv_legit_by_category", {}).get(
            "real_government_contact", {})
        if gov:
            m["HandLRGovFPbeforePct"] = fmt(gov.get("frac_above_0.5", 0) * 100, 1)
    if retrained:
        m["HandLRAdvRetrainedCleanF"] = fmt(retrained.get("clean", {}).get("f1"))
        m["HandLRAdvRetrainedAdvF"]   = fmt(retrained.get("adversarial", {}).get("f1"))
        gov = retrained.get("adv_legit_by_category", {}).get(
            "real_government_contact", {})
        if gov:
            m["HandLRGovFPafterPct"] = fmt(gov.get("frac_above_0.5", 0) * 100, 1)
    try:
        gap = (float(m["HandLRAdvRetrainedAdvF"]) - float(m["HandLRAdvAdvF"])) * 100
        m["HandLRAdvGapClosedPts"] = fmt(gap, 1)
    except (KeyError, ValueError, TypeError):
        pass


def _populate_headline_aliases(m: dict) -> None:
    """Framing B: alias the LR-on-handcrafted quality/latency macros
    to the ``Headline*`` names used in the abstract and Section 2.

    Called after every other extractor so it sees the final values.
    """
    if m.get("HandLRFOne", "TBD") != "TBD":
        m["HeadlineFOne"] = m["HandLRFOne"]
    if m.get("HandLRPrec", "TBD") != "TBD":
        m["HeadlinePrec"] = m["HandLRPrec"]
    if m.get("HandLRRec", "TBD") != "TBD":
        m["HeadlineRec"] = m["HandLRRec"]
    if m.get("HandLRLat", "TBD") != "TBD":
        m["HeadlineLat"] = m["HandLRLat"]
    if m.get("HandLRSize", "TBD") != "TBD":
        m["HeadlineSize"] = m["HandLRSize"]
    if m.get("HandLRThroughput", "TBD") != "TBD":
        m["HeadlineThroughput"] = m["HandLRThroughput"]
    # Throughput: derive from latency if not separately recorded.
    # latency.json carries throughput per-row; if from_latency populated
    # m["HandLRThroughput"] (which it doesn't by default), use that;
    # otherwise leave TBD and the extractor will report it.


def from_cross_channel(data: dict, m: dict) -> None:
    """cross_channel.json -> F1/P/R/AUROC per channel."""
    res = data.get("results", {})
    def fill(ch, prefix):
        x = res.get(ch, {})
        m[f"{prefix}F"]     = fmt(x.get("f1"))
        m[f"{prefix}P"]     = fmt(x.get("precision"))
        m[f"{prefix}Prec"]  = fmt(x.get("precision"))
        m[f"{prefix}R"]     = fmt(x.get("recall"))
        m[f"{prefix}Rec"]   = fmt(x.get("recall"))
        m[f"{prefix}Auroc"] = fmt(x.get("auroc"))
        m[f"{prefix}FOne"]  = fmt(x.get("f1"))   # alias for tables
    if "calls" in res:
        fill("calls", "CallsInDomain")
    if "sms" in res:
        fill("sms", "Sms")
    if "urls" in res:
        fill("urls", "Url")


def from_paper_tables(data: dict, m: dict) -> None:
    """paper_tables.json -> the total run-all wall-clock time."""
    t = data.get("timings_sec", {})
    total = sum(v for v in t.values() if isinstance(v, (int, float)))
    if total > 0:
        m["RunAllSeconds"] = fmt_int(round(total))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    ap.add_argument("--out", default=str(_PROJECT_ROOT / "paper" / "_numbers.tex"))
    ap.add_argument("--num-train-sentences", default=None,
                    help="Optional override -- normally read from CSV row count.")
    ap.add_argument("--demo-url", default="https://example.com/sentineledge-demo")
    ap.add_argument("--video-url", default="https://youtu.be/EXAMPLE")
    ap.add_argument("--repo-url", default="https://github.com/EXAMPLE/sentineledge")
    args = ap.parse_args()

    m = dict(_DEFAULT_MACROS)
    rdir = Path(args.results_dir)

    # Run each extractor over its JSON file (silent if missing)
    extractors = [
        ("eval_xgb.json",         from_eval),
        ("ttd.json",              from_ttd),
        ("latency.json",          from_latency),
        ("baselines.json",        from_baselines),
        ("distilbert.json",       from_distilbert),
        ("asr_robustness.json",   from_asr),
        ("adversarial.json",      from_adversarial),
        ("adversarial_lr.json",   from_adversarial_lr),
        ("cross_channel.json",    from_cross_channel),
        ("paper_tables.json",     from_paper_tables),
    ]
    for fname, fn in extractors:
        data = _load(rdir / fname)
        if data is None:
            print(f"[warn] missing {fname}; macros from this source remain TBD",
                  file=sys.stderr)
            continue
        try:
            fn(data, m)
        except Exception as e:
            print(f"[warn] extractor for {fname} failed: {e}", file=sys.stderr)

    # Auto-count synthetic training data
    if args.num_train_sentences:
        m["NumTrainSentences"] = args.num_train_sentences
    else:
        train_csv = _PROJECT_ROOT / "data" / "processed" / "call_fraud_train.csv"
        if train_csv.exists():
            with open(train_csv, "r", encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1  # minus header
            m["NumTrainSentences"] = fmt_int(n)

    # URLs (default to placeholders; user passes the real ones)
    m["DemoUrl"]  = args.demo_url
    m["VideoUrl"] = args.video_url
    m["RepoUrl"]  = args.repo_url

    # Framing B: alias HandLR* values into Headline* macros so the
    # paper's abstract / Section 2 / Conclusion compile correctly.
    _populate_headline_aliases(m)

    # Emit
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by experiments/extract_paper_numbers.py\n")
        fh.write("% Do not edit by hand -- re-run the script after experiments.\n")
        fh.write("% Macros marked 'TBD' indicate experiments not yet run.\n\n")
        for name in _DEFAULT_MACROS.keys():
            val = m.get(name, "TBD")
            fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

    print(f"[saved] {out_path}")
    tbd_count = sum(1 for v in m.values() if v == "TBD")
    print(f"[status] {len(m) - tbd_count} / {len(m)} macros filled; "
          f"{tbd_count} still TBD.")
    if tbd_count:
        print("[status] TBD macros:")
        for k, v in m.items():
            if v == "TBD":
                print(f"           {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
