"""Generate paper figures from the result JSON files.

Produces four PDF + PNG figures, sized for two-column EMNLP layout:

    fig1_pareto.pdf       -- F1 vs latency Pareto, models annotated
    fig2_ttd_cdf.pdf      -- Empirical CDF of time-to-detection
                             (sentences, with seconds on second axis)
    fig3_first_n.pdf      -- Streaming F1 as function of N sentences listened to
    fig4_asr_robustness.pdf  -- Streaming F1 vs ASR error rate, three perturbations

All plots use a colourblind-safe palette and 7 pt sans-serif labels so
they read cleanly when scaled to a column.

Usage
-----
    python experiments/make_figures.py
    python experiments/make_figures.py --outdir paper/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Colour-blind-safe (Wong palette excerpts)
_PALETTE = {
    "xgb":          "#0072B2",  # blue
    "tfidf_lr":     "#E69F00",  # orange
    "hand_lr":      "#009E73",  # green
    "hand_svm":     "#56B4E9",  # sky
    "combined_lr":  "#CC79A7",  # magenta
    "distilbert":   "#D55E00",  # vermilion
    "claude":       "#F0E442",  # yellow
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [save] {outdir / name}.pdf")


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [skip] missing {path}")
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Figure 1: F1 vs latency Pareto
# ---------------------------------------------------------------------------

def fig_pareto(latency: dict, baselines: dict, outdir: Path) -> None:
    """Streaming F1 (y) vs per-sentence latency (x, log scale)."""
    # Bring latency and F1 together by model name (best-effort matching).
    by_lat = {r["model"]: r for r in latency.get("results", []) if "p50_ms" in r}
    bl = baselines.get("results", {})

    # Mapping from baseline key -> latency key
    pair = [
        ("trained_xgb",          "xgb_tfidf_518d",        "XGBoost + TF-IDF", _PALETTE["xgb"], "o", 44),
        ("tfidf_handcrafted_lr", None,                    "LR (TF-IDF + hand-crafted)", _PALETTE["combined_lr"], "o", 44),
        ("tfidf_lr",             "logreg_tfidf_500d",     "LR (TF-IDF only)", _PALETTE["tfidf_lr"], "o", 44),
        ("handcrafted_lr",       "logreg_handcrafted_18d","LR (18 hand-crafted, ours)", _PALETTE["hand_lr"], "*", 110),
        ("handcrafted_svm",      None,                    "SVM (18 hand-crafted only)", _PALETTE["hand_svm"], "o", 44),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for bl_key, lat_key, label, colour, marker, size in pair:
        if bl_key not in bl:
            continue
        f1 = bl[bl_key]["per_call_streaming"]["f1"]
        # Pull latency if we have it; otherwise approximate from XGB
        if lat_key and lat_key in by_lat:
            lat_ms = by_lat[lat_key]["p50_ms"]
        else:
            lat_ms = float("nan")
        if np.isnan(lat_ms):
            continue
        ax.scatter(lat_ms, f1, s=size, color=colour, edgecolor="black",
                   linewidth=0.6, marker=marker, zorder=4, label=label)

    # If neural baselines are present, plot them. Prefer the dedicated
    # `distilbert.json` (from eval_distilbert.py) which has measured
    # latency; fall back to `baselines_neural.json` (legacy) which
    # doesn't have latency and gets a nominal x-coordinate.
    distilbert_path = _PROJECT_ROOT / "results" / "distilbert.json"
    if distilbert_path.exists():
        d = json.loads(distilbert_path.read_text())
        q = d.get("quality", {})
        lat = d.get("latency", {})
        if "per_call_streaming" in q and "p50_ms" in lat:
            f1 = q["per_call_streaming"]["f1"]
            lat_ms = lat["p50_ms"]
            ax.scatter(lat_ms, f1, s=44, color=_PALETTE["distilbert"],
                       edgecolor="black", linewidth=0.5, marker="^",
                       zorder=3, label="DistilBERT (fine-tuned)")

    neural_path = _PROJECT_ROOT / "results" / "baselines_neural.json"
    if neural_path.exists() and not distilbert_path.exists():
        ndata = json.loads(neural_path.read_text()).get("results", {})
        for key, label, colour in [
            ("distilbert_finetuned", "DistilBERT (fine-tuned)", _PALETTE["distilbert"]),
            ("claude_zeroshot",      "Claude Haiku (zero-shot)", _PALETTE["claude"]),
        ]:
            if key in ndata and "per_call_streaming" in ndata[key]:
                f1 = ndata[key]["per_call_streaming"]["f1"]
                # No measured latency in legacy path -- place at nominal x
                lat_ms = 50.0 if "distilbert" in key else 500.0
                ax.scatter(lat_ms, f1, s=44, color=colour, edgecolor="black",
                           linewidth=0.5, marker="^", zorder=3, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Per-sentence latency (ms, single CPU thread)")
    ax.set_ylabel("Streaming per-call F1")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.55, 1.02)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4,
              labelspacing=0.2)
    _save(fig, outdir, "fig1_pareto")


# ---------------------------------------------------------------------------
# Figure 2: TTD CDF
# ---------------------------------------------------------------------------

def fig_ttd_cdf(ttd: dict, outdir: Path, sec_per_sentence: float = 4.0) -> None:
    """Cumulative distribution of how many sentences before alert fires."""
    agg = ttd.get("ttd_aggregate", {})
    sorted_idx = agg.get("ttd_idx_sorted") or []
    n_scam = agg.get("n_scam_calls", 0)

    if not sorted_idx or n_scam == 0:
        return

    # Construct empirical CDF over ALL scam calls (so missed calls show
    # up as flat at the end, never reaching 1.0).
    xs = list(sorted_idx)
    ys = [(i + 1) / n_scam for i in range(len(xs))]
    # extend horizontally to a sensible max
    max_idx = max(max(xs) + 2, 25)
    xs = xs + [max_idx]
    ys = ys + [ys[-1]]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.step(xs, ys, where="post", color=_PALETTE["xgb"], linewidth=1.6)
    ax.fill_between(xs, 0, ys, step="post", alpha=0.15, color=_PALETTE["xgb"])

    # Median marker
    median = agg.get("ttd_idx_median")
    if median is not None:
        ax.axvline(median, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.text(median + 0.4, 0.05,
                f"median = {median:.0f} sentences\n(≈ {median*sec_per_sentence:.0f}s)",
                fontsize=7, alpha=0.8)

    ax.set_xlabel("Sentence index at which EMA first crosses 0.75")
    ax.set_ylabel("Fraction of scam calls detected")
    ax.set_xlim(0, max_idx)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis in seconds
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x * sec_per_sentence,
                   lambda x: x / sec_per_sentence),
    )
    secax.set_xlabel("Approximate wall-clock time (s)")
    _save(fig, outdir, "fig2_ttd_cdf")


# ---------------------------------------------------------------------------
# Figure 3: First-N degradation
# ---------------------------------------------------------------------------

def fig_first_n(ttd: dict, outdir: Path) -> None:
    fn = ttd.get("first_n", {})
    if not fn:
        return
    ns = sorted(int(k) for k in fn.keys())
    f1s = [fn[str(n)]["f1"] for n in ns]
    precs = [fn[str(n)]["precision"] for n in ns]
    recs = [fn[str(n)]["recall"] for n in ns]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(ns, f1s, "-o", color=_PALETTE["xgb"], label="F1", linewidth=1.4, markersize=4)
    ax.plot(ns, precs, "--s", color=_PALETTE["hand_lr"], label="Precision", linewidth=1.2, markersize=3)
    ax.plot(ns, recs, ":^", color=_PALETTE["tfidf_lr"], label="Recall", linewidth=1.2, markersize=3)

    ax.set_xlabel("N: sentences observed before classification")
    ax.set_ylabel("Per-call metric")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, outdir, "fig3_first_n")


# ---------------------------------------------------------------------------
# Figure 4: ASR robustness
# ---------------------------------------------------------------------------

def fig_asr(asr: dict, outdir: Path) -> None:
    table = asr.get("table", [])
    if not table:
        return

    by_pert: dict[str, list] = {}
    for row in table:
        by_pert.setdefault(row["perturbation"], []).append(row)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    colour_map = {
        "word_swap":   _PALETTE["hand_lr"],
        "word_delete": _PALETTE["xgb"],
        "char_noise":  _PALETTE["tfidf_lr"],
    }
    style_map = {
        "word_swap": "-o", "word_delete": "--s", "char_noise": ":^",
    }
    label_map = {
        "word_swap":   "Word substitution",
        "word_delete": "Word deletion",
        "char_noise":  "Character noise",
    }
    for pert, rows in by_pert.items():
        rows = sorted(rows, key=lambda r: r["rate"])
        xs = [r["rate"] for r in rows]
        ys = [r["stream_f1"] for r in rows]
        ax.plot(xs, ys, style_map.get(pert, "-o"),
                color=colour_map.get(pert, "black"),
                linewidth=1.4, markersize=3.5,
                label=label_map.get(pert, pert))

    ax.set_xlabel("Per-token perturbation rate")
    ax.set_ylabel("Streaming per-call F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    _save(fig, outdir, "fig4_asr_robustness")


# ---------------------------------------------------------------------------
# Figure 5: Cross-channel transfer
# ---------------------------------------------------------------------------

def fig_cross_channel(cc: dict, outdir: Path) -> None:
    """Grouped bar chart: F1 / Prec / Recall / AUROC across channels."""
    res = cc.get("results", {})
    if not res:
        return
    channels = [c for c in ["calls", "sms", "urls"] if c in res]
    if not channels:
        return
    labels = {"calls": "Calls (in-domain)", "sms": "SMS", "urls": "URLs"}
    metrics = ["f1", "precision", "recall", "auroc"]
    metric_labels = ["F1", "Precision", "Recall", "AUROC"]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(metrics))
    width = 0.25
    colours = {
        "calls": _PALETTE["xgb"],
        "sms":   _PALETTE["tfidf_lr"],
        "urls":  _PALETTE["hand_lr"],
    }
    for i, ch in enumerate(channels):
        vals = [res[ch].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width - width, vals, width=width,
               color=colours[ch], edgecolor="black", linewidth=0.4,
               label=labels[ch])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4,
              labelspacing=0.2, fontsize=7)
    _save(fig, outdir, "fig5_cross_channel")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    ap.add_argument("--outdir", default=str(_PROJECT_ROOT / "paper" / "figures"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    out = Path(args.outdir)

    latency = _load(rdir / "latency.json") or {}
    baselines = _load(rdir / "baselines.json") or {}
    ttd = _load(rdir / "ttd.json") or {}
    asr = _load(rdir / "asr_robustness.json") or {}
    cc = _load(rdir / "cross_channel.json") or {}

    print(f"[plot] writing figures to {out}")
    if latency and baselines:
        fig_pareto(latency, baselines, out)
    if ttd:
        fig_ttd_cdf(ttd, out)
        fig_first_n(ttd, out)
    if asr:
        fig_asr(asr, out)
    if cc:
        fig_cross_channel(cc, out)
    print("[plot] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
