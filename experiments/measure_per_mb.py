"""Measure quality/latency efficiency per model megabyte.

This script reads the already-computed paper result files and reports the
headline efficiency ratios for the paper's default LR-on-hand-crafted
classifier. It does not retrain models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latency_by_model(latency: dict) -> dict[str, dict]:
    return {row["model"]: row for row in latency.get("results", [])}


def _fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "efficiency.json"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    baselines = _load(results_dir / "baselines.json")["results"]
    latency = _latency_by_model(_load(results_dir / "latency.json"))
    distil = _load(results_dir / "distilbert.json")

    hand_lr_quality = baselines["handcrafted_lr"]["per_call_streaming"]
    hand_lr_latency = latency["logreg_handcrafted_18d"]
    tfidf_lr_quality = baselines["tfidf_lr"]["per_call_streaming"]
    tfidf_lr_latency = latency["logreg_tfidf_500d"]
    xgb_quality = baselines["trained_xgb"]["per_call_streaming"]
    xgb_latency = latency["xgb_tfidf_518d"]
    distil_quality = distil["quality"]["per_call_streaming"]
    distil_latency = distil["latency"]
    distil_info = distil["info"]

    rows = {
        "handcrafted_lr": {
            "f1": hand_lr_quality["f1"],
            "precision": hand_lr_quality["precision"],
            "recall": hand_lr_quality["recall"],
            "auroc": hand_lr_quality["auroc"],
            "size_mb": hand_lr_latency["disk_size_mb"],
            "p50_ms": hand_lr_latency["p50_ms"],
            "throughput": hand_lr_latency["throughput_sent_per_sec"],
        },
        "tfidf_lr": {
            "f1": tfidf_lr_quality["f1"],
            "precision": tfidf_lr_quality["precision"],
            "recall": tfidf_lr_quality["recall"],
            "auroc": tfidf_lr_quality["auroc"],
            "size_mb": tfidf_lr_latency["disk_size_mb"],
            "p50_ms": tfidf_lr_latency["p50_ms"],
            "throughput": tfidf_lr_latency["throughput_sent_per_sec"],
        },
        "xgb_tfidf": {
            "f1": xgb_quality["f1"],
            "precision": xgb_quality["precision"],
            "recall": xgb_quality["recall"],
            "auroc": xgb_quality["auroc"],
            "size_mb": xgb_latency["disk_size_mb"],
            "p50_ms": xgb_latency["p50_ms"],
            "throughput": xgb_latency["throughput_sent_per_sec"],
        },
        "distilbert": {
            "f1": distil_quality["f1"],
            "precision": distil_quality["precision"],
            "recall": distil_quality["recall"],
            "auroc": distil_quality["auroc"],
            "size_mb": distil_info["disk_size_mb_estimate"],
            "p50_ms": distil_latency["p50_ms"],
            "throughput": distil_latency["throughput_sent_per_sec"],
        },
    }

    for row in rows.values():
        row["f1_per_mb"] = row["f1"] / row["size_mb"]
        row["auroc_per_mb"] = row["auroc"] / row["size_mb"]

    ours = rows["handcrafted_lr"]
    ratios = {
        "vs_tfidf_lr_f1_per_mb": ours["f1_per_mb"] / rows["tfidf_lr"]["f1_per_mb"],
        "vs_xgb_f1_per_mb": ours["f1_per_mb"] / rows["xgb_tfidf"]["f1_per_mb"],
        "vs_distilbert_f1_per_mb": ours["f1_per_mb"] / rows["distilbert"]["f1_per_mb"],
        "vs_distilbert_size": rows["distilbert"]["size_mb"] / ours["size_mb"],
        "vs_distilbert_latency": rows["distilbert"]["p50_ms"] / ours["p50_ms"],
        "vs_distilbert_throughput": ours["throughput"] / rows["distilbert"]["throughput"],
    }

    out_rows = []
    for name, row in rows.items():
        out_rows.append({
            "model": name,
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "auroc": row["auroc"],
            "size_mb": row["size_mb"],
            "size_kb": row["size_mb"] * 1024,
            "p50_ms": row["p50_ms"],
            "throughput": row["throughput"],
            "f1_per_mb": row["f1_per_mb"],
            "f1_per_kb": row["f1_per_mb"] / 1024,
            "auroc_per_mb": row["auroc_per_mb"],
        })

    ratios["f1_retention_vs_distilbert"] = (
        ours["f1"] / rows["distilbert"]["f1"]
    )

    out = {
        "rows": out_rows,
        "headline_ratios": ratios,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== per-MB efficiency ===")
    for name, row in rows.items():
        print(
            f"{name:16s} F1={row['f1']:.3f}  size={row['size_mb']:.6f} MB  "
            f"F1/MB={row['f1_per_mb']:.1f}  p50={row['p50_ms']:.3f} ms"
        )

    print(
        "headline ratios: "
        f"handcrafted LR retains {_fmt(100 * ratios['f1_retention_vs_distilbert'], 0)}% "
        f"of DistilBERT's streaming F1 while being "
        f"{_fmt(ratios['vs_distilbert_size'], 0)}x smaller, "
        f"{_fmt(ratios['vs_distilbert_latency'], 0)}x lower latency, and "
        f"{_fmt(ratios['vs_distilbert_throughput'], 0)}x higher throughput; "
        f"as an auxiliary efficiency metric, it delivers "
        f"{_fmt(ratios['vs_xgb_f1_per_mb'])}x the F1/MB of XGBoost and "
        f"{_fmt(ratios['vs_distilbert_f1_per_mb'], 0)}x the F1/MB of DistilBERT."
    )
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
