# Reproducibility Map

This document maps every reported number, table, and figure in
`paper/sentineledge_emnlp2026.tex` to the script that produces it, the
intermediate `results/*.json` artifact it is read from, and whether the
number carries a bootstrap confidence interval (CI). It also lists the two
known gaps a reviewer will hit on a clean checkout.

All paths are relative to the repository root.

## 1. Quick start

```bash
# Reproducible environment for the xgboost / sklearn-pinned experiments
python3 -m venv .venv_exp
source .venv_exp/bin/activate
pip install -r requirements-experiments.txt   # xgboost==2.1.4, scikit-learn==1.6.1

# One-command evaluation regeneration (from released model artifacts)
python experiments/run_all.py
```

`run_all.py` regenerates the **evaluation** numbers from the released model
artifacts in `models/` and the evaluation splits. It does **not** retrain the
models; see the two gaps in Section 4.

## 2. How numbers reach the PDF

The paper never hard-codes a number. The pipeline is:

```
results/*.json ──> experiments/extract_paper_numbers.py ──> paper/_numbers.tex   (\newcommand macros)
results/*.json ──> experiments/bootstrap_cis.py          ──> paper/_cis.tex       (95% bootstrap CIs)
results/latency_breakdown.json     (run_latency_breakdown.py)     ──> paper/_latency.tex
results/sms_threshold_sweep.json   (run_sms_threshold_sweep.py)   ──> paper/_sms_sweep.tex
results/adversarial_youtube.json   (run_adversarial_youtube.py)   ──> paper/_adv_youtube.tex
```

The main `.tex` `\input`s `_numbers`, `_cis`, `_latency`, `_sms_sweep`, and
`_adv_youtube`. Every macro used in prose is defined in exactly one of these.

## 3. Paper element → results JSON → script → CI

| Paper element | `results/` artifact | Producing script | CI? |
|---|---|---|---|
| Headline LR numbers (Table `tab:headline`) | `baselines.json`, `eval_xgb.json` | `run_baselines.py`, `run_evaluation.py` | F1 ✓ (`bootstrap_cis.py`) |
| Six-head comparison (Table `tab:baselines`) | `baselines.json`, `baselines_neural.json`, `distilbert.json`, `latency.json`, `efficiency.json` | `run_baselines.py`, `run_baselines_neural.py`, `eval_distilbert.py`, `run_latency.py`, `measure_per_mb.py` | Stream F1 ✓; latency/size point est. |
| Time-to-detection (Fig. `fig:ttd`) | `ttd.json` | `run_time_to_detection.py` | point est. |
| ASR robustness (Fig. `fig:robust`, left) | `asr_robustness.json` | `run_asr_robustness.py` | point est. |
| Cross-channel table (Table `tab:cross`) | `cross_channel.json` | `run_cross_channel.py` | point est. |
| **SMS threshold sweep (Fig. `fig:sms_sweep`)** | `sms_threshold_sweep.json` | `run_sms_threshold_sweep.py --rescore` | point est. (full F1(τ) curve + PR curve) |
| Adversarial scam phrasings | `adversarial.json`, `adversarial_lr.json` | `run_adversarial.py`, `run_adversarial_lr.py` | point est. |
| **Adversarial on real YouTube (null result)** | `adversarial_youtube.json` | `run_adversarial_youtube.py` | point est. (n=80) |
| **Held-out-channel generalisation** | `channel_disjoint.json` (CIs), `robust_lr_channel_disjoint.json` (AUROC) | see Gap A below | F1 / precision / recall ✓ |
| **Per-stage latency breakdown** | `latency_breakdown.json` | `run_latency_breakdown.py` | point est. (median of 500 reps) |
| All CIs (`paper/_cis.tex`) | `bootstrap_cis.json` | `bootstrap_cis.py` | — (produces the CIs) |

Which numbers have CIs, at a glance:

- **With 95% bootstrap CIs:** per-head streaming F1; held-out-channel F1,
  precision, recall.
- **Point estimates only:** all AUROC values, latency (per-stage and p50),
  model sizes, cross-channel SMS/URL metrics, adversarial recovery deltas,
  time-to-detection, efficiency (F1/MB).

## 4. Known gaps on a clean checkout

**Gap A — held-out-channel training harness is not released.**
`results/robust_lr_channel_disjoint.json` (AUROC point estimates) was produced
by a training harness that is **not** part of the released scripts. The
held-out-channel **F1 / precision / recall CIs** are fully reproducible:
`channel_disjoint.json` stores the per-call confusion matrices, and
`bootstrap_cis.py` recomputes the intervals
(`\OldXgbDisjointFOneCI`, `\RetrainedTfidfLrDisjointFOneCI`, ...). Only the
two AUROC point estimates (`\DisjointOldXgbAuroc`, `\DisjointRetrainedAuroc`)
depend on the unreleased harness. This is disclosed in the paper's
Availability section.

**Gap B — training corpora are generated, not shipped.**
`data/processed/call_fraud_train.csv` (and the other processed CSVs) are **not**
committed. They are produced by the project's template-based generator:

```bash
python training/generate_synthetic_data.py
python training/prepare_datasets.py
```

Scripts that **retrain** (`run_baselines.py`, `run_adversarial.py`,
`run_evaluation.py`) therefore require running the generator first. Scripts that
only **evaluate released artifacts** do not:
`run_adversarial_youtube.py`, `run_sms_threshold_sweep.py --rescore`,
`run_latency_breakdown.py`, and `bootstrap_cis.py` run directly against
`models/` and `data/external/` with no processed CSVs.

## 5. Environment notes

- `models/call_fraud_xgb.json` was trained with **xgboost 2.1.4**; the pickled
  TF-IDF vectorisers come from **scikit-learn 1.6.1**. `requirements-experiments.txt`
  pins both to avoid binary/format mismatch warnings.
- Whisper-stage (ASR) latency is **not** measured in the default run; it is
  cited from Radford et al. and only measured if `--audio <wav>` is passed to
  `run_latency_breakdown.py` with `openai-whisper` installed.
