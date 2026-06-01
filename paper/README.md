# SentinelEdge -- EMNLP 2026 Demo Submission Package

This directory contains everything needed to reproduce the
SentinelEdge paper.

```
paper/
├── sentineledge_emnlp2026.tex   # paper draft (single-blind, ≤6 pages)
├── sentineledge.bib             # bibliography
├── demo_supplement.tex          # optional 1-page session description
├── SUBMISSION_CHECKLIST.md      # maps every EMNLP 2026 demo requirement to status
├── DEMO_VIDEO_SCRIPT.md         # 2.5-min screencast script
├── LIVE_DEMO_DEPLOYMENT.md      # deployment guide for the required live URL
├── figures/
│   ├── fig1_pareto.pdf          # F1 vs latency Pareto
│   ├── fig2_ttd_cdf.pdf         # time-to-detection CDF
│   ├── fig3_first_n.pdf         # first-N degradation
│   ├── fig4_asr_robustness.pdf  # ASR-error robustness
│   └── fig5_cross_channel.pdf   # cross-channel transfer
└── README.md                    # this file
```

experiments/
├── PREPARE_DATA.md              # how to obtain TeleAntiFraud-28k, BETTER30
├── dataset_loader.py            # unified per-call loader
├── run_evaluation.py            # per-sentence / per-call / streaming eval
├── run_time_to_detection.py     # TTD CDF + first-N curve
├── run_latency.py               # single-thread CPU latency benchmark
├── run_baselines.py             # 5 lightweight baselines + trained XGB
├── run_baselines_neural.py      # DistilBERT + Claude (needs HF/API)
├── run_asr_robustness.py        # word-swap/delete/char-noise perturbations
├── run_adversarial.py           # adversarial scams + hard negatives
├── run_all.py                   # master runner, produces paper_tables.json
└── make_figures.py              # builds all four paper figures

results/
└── paper_tables.json            # joined results from all experiments
```

## Reproducing the paper

```bash
pip install -r requirements.txt
python experiments/run_all.py --sources repo_real
```

This takes ~3 minutes on a single CPU core and produces every number
and every figure in the paper draft.

To include the external benchmarks, follow `experiments/PREPARE_DATA.md`
to download TeleAntiFraud-28k and BETTER30, then:

```bash
python experiments/run_all.py --sources repo_real teleantifraud_28k better30
```

To include the neural baselines (DistilBERT fine-tune + Claude
zero-shot):

```bash
export ANTHROPIC_API_KEY=...       # optional, for Claude
python experiments/run_all.py --sources repo_real --with-neural
```

DistilBERT fine-tuning takes ~10-30 minutes on CPU, depending on the
size of the synthetic training set.

## Building the paper PDF

The `.tex` file targets the EMNLP 2023 / 2026 ACL style (`emnlp2023.sty`).
Drop it into the ACL author kit and run:

```bash
pdflatex sentineledge_emnlp2026
bibtex sentineledge_emnlp2026
pdflatex sentineledge_emnlp2026
pdflatex sentineledge_emnlp2026
```

## What's deliberately not in this submission

The following are described in the paper's "Limitations" section but
are NOT claimed as results:

* Federated learning quality numbers (the simulator runs on a linear
  surrogate; federating XGBoost remains an open problem).
* Real Android deployment.
* End-to-end mic-to-alert measurements through Whisper.

Each of these is future work, called out explicitly in
Section~\ref{sec:limits}.
