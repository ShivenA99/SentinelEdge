# Experiments to Run for the EMNLP 2026 Submission

This is the complete list of experiments you need to run before
submission, in priority order. Each row tells you:

  - **What** experiment
  - **The command** to run it
  - **The file** it produces
  - **Which paper macros** it fills in

After running all the experiments below, regenerate the LaTeX
numbers file once and recompile the paper:

```bash
python experiments/extract_paper_numbers.py
cd paper && pdflatex sentineledge_emnlp2026 && bibtex sentineledge_emnlp2026 \
    && pdflatex sentineledge_emnlp2026 && pdflatex sentineledge_emnlp2026
```

The extraction script reports how many macros are still `TBD` after
each run, so you can see at a glance what's missing.

---

## Setup (one-time)

```bash
cd SentinelEdge

# Python deps already in requirements.txt
pip install -r requirements.txt

# Regenerate the training and adversarial CSVs (idempotent, ~1 min)
python training/generate_synthetic_data.py
python training/prepare_datasets.py
python training/generate_adversarial_data.py
```

After this, `data/processed/call_fraud_train.csv` and friends exist.

---

## Stage 0: Get real evaluation data

These are the *only* manual steps. Everything else is one command.

### 0a. BETTER30 (Kaggle, 30 minutes)

```bash
# Manual: download BETTER30 from Kaggle (search "call transcripts
# scam determinations"), place at:
mv ~/Downloads/BETTER30.csv data/external/better30.csv

# Verify:
python experiments/dataset_loader.py
# Should show better30 with ~30 calls.
```

### 0b. (Optional but high value) YouTube scam-baiter collection

```bash
# Follow experiments/SCAMBAITER_PROTOCOL.md
# - Pick 30-50 videos, create video_list.tsv
# - Run:
python experiments/collect_youtube_scambaiters.py \
    --video-list data/external/youtube_baiters/video_list.tsv

# - Manually annotate the resulting annotations.tsv (3-5 hours)
# - Verify:
python experiments/dataset_loader.py
# Should show youtube_baiters with ~50 calls.
```

### 0c. (Optional, requires email) Wu et al. 2024 corpus

Send the email in `experiments/EMAIL_TEMPLATE.md`. If they share,
place the files at `data/external/wu2024_corpus/{sc,sd,masc,our_real,our_synt}.csv`.

---

## Stage 1: Core experiments (~3 minutes total)

These all run on the data you already have. Use `--sources` to
include whichever real-data sources you've acquired:

```bash
# Define your evaluation source list once
SRC="repo_real better30"
# If you have YouTube baiters: SRC="repo_real better30 youtube_baiters"
# If you have Wu corpus too:   SRC="repo_real better30 youtube_baiters wu2024_corpus"
```

| # | Experiment | Command | Output file | Macros filled |
|---|---|---|---|---|
| 1 | Headline eval | `python experiments/run_evaluation.py --sources $SRC` | `results/eval_xgb.json` | `\PerSent*`, `\Mean*`, `\Stream*`, `\NumEvalCalls`, `\NumScamCalls`, `\NumLegitCalls`, `\EvalSources`, `\StreamMissed` |
| 2 | Time-to-detection | `python experiments/run_time_to_detection.py --sources $SRC` | `results/ttd.json` | `\TtdMedianSent`, `\TtdMedianSec`, `\TtdMidSent`, `\TtdMidSec`, `\TtdHighSent`, `\Full*` |
| 3 | Latency | `python experiments/run_latency.py` | `results/latency.json` | `\XgbLatencyMs`, `\XgbThroughput`, `\XgbSizeMB`, `\HandLR*`, `\TfidfLR*`, `\CombLR*`, `\HandSVM*` |
| 4 | Baseline quality | `python experiments/run_baselines.py --eval-sources $SRC` | `results/baselines.json` | `\HandLRFOne`/`Prec`/`Rec`, `\HandSVMFOne`/`Prec`/`Rec`, `\TfidfLRFOne`/`Prec`/`Rec`, `\CombLRFOne`/`Prec`/`Rec` |
| 5 | ASR robustness | `python experiments/run_asr_robustness.py --sources $SRC` | `results/asr_robustness.json` | `\AsrCleanF`, `\AsrSwapHi`, `\AsrDelHi`, `\AsrCharHi` |
| 6 | Adversarial | `python experiments/run_adversarial.py` | `results/adversarial.json` | `\AdvNumSents`, `\AdvCleanF`, `\AdvAdvF`, `\AdvRetrainedCleanF`, `\AdvRetrainedAdvF`, `\AdvGapClosedPts`, `\GovFPbeforePct`, `\GovFPafterPct` |
| 7 | Cross-channel | `python experiments/run_cross_channel.py` | `results/cross_channel.json` | `\CallsInDomain*`, `\Sms*`, `\Url*` |

Or do all seven at once:

```bash
python experiments/run_all.py --sources repo_real better30
```

This regenerates every figure under `paper/figures/` automatically.

---

## Stage 2: DistilBERT baseline (~10-30 minutes CPU; ~2 minutes GPU)

The deferred neural baseline. Run separately because of the
training cost.

```bash
# Train (writes models/distilbert_scam/)
python experiments/train_distilbert.py

# Evaluate with measured latency
python experiments/eval_distilbert.py --eval-sources $SRC
```

| Output file | Macros filled |
|---|---|
| `results/distilbert.json` | `\DistilFOne`, `\DistilPrec`, `\DistilRec`, `\DistilLat`, `\DistilSize`, `\SpeedupVsDistil`, `\SizeRatioVsDistil` |

The two derived macros (`\SpeedupVsDistil` = ratio of DistilBERT to
XGB latency, `\SizeRatioVsDistil` = ratio of disk sizes) are
computed automatically by the extractor.

What to expect from the run: see `experiments/DISTILBERT_RUN.md`
for realistic ranges, failure modes, and what a healthy
`training_log.jsonl` should look like.

---

## Stage 3: Regenerate paper

```bash
# Pull every number from results/*.json into LaTeX macros
python experiments/extract_paper_numbers.py

# Build paper PDF
cd paper
pdflatex sentineledge_emnlp2026
bibtex sentineledge_emnlp2026
pdflatex sentineledge_emnlp2026
pdflatex sentineledge_emnlp2026
```

The extractor prints "X / 94 macros filled" and lists any that are
still `TBD`. If `TBD` appears in the compiled PDF, you missed an
experiment.

Pass URLs to the extractor on submission day so the paper carries
the right links:

```bash
python experiments/extract_paper_numbers.py \
    --demo-url https://sentineledge-demo.fly.dev \
    --video-url https://youtu.be/abcdef \
    --repo-url https://github.com/yourgroup/sentineledge
```

---

## Stage 4: One-shot reproduction script

For the camera-ready reproducibility statement, verify that the
single command in the paper still runs:

```bash
# On a fresh checkout, with only requirements.txt installed:
python experiments/run_all.py --sources repo_real
```

This should complete in roughly 2 minutes and produce identical
numbers to your local run (modulo the latency benchmark, which is
hardware-dependent).

---

## Total time budget

| Stage | Wall-clock | Owner |
|---|---|---|
| Setup | 5 min | you |
| Stage 0a -- BETTER30 | 30 min | you, manual |
| Stage 0b -- YouTube collection | 3-5 days (incl. annotation) | you, manual |
| Stage 0c -- Wu corpus | 1-2 weeks wait | you, email then wait |
| Stage 1 -- core experiments | 2 min | one command |
| Stage 2 -- DistilBERT | 10-30 min CPU or 2 min GPU | one command after train |
| Stage 3 -- paper | 1 min | one command |

**Critical path**: Stage 0a + Stage 1 + Stage 2 + Stage 3 = about
**1 hour** of active work to get a complete paper with real numbers
on \textsc{repo-real} + BETTER30 + DistilBERT. Everything beyond
that strengthens the paper but is not strictly required.

---

## Final pre-submission checklist

After the runs above, walk through `paper/SUBMISSION_CHECKLIST.md`
and confirm:

1. `paper/_numbers.tex` has zero `TBD` entries
2. The paper PDF compiles cleanly and is `<= 6` pages of content
3. The author block has real names and affiliations
4. The demo URL works in incognito
5. The video URL plays
6. The repo URL is public
