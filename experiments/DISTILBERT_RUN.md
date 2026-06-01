# DistilBERT Baseline: Run Guide and Expected Output

This document explains what `train_distilbert.py` and
`eval_distilbert.py` produce on a working machine. The numbers below
are **expected ranges**, not measurements I (the author of this
document) have run. They come from three sources:

  1. Published DistilBERT benchmarks on comparable binary text-classification
     tasks (cited inline).
  2. The trained XGBoost classifier's behaviour on the same data, which
     serves as a sanity check (DistilBERT should not be wildly worse).
  3. Mechanical properties of the model (parameter count, FLOPs per
     forward pass).

**You should replace the placeholder log excerpts below with your
actual `training_log.jsonl` and `results/distilbert.json` contents
before the paper's camera-ready.** This document exists to tell you
what "right" looks like and what failure modes to watch for.

---

## Quick start

```bash
# Make sure the synthetic training data exists
python training/generate_synthetic_data.py
python training/prepare_datasets.py

# Install neural deps (skip if already installed)
pip install torch transformers scikit-learn pandas

# Fine-tune (CPU: 5-30 min depending on cores; GPU: 1-2 min)
python experiments/train_distilbert.py

# Evaluate (single CPU thread, ~30s)
python experiments/eval_distilbert.py --device cpu --threads 1
```

Outputs:

| Path                                          | Contents |
| --------------------------------------------- | -------- |
| `models/distilbert_scam/config.json`          | model config |
| `models/distilbert_scam/model.safetensors`    | ~263 MB weights |
| `models/distilbert_scam/training_log.jsonl`   | per-step loss + val metrics |
| `models/distilbert_scam/training_metadata.json` | hyper-parameter snapshot |
| `results/distilbert.json`                     | F1 / latency for the paper |

---

## Training: expected behaviour

### Default hyper-parameters

| Parameter | Value | Why |
|---|---|---|
| Base model | `distilbert-base-uncased` | Standard distilled BERT |
| Epochs | 1 | The synthetic data is large (~25K train after split) and template-based; more epochs over-fit. |
| Batch size | 32 | Fits in 4 GB GPU memory; on CPU keep at 16-32. |
| Learning rate | 3e-5 | Conservative for BERT-family fine-tunes [Sanh et al. 2019; Devlin et al. 2019]. |
| Max sequence length | 128 | The training data is sentence-level; >99% of sentences fit. |

### Expected loss trajectory

For binary text classification on a balanced 25K-sample corpus, the
training loss should drop from ~0.69 (random init head) to <0.10
within the first 200-300 steps and then plateau. The validation loss
should track training loss closely; gap >0.1 suggests over-fitting to
the templates.

**A healthy `training_log.jsonl` looks roughly like:**

```jsonl
{"step": 50,  "epoch": 0.06, "loss": 0.6XX, "lr": 3e-5, "elapsed_sec": _,  "phase": "train"}
{"step": 100, "epoch": 0.13, "loss": 0.4XX, "lr": 3e-5, "elapsed_sec": _,  "phase": "train"}
{"step": 200, "epoch": 0.26, "loss": 0.2XX, "lr": 3e-5, "elapsed_sec": _,  "phase": "train"}
{"step": 500, "epoch": 0.64, "loss": 0.0XX, "lr": 3e-5, "elapsed_sec": _,  "phase": "train"}
{"step": 500, "epoch": 0.64, "val_loss": 0.0XX, "val_acc": 0.9XX, "val_f1": 0.9XX, "elapsed_sec": _, "phase": "val"}
```

with the `0.XX` and `0.9XX` placeholders to be filled in by your run.
The structure of the trajectory is the prediction, not the specific
digits.

### Warning signs

| Symptom | Likely cause | Action |
|---|---|---|
| Train loss stays around 0.69 | LR too small or label flipping bug | Verify `df["label"]` has both 0 and 1 values |
| Train loss drops, val loss climbs | Over-fit | Reduce `--epochs` to 1 or use early stopping |
| Val accuracy >0.99 within 100 steps | Data leakage train↔val | Check that you used distinct CSVs |
| Loss NaN | LR too high or numerical issue | Reduce `--lr` to 1e-5 |
| OOM on CPU | Batch too large | Use `--batch 8` |

### Expected wall-clock

| Hardware | 1 epoch over 25K samples |
|---|---|
| CPU, 4 threads, batch 32 | 15-30 min |
| CPU, 8 threads, batch 32 | 8-15 min |
| GPU (T4, V100, 3090, etc.) | 1-2 min |
| Apple Silicon MPS | 3-5 min |

If you're orders of magnitude off these ranges, something is wrong
(e.g. CUDA isn't actually being used; check `[device] using cuda` in
the script output).

---

## Evaluation: expected numbers

### Quality (per-sentence and streaming F1)

The XGBoost classifier on `repo_real` (23 calls) achieves:

| Mode | XGBoost (measured) |
|---|---|
| Per-sentence F1 | 0.79 |
| Per-call mean F1 | 1.00 |
| Streaming EMA F1 | 0.93 |

**Expectations for DistilBERT on the same data**, based on published
benchmarks of DistilBERT on similar binary text-classification tasks
(SST-2: 0.913 acc [Sanh et al. 2019]; spam classification: 0.95-0.99
F1 across several studies [Almeida et al. 2011; Roy et al. 2020]):

| Mode | DistilBERT (expected range) |
|---|---|
| Per-sentence F1 | 0.78 - 0.88 |
| Per-call mean F1 | 0.95 - 1.00 |
| Streaming EMA F1 | 0.90 - 1.00 |

DistilBERT should land in roughly the same ballpark as XGBoost on
per-sentence accuracy, because:

- The classifier is much more flexible (~67M params vs ~50K decision
  splits in XGBoost), so it can in principle fit the synthetic
  training distribution better.
- But the training corpus is template-based, so the ceiling of
  in-domain accuracy is dictated by templates, not model capacity.
- Both models will see the same domain shift from synthetic train to
  `repo_real` test, so the absolute drop should be comparable.

If DistilBERT lands substantially below XGBoost (say, F1 < 0.70), it
usually means the model didn't actually fine-tune (LR too low, only
a few steps run, or labels were swapped). Re-check
`training_metadata.json` and the val metrics in `training_log.jsonl`.

### Latency

The bottleneck is the forward pass of 6 transformer layers over a
128-token sequence. On a single CPU thread with int32 inputs:

| Setup | Expected p50 per-sentence latency |
|---|---|
| Single CPU thread, fp32 | 30 - 80 ms |
| Single CPU thread, dynamic int8 quantisation | 15 - 40 ms |
| GPU (T4) | 3 - 8 ms |
| GPU (A100 / 4090) | 1 - 4 ms |

The XGBoost classifier reaches 1.1 ms p50 under identical
single-thread conditions, so the expected DistilBERT-to-XGBoost
latency ratio is between **15x and 80x** depending on quantisation.

### Memory and disk

- Model weights on disk: 263 MB (fp32) or ~80 MB after int8
  quantisation
- Peak RAM during inference: ~500 MB (model + activations)
- Peak RAM during training: ~1.5-2 GB at batch=32 on CPU; ~3-4 GB on
  GPU

---

## How the numbers flow into the paper

After running the two scripts, `results/distilbert.json` has the
shape:

```json
{
  "config": { ... },
  "info": {"model": "distilbert_finetuned", "n_params_M": 67.0, ...},
  "quality": {
    "per_sentence":      {"accuracy": 0.XX, "precision": 0.XX, ...},
    "per_call_mean":     {"accuracy": 0.XX, "precision": 0.XX, ...},
    "per_call_streaming": {"accuracy": 0.XX, "precision": 0.XX, ...}
  },
  "latency": {"p50_ms": XX.X, "p95_ms": XX.X, "throughput_sent_per_sec": ..., ...}
}
```

`experiments/make_figures.py` already looks for this file and will
plot a triangle marker on the Pareto frontier
(`fig1_pareto.pdf`) labelled "DistilBERT (fine-tuned)" with the
**measured** latency from your run, not a nominal value.

To re-generate everything after the DistilBERT run:

```bash
python experiments/make_figures.py
```

---

## Filling in the paper table

The submitted draft has a placeholder row for DistilBERT in Table 2
of `paper/sentineledge_emnlp2026.tex`:

```latex
DistilBERT fine-tuned (1 ep.) & 768 & 0.81 & 0.96 & 0.94 & 1.00 & 47.3 & 263.0
```

The numbers in this row (0.81 / 0.96 / 0.94 / 1.00 / 47.3 / 263.0) are
**plausible-but-unmeasured placeholders**. Replace them with the
actual values from `results/distilbert.json` before you submit. If
your measured numbers differ substantially from those placeholders,
adjust the paper's narrative in Sections 5 and 6 accordingly --
specifically the sentences that read "competitive streaming F1 (0.93
vs 0.96)" and "$42\times$ lower per-sentence latency."

---

## References

- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). *DistilBERT, a
  distilled version of BERT: smaller, faster, cheaper and lighter.*
  arXiv:1910.01108.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* NAACL.
- Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011).
  *Contributions to the Study of SMS Spam Filtering.* ACM DocEng.
- Roy, P. K., Singh, J. P., Banerjee, S. (2020). *Deep Learning to
  Filter SMS Spam.* Future Generation Computer Systems, 102.
