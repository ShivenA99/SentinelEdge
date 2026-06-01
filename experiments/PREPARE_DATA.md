# Preparing External Datasets for Evaluation

This document explains how to acquire the two external datasets that the
SentinelEdge paper evaluation uses. The repo itself ships with only the
23 hand-written call transcripts in `data/real/call_transcripts/`; the
external datasets need to be downloaded once and placed in
`data/external/`.

The dataset loaders in `experiments/dataset_loader.py` are tolerant of
column-name variations and will work as long as the expected files are
present at the paths below.

---

## TeleAntiFraud-28k (primary external benchmark)

A 28,511-sample audio+text scam call corpus released in March 2025
(arXiv:2503.24115). Roughly balanced (46% scam, 54% normal). The audio
is useful for future ASR experiments; for now we only need the text.

### Files expected

```
data/external/teleantifraud_28k/
├── train.jsonl   # 21,490 records
└── test.jsonl    #  7,021 records
```

Each line is a JSON object. The loader is tolerant but expects at
minimum:

```json
{"id": "taf_test_0001",
 "label": 1,
 "transcript": "Hello, this is Officer Wilson with the SSA Fraud Division...",
 "category": "government_impersonation"}
```

### How to obtain

The dataset is released through the corresponding HuggingFace mirror.
At the time of writing the canonical link is announced in the arXiv
paper (search the project page for "TeleAntiFraud" on HuggingFace).
Two options:

**Option A: HuggingFace Hub**

```bash
pip install huggingface_hub
mkdir -p data/external/teleantifraud_28k
huggingface-cli download <ORG>/TeleAntiFraud-28k \
    --repo-type dataset \
    --local-dir data/external/teleantifraud_28k
```

You may need to convert from Parquet to JSONL:

```bash
python -c "
import pandas as pd, json, glob, os
for split in ('train', 'test'):
    pq = glob.glob(f'data/external/teleantifraud_28k/{split}*.parquet')
    if not pq: continue
    df = pd.concat([pd.read_parquet(p) for p in pq])
    out = f'data/external/teleantifraud_28k/{split}.jsonl'
    with open(out, 'w') as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({
                'id': r.get('id', ''),
                'label': int(r.get('label', 0)),
                'transcript': r.get('transcript', r.get('text', '')),
                'category': r.get('category', 'unknown'),
            }) + '\n')
    print('wrote', out)
"
```

**Option B: arXiv supplementary**

If the HF mirror is gated, the arXiv paper's GitHub typically links to
the raw data. Place either `data.jsonl` or the split files in
`data/external/teleantifraud_28k/`; the loader looks for either layout.

### Verify

```bash
python experiments/dataset_loader.py
```

You should see something like:

```
source                   n_calls    n_scam   n_legit   sent/call
----------------------------------------------------------------
repo_real                     23        15         8        21.0
teleantifraud_28k           7021      3697      3324         9.2
better30                  (none)
```

---

## Kaggle BETTER30 transcript dataset

A smaller real-life-annotated call-transcript dataset on Kaggle,
"Call Transcripts Scam Determinations" (filename `BETTER30.csv`).
Useful as a second cross-validation source.

### Files expected

```
data/external/better30.csv
```

Columns expected (case-insensitive, the loader handles synonyms):
`transcript`, `label`, optional `category`, optional `id`.

### How to obtain

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d <USERNAME>/call-transcripts-scam-determinations \
    -p data/external/
unzip data/external/call-transcripts-scam-determinations.zip -d data/external/
mv data/external/BETTER30.csv data/external/better30.csv
```

If Kaggle auth is unavailable, the dataset can be downloaded manually
from kaggle.com. The loader accepts any CSV with the expected columns.

### Verify

```bash
python experiments/dataset_loader.py
```

You should now see all three sources populated.

---

## Optional: real Whisper transcripts on TeleAntiFraud audio

The "ASR-error robustness" experiment in `run_asr_robustness.py` uses
synthetic perturbations as a stand-in for true Whisper errors. If you
have GPU time, you can run actual Whisper-tiny on TeleAntiFraud-28k
audio and store the resulting transcripts at:

```
data/external/teleantifraud_28k_whisper_tiny/{train,test}.jsonl
```

with the same schema as the gold transcripts. Then add a comparison
section to the paper: gold-vs-Whisper transcript F1.

A minimal script:

```python
import whisper, json, glob, os
model = whisper.load_model("tiny.en")
out = open("data/external/teleantifraud_28k_whisper_tiny/test.jsonl", "w")
for wav in glob.glob("data/external/teleantifraud_28k/audio/test/*.wav"):
    r = model.transcribe(wav)
    cid = os.path.splitext(os.path.basename(wav))[0]
    out.write(json.dumps({"id": cid, "transcript": r["text"]}) + "\n")
```

This is left as an extension because (a) it needs GPU, (b) it needs the
audio half of the dataset which is ~10 GB, and (c) the synthetic ASR
perturbations are already a defensible lower bound for the paper.

---

## Disk and time budget

| Step | Disk | Wall-clock (CPU) |
|---|---|---|
| Synthetic training/test CSVs (regen) | 3 MB | 1 min |
| Adversarial CSV (regen)              | 1 MB | 30 s  |
| TeleAntiFraud-28k text only         | 50 MB | 2 min download |
| TeleAntiFraud-28k audio (optional)  | ~10 GB | 1-2 hrs download |
| BETTER30                            | < 5 MB | 30 s |
| Whisper-tiny on TAF-28k audio       | --     | ~6 hrs on a 4090, otherwise skip |

After preparation, `python experiments/run_all.py --sources repo_real
teleantifraud_28k better30` will execute the full evaluation in roughly
10-20 minutes on a single CPU core (excluding any DistilBERT fine-tune).
