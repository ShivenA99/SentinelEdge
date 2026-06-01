# Preparing External Data for SentinelEdge Evaluation

> **Correction**: an earlier version of this document listed
> TeleAntiFraud-28k as the primary external benchmark. That dataset is
> **Mandarin Chinese**, not English, and is **not used** for the
> headline evaluation. It can be considered for a future cross-lingual
> extension; it is not part of the EMNLP demo paper's evaluation.

The repository ships with three real-data sources already:

| In-repo source | Path | Size | Used as |
|---|---|---|---|
| `repo_real` call transcripts | `data/real/call_transcripts/` | 23 calls | primary in-domain test |
| SMS Spam Collection | `data/real/sms_spam/SMSSpamCollection.tsv` | 5,574 SMS | cross-channel transfer test |
| Phishing URLs (combined) | `data/real/phishing_urls/combined_real_urls.csv` | 4,001 URLs | cross-channel transfer test |

This file documents the **external** data you need to add to
strengthen the evaluation. There are two tiers: a minimum-viable
set, and a stronger set worth the extra effort given the EMNLP demo
deadline.

---

## Tier 1 (download in 30 minutes): BETTER30

**What it is.** A Kaggle dataset titled *"Call Transcripts: Scam
Determinations"*. Real call transcripts, manually annotated, English.
About 30 calls -- small but real.

**Why use it.** It is the only publicly downloadable English
call-transcript scam corpus that ships ready-to-use. Pair with the
23 `repo_real` calls and you have ~50 real calls for evaluation,
which is the realistic floor.

**How to download.**

```bash
# Install Kaggle CLI if you don't have it
pip install kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json

# Search for the dataset; the file is consistently called BETTER30.csv.
# Common slug (confirm on Kaggle):
kaggle datasets download -d narayanaramaiyer/call-transcripts-scam-determinations \
    -p data/external/
cd data/external && unzip -o *.zip
mv BETTER30.csv better30.csv
```

If the slug above fails, search Kaggle for *"BETTER30"* or *"call
transcripts scam determinations"* manually. Place the resulting CSV
at `data/external/better30.csv`. The `dataset_loader.load_better30`
function is tolerant of column-name variations.

**Verify.**

```bash
python experiments/dataset_loader.py
```

You should see `better30` populated with ~30 records.

---

## Tier 2 (most important external data, requires email): Wu et al. 2024 aggregated corpus

**What it is.** The corpus used in *Combating Phone Scams with
LLM-based Detection: Where Do We Stand?* (Shen, Wang, Zhang, Ngai, Fu;
AAAI 2025, arXiv 2409.11643). It aggregates:

- **SC**, **SD**, **MASC** -- three synthesized dialogue corpora from
  Gumphusiri 2024 (IEEE WI-IAT)
- **Our-Real** -- real fraudulent calls scraped from YouTube
  scam-baiter videos, Whisper-transcribed
- **Our-Synt** -- their own LLM-generated synthesized corpus

This is the most directly comparable evaluation set to your work and
the most important one to acquire if you can. It is **not on GitHub**
as of the searches conducted for this document; it requires
contacting the authors.

**How to obtain.** Email the corresponding author:

> Zitong Shen --
> [esther.shen@connect.polyu.hk](mailto:esther.shen@connect.polyu.hk)
> (Hong Kong Polytechnic University)

A template email is in `experiments/EMAIL_TEMPLATE.md`. Researchers
in this space are usually responsive to evaluation-only requests
from non-competing groups; expect a response in 1-2 weeks. If they
share it, place the files at:

```
data/external/wu2024_corpus/
├── sc.csv
├── sd.csv
├── masc.csv
├── our_real.csv
└── our_synt.csv
```

The loader (`load_wu2024` in `dataset_loader.py`) accepts any of the
columns `text`, `transcript`, `dialogue`, `content` for the call
text and `label`, `is_scam`, `class` for the label.

If you cannot obtain Wu et al.'s corpus before the deadline, the
backup is the YouTube scam-baiter collection below, which mirrors
the methodology of their "Our-Real" subset.

---

## Tier 2 backup: build your own YouTube scam-baiter corpus

**What it is.** A small (50-100 call) test set built by downloading
videos from public scam-baiter YouTube channels and transcribing
them with Whisper.

**Why.** This is exactly what Wu et al. did for their "Our-Real"
subset, so doing it yourself gives you a comparable evaluation
without needing their cooperation, and it can be done in a few days.

**Step-by-step protocol** lives in
`experiments/SCAMBAITER_PROTOCOL.md`. Summary:

1. Pick 30-50 videos from the channels listed in the protocol.
   Several large channels publish under standard YouTube license;
   using their content for non-commercial academic evaluation is
   accepted research practice (Wu et al. 2024 and Wood et al. 2023
   do this), but verify each channel's terms before publishing.
2. Download with `yt-dlp` (you install it; we do not bundle it).
3. Run `experiments/collect_youtube_scambaiters.py` to Whisper-tiny
   transcribe the audio.
4. Manually annotate which segments are scammer speech vs.
   baiter/host speech. The protocol provides an annotation TSV
   template.
5. Pick 20-30 legitimate calls as negatives, either from the Santa
   Barbara Corpus (see Tier 3 below) or by recording short consented
   conversations.

Output goes to `data/external/youtube_baiters/` and is automatically
picked up by `dataset_loader.load_youtube_baiters`.

---

## Tier 3 (optional, for legitimate-call negatives): Santa Barbara Corpus

**What it is.** A standard corpus of ~60 hours of transcribed
everyday American English conversation. Used by multiple
scam-detection papers (Wu et al., FraudCallDetector) as the negative
class.

**Why.** If you build a sizeable scam-baiter corpus and want a
matched-size legitimate-call set, this is the standard choice.

**How to obtain.** Distributed by the Linguistic Data Consortium
(LDC) and the UC Santa Barbara linguistics department:

> https://www.linguistics.ucsb.edu/research/santa-barbara-corpus

Place at `data/external/sbcorpus/` after registration.

---

## Tier 4 (cross-channel adjacent data, already in your repo)

These are **already downloaded** and do not require external
acquisition. They are used by `experiments/run_cross_channel.py` to
measure how well your call classifier transfers to adjacent fraud
channels.

- `data/real/sms_spam/SMSSpamCollection.tsv` -- 5,574 SMS messages
  (Almeida et al. 2011). 13% spam.
- `data/real/phishing_urls/combined_real_urls.csv` -- 4,001 URLs
  labelled `phishing` / `benign` from PhishTank and other public
  feeds.

For an even larger SMS scam corpus, consider Salman et al. 2022
(arXiv 2210.10451): 153,551 SMS, the largest publicly-available SMS
scam dataset. Distribution method varies; contact Salman/Ikram/Kaafar
at Macquarie if needed.

---

## Recommended sequence given a July deadline

| Day | Action |
|---|---|
| 1 | Tier 1: download BETTER30. Email Wu et al. for Tier 2. |
| 2-5 | Start YouTube scam-baiter collection. 50 videos is enough. |
| 6-7 | Re-run `experiments/run_all.py` with all sources, update paper. |
| any | Run cross-channel experiment (always feasible -- data already present). |

If Wu et al. respond positively in time, add their corpus and
report both. If they do not, the YouTube scam-baiter set is a
defensible substitute that follows the exact methodology of their
"Our-Real" subset.

## What the final paper-evaluation source list should look like

| Source | Calls | Type | Origin |
|---|---|---|---|
| `repo_real` | 23 | call transcripts | in-repo, human-written from scam advisories |
| `better30` | ~30 | call transcripts | Kaggle public |
| `wu2024_corpus` | ~5,000 | call transcripts | Wu et al. 2024 (if obtained) |
| `youtube_baiters` | 50-100 | Whisper-transcribed real calls | YouTube scam-baiter, manual collection |
| `sms_spam` (cross-channel) | 5,574 | SMS messages | Almeida 2011, in-repo |
| `phishing_urls` (cross-channel) | 4,001 | URLs | in-repo, PhishTank etc. |

Headline numbers in the paper should come from the union of
`repo_real`, `better30`, `wu2024_corpus`, and `youtube_baiters`
where available. Cross-channel transfer is a separate section.
