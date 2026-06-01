# YouTube Scam-Baiter Corpus: Collection Protocol

This protocol describes how to assemble a 50-100 call test corpus
from public YouTube scam-baiter content for evaluating
SentinelEdge. It mirrors the methodology used by Wu et al. 2024
(*"Combating Phone Scams with LLM-based Detection"*, AAAI 2025) for
their "Our-Real" subset and by Wood et al. 2023 (*"An analysis of
scam baiting calls"*, Macquarie University).

## What you are building

A test corpus at `data/external/youtube_baiters/` consisting of:

```
data/external/youtube_baiters/
├── annotations.tsv               # human-curated label file
├── transcripts/                  # Whisper output, one .json per call
│   ├── call_001.json
│   ├── call_002.json
│   └── ...
└── audio/                        # source audio (gitignored, optional)
    └── ...
```

`annotations.tsv` is the file the loader reads. Each row is one call.

| Column | Type | Description |
|---|---|---|
| `call_id` | str | unique ID, e.g. `yt_001` |
| `label` | int | 1 = scam, 0 = legit / non-scam segment |
| `category` | str | scam type, e.g. `tech_support`, `irs`, `refund` |
| `youtube_url` | str | source video URL |
| `start_sec` | float | clip start in source video |
| `end_sec` | float | clip end |
| `scammer_speaker` | str | which Whisper speaker label is the scammer (`SPK0`, `SPK1`, or blank for whole-clip) |
| `transcript_file` | str | path to JSON transcript, e.g. `transcripts/call_001.json` |
| `notes` | str | annotator notes |

## Why scam-baiter content is a valid test source

Scam-baiter videos are recordings of scammers interacting with
people pretending to be victims. The scammer's speech is genuine
fraudulent conversation; they are unaware the call is being baited.
This is exactly the input modality the deployed system would see.

Three peer-reviewed papers have used this source:

- Wood et al. 2023 (Macquarie / arXiv 2307.01965): "we draw from the
  community of 'scam baiters' on YouTube... scammers are unaware that
  they are not speaking to a true scam victim for the bulk of the call"
- Wu et al. 2024 ("Our-Real" subset, AAAI 2025)
- Basta et al. 2025 (Macquarie / arXiv 2503.07036)

Three caveats your paper should acknowledge:

1. The baiter's speech is *not* a real victim and should be excluded
   from the evaluation. Annotate which speaker is the scammer.
2. Baiter audiences self-select for entertaining calls; the
   distribution skews toward longer, more theatrical scams.
3. Channels may apply audio post-processing (music, sound effects).
   Discard segments where music dominates.

## Channels to source from

| Channel | Style | Volume |
|---|---|---|
| Kitboga | Tech-support scams, gift-card scams; high-volume | very large |
| Pleasant Green | Refund scams, IRS scams | large |
| Jim Browning | Long-form investigations, IRS / refund scams | large |
| Scammer Payback | IRS, refund, romance | large |
| Trilogy Media | Tech support, refund | medium |
| Scambaiter | Tech support, dating | medium |

Each has hundreds of public uploads. **Aim for diversity**: pull
videos from at least 3 channels and at least 4 distinct scam
categories. A 60-call corpus drawn from one channel can be
characterised by your reviewer as overfitting to that channel's
production style. A 60-call corpus drawn from 4 channels and 5
categories is robust.

## Legal and ethical notes

These channels publish under the standard YouTube licence. Academic
non-commercial use for evaluation is broadly accepted research
practice in this area (see the three papers above). You should
nonetheless:

- Cite the channel and the video URL in your annotations.
- Do **not** redistribute the audio or full transcripts. The
  `audio/` directory should be in `.gitignore`.
- Redistribute only the `annotations.tsv` (a list of URLs, timing,
  and labels) so that any reader can reconstruct the corpus from
  scratch by re-running the collection script.
- If you publish the per-sentence transcripts (which Wu et al.
  arguably did), redact any personal information mentioned by the
  scammer (often fake names, but sometimes also numbers).
- Get your institution's IRB / ethics approval if your university
  requires it for analysis of recorded conversations.

## Step-by-step

### Step 1 -- Install tooling

```bash
# Audio download
pip install yt-dlp

# Whisper transcription (CPU is fine; tiny.en is the right model)
pip install openai-whisper
# or pip install faster-whisper  -- 3x faster, same quality

# ffmpeg for audio extraction (yt-dlp needs it)
# On macOS: brew install ffmpeg
# On Ubuntu: sudo apt install ffmpeg
```

### Step 2 -- Pick videos and clip ranges

Manually browse the channels above and select 30-50 videos. For
each, identify the timestamp range that contains continuous scammer
conversation (skip intros, sponsor segments, baiter commentary
overlaid on the call audio).

Create a file `data/external/youtube_baiters/video_list.tsv`:

```
video_id    youtube_url                                  start_sec   end_sec     category        notes
yt_001      https://www.youtube.com/watch?v=XXXXXXXXXXX  120         480         tech_support    Kitboga; clean call audio after intro
yt_002      https://www.youtube.com/watch?v=YYYYYYYYYYY  60          540         refund          Pleasant Green
...
```

### Step 3 -- Download and transcribe

```bash
python experiments/collect_youtube_scambaiters.py \
    --video-list data/external/youtube_baiters/video_list.tsv \
    --output-dir data/external/youtube_baiters/ \
    --whisper-model tiny.en
```

This downloads each video's audio (mp3, ~5 MB each), trims to the
specified range, runs Whisper-tiny.en, and writes one JSON
transcript per call. Wall-clock: roughly 30-90 s per 5-min clip on
a single CPU core.

The script supports `--resume`: re-running skips videos already
transcribed.

### Step 4 -- Annotate

Open each `transcripts/call_NNN.json`, look at the Whisper output
(it includes per-sentence timestamps), and decide:

- Is this a scam call? (Almost always yes for the channels above,
  but verify -- sometimes channels include legitimate phone
  conversations as filler.)
- Which speaker is the scammer? Whisper does not do diarization by
  default; if the call has speaker turns, the script attempts a
  best-effort diarization but you should verify by reading the
  first few utterances.
- What scam category? Pick from a small fixed vocabulary:
  `irs`, `ssa`, `tech_support`, `refund`, `bank_fraud`,
  `prize_lottery`, `crypto_investment`, `grandparent`, `utility_shutoff`,
  `other`.

Fill in `data/external/youtube_baiters/annotations.tsv` (the loader
needs this file).

For a corpus of 50 calls this is roughly 3-5 hours of annotation
work for one person.

### Step 5 -- Add legitimate-call negatives

You need negative examples too. Three options:

(a) **Use existing `repo_real` legit calls**: the 8 legit transcripts
already in the repo can be reused.

(b) **Use Santa Barbara Corpus**: register at UCSB, download SBC, pull
~20-30 short phone-conversation excerpts. Place at
`data/external/sbcorpus/` and use the SBC loader (`load_sbcorpus`, to
be added once the corpus is in place).

(c) **Synthesize light**: generate 20-30 legitimate-call transcripts
with an LLM, manually screen for realism. Lower quality than (a) or
(b) but fastest. Annotate these in `annotations.tsv` with `label=0`.

### Step 6 -- Verify and include in evaluation

```bash
python experiments/dataset_loader.py
```

You should see `youtube_baiters` populated with your annotated set.

```bash
python experiments/run_all.py --sources repo_real better30 youtube_baiters
```

This regenerates every paper number with the new corpus included.

## Target corpus size

| Calls (scam + legit) | Strength as evidence | Effort |
|---|---|---|
| 20 + 10 | weak, save for ablation only | 1 day |
| 50 + 30 | sufficient for a paper, comparable to Wu et al.'s Our-Real | 3-5 days |
| 100+ + 50 | strong, near publishable benchmark | 1-2 weeks |

Aim for the 50+30 row given a five-week timeline. Anything larger
is a bonus.

## Failure modes to watch for

| Symptom | Cause | Fix |
|---|---|---|
| Whisper output is gibberish | music or sound effects in the audio | manually trim to the scammer-speech segment only |
| Whisper merges scammer + baiter into one transcript | no diarization | use `--diarize` flag (if pyannote installed) or annotate manually |
| Classifier scores all calls as scam | sampled too narrowly from one channel/category | broaden sources |
| Annotation drift between annotators | no rubric | use the fixed `category` vocabulary and have one person do all 50 in one sitting |
