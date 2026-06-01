# Demonstration Video Script (≤ 2 min 30 s)

EMNLP 2026 demo track requires a screencast video of at most 2.5
minutes. Production quality is not a priority -- a one-take
screencast with audio narration is the expected format. This file
is a shot-by-shot script you can record from in one sitting.

## Tooling

- **Recorder**: QuickTime on macOS, OBS on Linux/Windows, or
  Loom/Zoom local recording. All produce MP4.
- **Mic**: a USB headset or even a laptop mic is fine; clean speech
  matters more than studio quality.
- **Resolution**: 1280x720 minimum, 1920x1080 ideal. Bigger
  resolutions inflate the file with no benefit.
- **Editing**: minimal. Optional 5-10 s top-and-tail trim.

## Pre-recording setup

1. Start the demo locally: `python deploy_server.py` (or whatever
   the repo command is). Open the frontend at `http://localhost:5173`.
2. Open a terminal next to the browser window so the audience can
   see the backend is running locally.
3. Pre-load the IRS scam transcript so you can start it on cue.
4. Have a text file with the alternative pasted-transcript example
   ready to paste in shot 4.
5. Mute system notifications.

## Script

Total budget: **150 s**. Each shot's duration is conservative;
adjust if a single take goes long. The audio narration is in
quotes; the screen direction is in brackets.

### Shot 1 -- Title and problem (0:00 - 0:20)

[Static title card: project name + your name + affiliation.]

> "SentinelEdge is an open-source, on-device scam-call detector
> that runs entirely on a single CPU thread. The contribution is
> not a new model. It's a careful demonstration that a streaming
> per-sentence classifier can detect a scam in roughly 36 seconds
> of speech, using a 0.25 megabyte model. The whole pipeline is
> auditable -- no audio, no transcript, no feature vector ever
> leaves the device."

### Shot 2 -- Architecture (0:20 - 0:35)

[Show Figure 1 from the paper -- the seven-stage pipeline, or the
ARCHITECTURE.md diagram if you have a cleaner version. Highlight
the streaming arrow with cursor.]

> "Audio is captured, windowed, transcribed by Whisper-tiny,
> sentence-split, scored, and smoothed with an exponential moving
> average. Today's demo exercises everything from the sentence
> splitter onward, with the trained model in the live code path."

### Shot 3 -- Live scam call demo (0:35 - 1:25)

[Switch to browser. Click "Start" on the pre-loaded IRS scam call.]

> "I'm going to play the IRS scam call. Each sentence is scored
> independently, and the risk gauge tracks the smoothed score."

[Let it play, narrating once the gauge moves.]

> "By the third sentence, the EMA is climbing past the medium-risk
> threshold. At sentence five, we cross 0.75 and an alert fires."

[Point at the feature breakdown panel.]

> "The right panel shows which of the 18 hand-crafted features
> fired: impersonation, urgency, threat. Below it, the alert reason
> string -- 'urgency language detected; impersonation pattern: IRS'.
> The inference-latency display reads about one millisecond per
> sentence -- this is real `time.perf_counter()`, not a number we
> pulled from training-time benchmarks."

[Show the alert overlay clearly for 1-2 seconds.]

### Shot 4 -- Custom paste (1:25 - 1:50)

[Click "Paste your own". Paste in a short transcript that's
*not* in the sample set. A doctor's-office confirmation works well
as a legitimate-call counterexample.]

> "Pasting a real legitimate call -- a doctor's office
> confirmation -- the EMA stays well below 0.3 and no alert fires.
> The features panel shows zero counts on urgency, threat, and
> impersonation."

### Shot 5 -- Numbers (1:50 - 2:15)

[Switch to the paper or a slide that shows Table 1
(per-call results) and Table 2 (baselines).]

> "On 23 real call transcripts, the streaming detector reaches F1
> 0.93 with perfect precision, at 1.1 milliseconds per sentence on
> a single CPU thread, with a 0.25 megabyte model. The median scam
> is flagged within eight sentences. A fine-tuned DistilBERT
> achieves comparable accuracy but at 42 times the latency and a
> thousand times the disk footprint."

### Shot 6 -- Reproducibility close (2:15 - 2:30)

[Switch to a terminal. Run `python experiments/run_all.py` and
let the first stage start printing. Don't wait for it to finish.]

> "The full evaluation runs in about two minutes on a single CPU
> core. Code, trained model, and a one-command reproduction script
> are at the URL on screen. MIT-licensed. Thank you."

[End card with the demo URL and GitHub URL, hold for 3 s.]

---

## Recording checklist

- [ ] Microphone selected and tested
- [ ] Demo backend + frontend running locally
- [ ] IRS scam transcript pre-loaded
- [ ] Doctor-office legit transcript copied to clipboard
- [ ] Paper Tables 1 and 2 open in a separate tab or PDF reader
- [ ] Terminal window visible behind the browser
- [ ] System notifications muted
- [ ] Recording resolution at least 1280x720

## After recording

1. Trim only the dead air at the start and end (5-10 s each side).
2. Export as MP4 (H.264 + AAC).
3. Either:
   - **Upload to YouTube as unlisted**, paste link into the paper's
     abstract and submission OpenReview metadata, OR
   - **Submit as MP4 supplementary file** on OpenReview if you
     prefer not to publish.
4. Watch it once start-to-finish before submitting. Specifically
   check: audio sync, that the alert moment is clearly visible,
   that the URL at the end is legible.

## Common pitfalls

| Problem | Fix |
|---|---|
| Browser auto-fills with personal data on a paste | use an incognito window |
| Audio narration drifts out of sync with screen | re-record; do not try to fix in editing |
| The "real" inference-latency display reads in a suspicious round number (5 ms, 10 ms) | the demo backend patch is not applied; pull the latest `demo/backend/main.py` |
| Video exceeds 2.5 min | trim Shot 3 first, it's the easiest to compress |
| Mic picks up keyboard click | record narration as a separate track and overlay, or use a mic stand |
