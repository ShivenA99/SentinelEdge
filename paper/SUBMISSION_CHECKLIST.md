# EMNLP 2026 System Demonstrations -- Submission Checklist

Source: <https://2026.emnlp.org/calls/demos/>

## Hard deadlines

| Milestone | Date | Status |
|---|---|---|
| Paper submission | **Friday, July 10, 2026** (11.59 pm AoE) | open |
| Notification | Friday, August 20, 2026 | -- |
| Camera ready | Sunday, August 30, 2026 | -- |
| Conference | October 24-29, 2026, Budapest | -- |

**No rebuttal stage.** The paper must be defensible as-submitted.

## Required submission components

### 1. Paper PDF

| Requirement | Status | Notes |
|---|---|---|
| ≤ 6 pages content (longer = desk reject) | [draft fits, verify after compilation] | currently in `paper/sentineledge_emnlp2026.tex` |
| Unlimited references | OK | bib file `paper/sentineledge.bib` |
| Optional ≤ 2-page appendix | not used | |
| Unlimited ethics/broader-impact section | INCLUDED | `Ethics and Broader Impact` section after Conclusion |
| Official EMNLP 2026 style file | TODO | `.tex` currently uses EMNLP2023.sty; replace with the 2026 style once the ACL repo publishes it (https://github.com/acl-org/emnlp-2026) |
| Single-blind: authors named | INCLUDED (placeholder) | replace `[Last Name]`, `[Affiliation]`, `[email]` with real values |
| PDF format | from LaTeX | |
| Original, unpublished work | OK | not previously published |

### 2. Demonstration video (≤ 2.5 min)

| Requirement | Status |
|---|---|
| Screencast or user-interaction video | TODO -- script in `paper/DEMO_VIDEO_SCRIPT.md` |
| With audio narration (recommended) | TODO |
| MP4 if not on YouTube | TODO |
| YouTube link in the paper if uploaded publicly | TODO -- placeholder is `[URL]` in `\maketitle` block |

### 3. Live demo URL or installable package

**Hard requirement -- no link = desk reject.**

| Requirement | Status |
|---|---|
| Public URL to running demo, OR installable package link | TODO -- deployment guide in `paper/LIVE_DEMO_DEPLOYMENT.md` |
| URL stated in the paper | placeholder in abstract + Availability section |
| Must remain reachable until at least the camera-ready deadline | required; budget for hosting through end of October 2026 |

If hosting is impossible (uncommon for a web-based demo), the
authors must explicitly state why in the paper. SentinelEdge is a
plain FastAPI + React app, so a public URL is the right choice.

## Required paper content

The call explicitly asks every demo paper to address these eight
questions. Current draft status:

| # | Question | Where addressed |
|---|---|---|
| 1 | What problem does the proposed system address? | §1 "Problem" |
| 2 | Why is the system important and what is its impact? | §1 "Importance and target audience" |
| 3 | What is the novelty? | §1 "Novelty" |
| 4 | Who is the target audience? | §1 "Importance and target audience" |
| 5 | How does the system work? | §2 System |
| 6 | How does it compare with existing systems? | §4 "Comparison to existing systems" + Table 2 |
| 7 | How is the system licensed? | §6 "Code and license" -- MIT |
| 8 | How was the system evaluated? Were user studies/human evaluation? | §4 Evaluation, §5 Robustness; §6 "Evaluation scope" notes no user studies and why |

## Submission system

OpenReview. The link will be posted on the EMNLP 2026 demos page at
least two weeks before the July 10 deadline. Sign in with the same
OpenReview account used for the main conference.

## Reviewing

Single-blind, two reviewers per paper (typical for ACL demo tracks),
no rebuttal. Reviewers will:

- Read the paper for the eight required answers above
- Watch (some of) the video
- Click the live demo URL and try it
- Skim the released code

Any of those failing -- demo down, video not working, code not
runnable -- is reason for rejection that the authors cannot rebut.

## Pre-submission self-review

A useful internal pass before clicking submit. Check each item:

| Pre-submission item | Status |
|---|---|
| Paper PDF compiles cleanly with the official 2026 style file | [verify] |
| Page count ≤ 6 excluding references and ethics section | [verify -- run wc on output] |
| Author block has real names and affiliations (NOT "Anonymous") | replace placeholders |
| Live demo URL works in an incognito browser, from outside your network | [test] |
| Video URL accessible from an anonymous browser | [test] |
| `experiments/run_all.py` runs to completion in ≤ 10 minutes on a fresh checkout | [verify -- runs in ~2 min on the working copy] |
| README in the repo gives the one-command reproduce instruction | already present in `paper/README.md`; copy to repo root |
| MIT LICENSE file at repo root | [verify -- not present in current `experiments/` tree; add] |
| All figures referenced in the .tex compile (5 PDFs in `paper/figures/`) | [verify after compile] |
| Bibliography includes real DOIs / arXiv IDs, not placeholder author names | [check `sentineledge.bib` -- some entries still say "[author]"; fix before submission] |
| Ethics statement addresses dual-use, data sourcing, redaction | INCLUDED |
| Reproducibility statement points at a public repo URL (not "available on request") | replace `[URL]` placeholder |

## Things to NOT do

- Do not anonymise. EMNLP 2026 demos are single-blind.
- Do not exceed 6 pages of content. Long paper = desk reject.
- Do not skip the video. Submissions without a screencast cannot be
  reviewed effectively.
- Do not omit the live demo URL. Strict requirement this year.
- Do not claim results not in the released `results/` JSON files.
- Do not use the EMNLP 2023 style file in the final submission --
  switch to 2026 once it ships.

## Final pre-flight (the morning of July 10)

1. Re-run `experiments/run_all.py` end-to-end on the public commit;
   confirm every number in the paper still matches.
2. Open the live demo URL on a fresh browser.
3. Click through all nine pre-loaded calls.
4. Watch the video at 1.5x speed once to make sure audio sync is
   intact.
5. Submit on OpenReview at least four hours before AoE midnight.
