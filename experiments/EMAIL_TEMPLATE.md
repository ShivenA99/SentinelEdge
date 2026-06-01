# Email Template: Requesting the Wu et al. 2024 Corpus

To send to: **Zitong Shen** <esther.shen@connect.polyu.hk>
(corresponding author of *"Combating Phone Scams with LLM-based
Detection: Where Do We Stand?"*, AAAI 2025, arXiv 2409.11643)

CC the other authors as a courtesy:
- Kangzhong Wang
- Youqian Zhang
- Grace Ngai
- Eugene Y. Fu

Subject line: **Data access request: SC/SD/MASC/Our-Real corpora for academic evaluation**

---

```
Dear Dr. Shen and colleagues,

I am [Your Name], a researcher at [Your Institution] working on
lightweight on-device scam-call detection. I am preparing a system
demonstration paper for EMNLP 2026 (deadline July 4, 2026) and your
AAAI 2025 paper "Combating Phone Scams with LLM-based Detection" is
the closest published benchmark to our system.

I am writing to ask whether the aggregated evaluation corpus
described in your paper -- the SC, SD, MASC, Our-Real, and Our-Synt
splits -- is available for non-commercial academic evaluation. We
would use it strictly as a held-out test set; we do not plan to
train on it.

Our system is a streaming, per-sentence scam classifier (TF-IDF +
handcrafted features + XGBoost) intended for on-device deployment.
The work is complementary to your LLM-based approach: we are
quantifying the quality-latency Pareto for the lightweight end of
the spectrum. Citing and comparing against your numbers is essential
to position our contribution honestly.

We would be glad to:
- Sign any data-use agreement you require.
- Acknowledge your group prominently in the paper.
- Share our results on your splits with you before submission so you
  can verify they are reasonable.
- Return any per-sample errors / labelling concerns we discover
  during evaluation.

If sharing the full corpus is not possible, we would also be
grateful for access to just the Our-Real subset, which is the most
critical for a real-world evaluation.

The code, the model checkpoints, and the evaluation harness are
public at [your anonymous repo URL, or "available on request" until
the paper is accepted].

Thank you for your time and for releasing your work.

Best regards,
[Your Name]
[Your Title and Affiliation]
[Your University Email]
[Your Lab / Group Webpage]
```

---

## Notes on practical etiquette

1. **Send to the corresponding author only**; cc-ing all five authors
   on a first email can come across as a broadcast. Reach out to the
   others only if Shen doesn't respond in 10 working days.
2. **Be specific about what you need.** "All five splits if possible;
   Our-Real if not" is much easier to act on than a generic ask.
3. **Be specific about what you give back.** Offering to share your
   results before submission is genuinely useful to them as
   validation of their benchmark.
4. **Mention the deadline but don't lean on it.** "EMNLP July 4" is
   information, not a demand. People in academia know what
   deadlines mean.
5. **Have a fallback ready.** If you don't hear back in two weeks,
   the YouTube scam-baiter collection (Tier 2 backup in
   `PREPARE_DATA.md`) gets you a comparable corpus on your own.
   Don't wait passively; start that collection at the same time as
   sending the email.

## If they share the data

Place the files at `data/external/wu2024_corpus/` as documented in
`PREPARE_DATA.md`. The loader function `load_wu2024` in
`dataset_loader.py` (added by this delivery) will pick them up
automatically.

Run:

```bash
python experiments/dataset_loader.py
```

You should see `wu2024_corpus` populated.

## If they decline or do not respond

Document this in your paper's "Limitations" or "Reproducibility"
section. Specifically:

> "We attempted to obtain the corpus from Shen et al. (2024) for
> direct comparison; this was not possible within the submission
> window. Our YouTube scam-baiter collection (Section X) follows the
> exact methodology of their Our-Real subset (YouTube scam-baiter
> recordings, Whisper transcription) and is a defensible substitute
> for the headline external evaluation."

This kind of explicit acknowledgment is far more credible than
either (a) leaving Wu et al. uncompared or (b) silently substituting
a different corpus.
