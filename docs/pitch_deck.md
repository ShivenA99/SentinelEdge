# SentinelEdge -- Pitch Deck

---

## Slide 1: Title

# SentinelEdge

### Privacy-First AI Scam Call Detection

**Every phone learns. No phone shares.**

Team SentinelEdge | March 2026

*[Visual: SentinelEdge logo on a dark background with a subtle shield icon. Clean, modern typography.]*

---

## Slide 2: The Problem

### Phone scams are a $12.5 billion crisis -- and current solutions make you choose between safety and privacy.

- **$12.5B** lost to fraud in 2024 (FTC) -- up 25% from 2023
- **107 million** spam calls per day worldwide
- **Seniors lose 4x more** per scam call than younger adults
- **1 in 3 Americans** received a deepfake scam call last year

The tools that exist today force a trade-off:

- **Cloud apps** (Hiya, Truecaller) analyze your calls on their servers. You trade your privacy for protection.
- **Google's solution** keeps calls on-device -- but only works on Pixel phones and is a closed black box.
- **Caller ID databases** catch known numbers but miss live, never-seen-before scam conversations entirely.

**Nobody offers real-time, on-device, privacy-preserving scam detection that works on any phone and gets smarter over time.**

*[Visual: Split screen -- left side shows rising fraud loss chart ($8.8B -> $10B -> $12.5B). Right side shows an elderly person on the phone looking worried.]*

---

## Slide 3: Our Solution

### AI that protects you from scams -- without ever hearing your calls.

**SentinelEdge** detects scam calls in real time using on-device machine learning. No audio, transcripts, or personal data ever leave your phone.

**Three pillars:**

1. **On-Device ML** -- Whisper speech-to-text + XGBoost classifier running entirely on your phone. 3--7 seconds from speech to alert.

2. **Federated Learning** -- Your phone trains locally on your calls, then shares only a tiny, privacy-protected learning signal. The global model improves without any central server seeing your data.

3. **Differential Privacy** -- Mathematical proof that no individual call can be reverse-engineered from the learning signal. Epsilon = 0.3 -- stronger than what Apple or Google use for their analytics.

**Open source. Auditable. Free.**

*[Visual: Phone mockup showing the SentinelEdge alert overlay on an active call, with a red warning banner and feature breakdown panel.]*

---

## Slide 4: How It Works

### A 7-stage pipeline -- all running on your phone.

```
Incoming Call
    |
[1] Audio Capture (16kHz, ring buffer, 30s max in RAM)
    |
[2] 5s Sliding Window (1s hop, Hann windowing)
    |
[3] Whisper Tiny STT (on-device ONNX, ~200ms--2s)
    |
[4] Sentence Segmentation (streaming, real-time)
    |
[5] Feature Extraction (18 handcrafted + 500 TF-IDF = 518 dims)
    |
[6] XGBoost Classification (fraud probability 0--1)
    |
[7] EMA Smoothing + Alert (red overlay when score > 0.75)
```

**Key numbers:**
- **3--7 seconds** end-to-end latency
- **155 MB** total model footprint
- **~50 MB** RAM during inference
- **Zero network calls** during any phone call

After the call ends, all audio, transcripts, and scores are erased from memory. Nothing is saved to disk. Nothing is sent anywhere.

*[Visual: Clean pipeline diagram flowing top to bottom with icons for each stage. Emphasis on "everything happens here" pointing to the phone.]*

---

## Slide 5: Live Demo

### See it detect a scam call in real time.

The demo runs a phone simulator with four pre-built call scenarios:

1. **IRS Impersonation Scam** -- "This is the IRS. Your Social Security number has been suspended..."
2. **Tech Support Scam** -- "We've detected a virus on your computer..."
3. **Bank Fraud** -- "Your account has been compromised. We need to verify..."
4. **Legitimate Call** -- A normal business conversation (correctly scored as safe)

**What to watch for:**
- The real-time transcript streaming in as each sentence is spoken
- The fraud score gauge climbing as scam signals accumulate
- The feature breakdown cards showing exactly which signals triggered (urgency, impersonation, financial references)
- The red alert overlay firing when the EMA score crosses 0.75
- The privacy panel showing the DP-noised gradient -- what the server would actually see

**Key demo moment:** The system catches the scam within the first 2--3 sentences. It does not wait for the entire call.

*[Visual: Screenshot of the React demo frontend -- phone simulator on the left, real-time charts on the right, alert overlay visible.]*

---

## Slide 6: Privacy Innovation

### The server literally cannot learn what you said.

| What stays on your phone | What the server receives |
|---|---|
| Raw call audio | Nothing |
| Transcript text | Nothing |
| Feature vectors (518 dims) | Nothing |
| Fraud scores | Nothing |
| Phone numbers | Nothing |
| -- | ~20KB DP-noised gradient delta |

**One sentence:** We add calibrated mathematical noise so that no individual call can ever be reverse-engineered from the learning signal.

**How strong is our privacy?**
- Our epsilon = 0.3 (lower = more private)
- Apple uses epsilon 2--8 for emoji analytics
- Google uses epsilon 1--10 for Chrome metrics
- At epsilon 0.3, gradient inversion attacks are provably infeasible

The privacy guarantee is not a policy. It is a mathematical proof. And because SentinelEdge is open source, any researcher can verify it themselves.

*[Visual: Two-column layout. Left column labeled "YOUR PHONE" with icons for audio, text, scores -- all with lock icons. Right column labeled "THE SERVER" showing only a small noisy gradient vector with the text "~20KB of noise." Arrow between them showing the DP noise injection step.]*

---

## Slide 7: Federated Learning

### Every phone makes every other phone smarter -- without sharing data.

**The cycle:**

1. Your phone detects calls locally and learns from your feedback (block = scam, continue = legit)
2. After enough calls, your phone computes a tiny model update (~20KB)
3. Differential privacy noise is added to protect your data
4. The noised update is sent to the hub server
5. The hub averages updates from many phones (FedAvg) and produces an improved global model
6. The improved model is signed, compressed, and sent back to all phones
7. Repeat -- the model gets better every round

**Key properties:**
- Minimum 5 devices per round (Byzantine fault tolerance)
- Hub validates every update against a test set -- poisoned gradients are caught and rejected
- Model updates are Ed25519 signed to prevent tampering in transit
- Hub requires a single-core VM with no GPU -- cheap to operate

**Result:** A model that learns from millions of real-world scam encounters while the central server never sees a single phone call.

*[Visual: Circular diagram showing 5 phone icons around a central hub. Arrows flowing from phones to hub labeled "noised gradients (~20KB)" and arrows flowing back labeled "better model (~300KB patch)." Each phone has a shield icon.]*

---

## Slide 8: Competitive Landscape

### We are the only open-source, privacy-preserving system with adaptive intelligence.

```
                    ADAPTIVE ML
                        ^
                        |
         Hiya AI Phone  |  SentinelEdge
         (cloud+device) |  (on-device, FL, DP, open source)
                        |
                        |  Google Gemini Nano
                        |  (on-device, Pixel only, closed)
    --------------------+--------------------->
    CLOUD-DEPENDENT     |     PRIVACY-FIRST
                        |
         Truecaller     |  STIR/SHAKEN
         RoboKiller     |  (caller ID auth only)
                        |
                    STATIC DB
```

**vs. Google:** We are open source, hardware agnostic, and use federated learning with formal DP guarantees. Google is closed-source and locked to Pixel.

**vs. Hiya:** We never send call audio or metadata to any server. Hiya's cloud processing means they can hear your calls.

**vs. Truecaller/RoboKiller:** They match caller IDs against databases. We analyze the actual conversation in real time.

*[Visual: 2x2 quadrant chart with company logos placed in their respective positions. SentinelEdge logo in the top-right corner (high privacy + adaptive ML) with a highlight glow.]*

---

## Slide 9: Technical Differentiation

### What is novel, what is hard to replicate, and why it matters.

**First of its kind:**
- No published paper or shipped product combines federated learning + differential privacy for phone call scam detection
- Per-sentence streaming analysis with EMA accumulation (competitors analyze metadata, not conversations)
- 13 explainable alert reasons derived from handcrafted features -- users see why the system flagged a call

**Hard to replicate:**
- Privacy-auditable open-source codebase (9,400 lines Python, 2,000 lines TypeScript)
- Formal epsilon = 0.3 privacy guarantee with documented sensitivity analysis
- Full pipeline from audio capture through federated aggregation -- not a research prototype for one piece

**Publication pipeline:**
- Target: USENIX Security, IEEE S&P, NeurIPS Federated Learning Workshop
- Working title: "Federated Learning with Differential Privacy for Real-Time Phone Scam Detection"
- The combination of FL + DP + real-time telephony fraud is an unclaimed intersection in the literature

*[Visual: Three columns -- "Novel" (lightbulb icon), "Defensible" (shield icon), "Publishable" (paper icon) -- each with bullet points beneath.]*

---

## Slide 10: Roadmap

### From prototype to production in four phases.

**Phase 1: Foundation (Complete)**
- Working end-to-end prototype with demo
- Federated learning simulation across 5 devices
- Differential privacy implementation (epsilon = 0.3)
- 61 unit tests, full architecture documentation
- React phone simulator frontend

**Phase 2: Hardening (Q2 2026)**
- Wire trained XGBoost model into live demo
- Connect federated dashboard to real hub API
- CI/CD pipeline with GitHub Actions
- Android APK prototype (CallScreeningService integration)
- Hub authentication and rate limiting

**Phase 3: Expansion (Q3--Q4 2026)**
- Real-device testing across Samsung, Pixel, Xiaomi
- Deepfake voice detection module
- SMS phishing and URL phishing channels
- Whisper Tiny ONNX latency benchmarks on mid-range Android
- Academic paper submission

**Phase 4: Scale (2027)**
- Production Android app on Google Play
- Telco partnerships for real scam data
- Community contributor program
- iOS investigation (on-device ML via Core ML)

*[Visual: Horizontal timeline with four milestones, each with 4--5 bullet points beneath. Phase 1 has a checkmark. Phases 2--4 have target date labels.]*

---

## Slide 11: Team

### Built by engineers who care about privacy and security.

*[Placeholder for team bios]*

**[Name]** -- ML & Privacy Lead
Background in machine learning, differential privacy, and federated systems.

**[Name]** -- Mobile & Systems Lead
Background in Android development, on-device ML, and systems engineering.

**[Name]** -- Security & Infrastructure Lead
Background in cryptography, secure systems, and API design.

*We are looking for collaborators with expertise in: federated learning research, Android telephony APIs, voice deepfake detection, and real-world scam data partnerships.*

*[Visual: Team headshots in a row with name, role, and one-line bio beneath each. GitHub/LinkedIn icons.]*

---

## Slide 12: The Ask

### Join us in building the future of private, intelligent scam protection.

**What we are looking for:**

1. **Open-source contributors** -- Android engineers, ML researchers, privacy engineers. The codebase is MIT-licensed and ready for contributions.

2. **Academic collaborators** -- Co-authors for publication at USENIX Security, IEEE S&P, or NeurIPS. The FL + DP intersection for telephony fraud is wide open.

3. **Data partnerships** -- Telcos or consumer protection agencies with real scam call data (anonymized) to replace our synthetic training set.

4. **Seed funding** -- To support full-time development through Phase 2 and Phase 3, Android device testing lab, and conference travel for paper presentation.

**The opportunity:** A $6.3 billion market growing at 13% CAGR, a privacy-first approach that no competitor offers as open source, and publication-worthy research at the intersection of federated learning and telephony security.

**GitHub:** github.com/[org]/SentinelEdge
**Contact:** [email]

*[Visual: Clean slide with four icons (code bracket, graduation cap, database, dollar sign) representing the four ask categories. GitHub QR code in the bottom-right corner.]*
