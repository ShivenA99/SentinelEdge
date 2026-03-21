# SentinelEdge -- Technical Assessment

**Date:** 2026-03-21
**Reviewer:** Senior Engineer Code Audit
**Scope:** Full codebase review of all Python, TypeScript, and configuration files
**Codebase size:** ~9,400 lines Python | ~2,000 lines TypeScript/TSX | 565 lines tests

---

## Executive Summary

SentinelEdge is a federated edge AI system for real-time phone call fraud detection.
The project contains solid foundational components -- well-written NLP feature
extraction, a trained XGBoost classifier, a functional FastAPI hub server, and a
polished React demo frontend. However, the gap between what the demo _shows_ and
what the system actually _does_ is significant. The demo runs on pre-written
transcript scripts scored by a hand-tuned heuristic, not the trained model. The
federated learning simulation federates a linear classifier while claiming to
federate XGBoost. There is no Android app. No real audio capture. No real phone
calls.

This is a well-architected prototype with real engineering behind it. It is not
a production system, and it should not be presented as one.

---

## System Status Matrix

| Component | Status | Quality | Notes |
|---|---|---|---|
| 18 handcrafted NLP features | **Working** | Production | Regex, word counting, URL detection. Clean code. |
| TF-IDF vectorizer (500 features) | **Working** | Production | Fitted on ~28K samples, serialized to pickle. |
| XGBoost classifier (100 trees) | **Working** | Trained on synthetic | F1=1.0 on test set -- a red flag, see below. |
| EMA score accumulator (alpha=0.3) | **Working** | Production | Correct implementation, well-tested. |
| Alert engine (risk levels + reasons) | **Working** | Production | 5-tier risk, 13 reason generators from features. |
| Ring buffer (30s circular, 960KB) | **Working** | Production | O(1) write, correct wrap-around logic. |
| Sentence splitter (streaming) | **Working** | Production | Rule-based, handles ellipsis and abbreviations. |
| DP noise injection (Gaussian, eps=0.3) | **Working** | Mathematically correct | Proper sigma calibration, gradient clipping. |
| FedAvg aggregation + Byzantine detection | **Working** | Correct | Z-score outlier detection, safe aggregation. |
| Ed25519 model signing (PyNaCl) | **Working** | Real crypto | Signs model bytes, exposes public key endpoint. |
| Hub server API (FastAPI) | **Working** | Functional | 7 endpoints, delta compression, versioned store. |
| React frontend + WebSocket | **Working** | Polished | Phone simulator, charts, privacy demo panel. |
| 61 unit tests | **Working** | Good but gaps | No integration tests. No end-to-end tests. |
| Audio capture (phone calls) | **Simulated** | N/A | Backend streams pre-written text, no audio input. |
| Whisper transcription | **Simulated** | N/A | Completely skipped; sentences fed directly to features. |
| Phone calls (4 scripts) | **Simulated** | N/A | Pre-written with artificial delays, not real calls. |
| Demo fraud scoring | **Simulated** | Misleading | Uses hand-tuned heuristic, NOT trained XGBoost. |
| Federated learning simulation | **Simulated** | Artificially boosted | Linear classifier, server LR inflated 15%/round. |
| On-device training | **Simulated** | N/A | Mini-batch SGD on synthetic vectors. |
| Training data (28K samples) | **Simulated** | Template-based | ~757 template phrases expanded combinatorially. |
| Privacy demo | **Simulated** | Visual only | Shows random gradient vectors, not real gradients. |
| Federated dashboard charts | **Simulated** | Client-side mock | `generateRoundData()` in TSX, no hub API connection. |
| Android app | **Not Built** | Empty directory | Directory exists, zero Kotlin/Java files. |
| SMS classifier | **Stub** | Script only | Training script exists, no Kaggle data downloaded. |
| URL classifier | **Stub** | Script only | Training script exists, no PhishTank data downloaded. |
| ONNX model export | **Stub** | Script only | Export script exists, never run against trained model. |
| Live microphone capture | **Stub** | Code exists | `live_mic.py` present, not wired into demo. |
| Federating XGBoost | **Needs Research** | Architectural gap | Trees are not differentiable. See analysis below. |
| Real scam transcript data | **Needs Research** | No dataset exists | FTC reports are narrative, not transcripts. |
| Android CallScreeningService | **Needs Research** | OEM-dependent | API varies by Samsung, Pixel, Xiaomi. |
| Whisper Tiny ONNX on mobile | **Needs Research** | Unvalidated | Need latency benchmarks on mid-range Android. |

---

## Detailed Findings

### 1. The XGBoost Model Is Real But Disconnected

The trained XGBoost model (`models/call_fraud_xgb.json`) exists and was trained
properly: 100 estimators, max_depth=6, binary:logistic objective, class-weighted.
The `FraudClassifier` wrapper in `sentinel_edge/classifier/xgb_classifier.py`
correctly loads it and supports both native XGBoost and ONNX Runtime inference.

**The problem:** The demo backend (`demo/backend/main.py`) does not use it.
Line 140 calls `compute_heuristic_score(features)` -- a hand-tuned weighted sum
with manually assigned weights (urgency: 0.08, impersonation: 0.10, etc.) plus
Gaussian noise for "realism." The trained model sits unused.

This is approximately a 5-line fix: import `FraudClassifier`, load the model at
startup, and replace the heuristic call with `classifier.predict_proba(features)`.
The feature vector dimensions will need alignment (the heuristic uses 18 handcrafted
features; the trained model expects 518 = 18 handcrafted + 500 TF-IDF), so the
TF-IDF vectorizer would also need to be loaded in the demo backend.

### 2. F1 = 1.0 on Test Data Should Raise Alarms

The XGBoost classifier achieves perfect F1 on the test set. This is not evidence
of a good model -- it is evidence that the synthetic data is too easy. The training
data comes from `training/generate_synthetic_data.py`, which generates sentences by
randomly sampling from ~757 template phrases across 10 scam categories and various
legitimate categories. The scam templates are dense with keywords that the handcrafted
features directly detect (urgency words, impersonation words, threat patterns). The
TF-IDF features then further memorize the exact vocabulary.

A real scam dataset would contain indirect manipulation, conversational context,
legitimate-sounding preambles, and adversarial phrasings designed to bypass
keyword detection. The model has never seen any of this.

### 3. The Federated Simulation Has Two Fundamental Problems

**Problem A: It federates a linear classifier, not XGBoost.**
`federated/simulate.py` line 319: `logits = X @ self.global_weights` -- this is
logistic regression. The gradient deltas, the FedAvg aggregation, the DP noise
injection -- all of this operates on a weight vector for a linear model. XGBoost
decision trees have no weight vector to average. The simulation's metrics
(accuracy, F1) evaluate a linear classifier and attribute those results to
"federated XGBoost."

**Problem B: Convergence is artificially boosted.**
`federated/simulate.py` line 240: `server_lr = 1.0 + 0.15 * round_num`. The
server-side learning rate increases by 15% each round. By round 5, it is 1.75x.
This forces the linear model to converge aggressively regardless of the quality
of the gradient signal. The comment says this is because "noise averages out" but
that is not a valid justification for an escalating learning rate. Real federated
systems use a constant or decaying server learning rate.

**The deeper issue:** Federating XGBoost is an open research problem. Decision
trees are not differentiable, so you cannot compute gradients and send them to a
hub. Possible approaches:
- (a) Replace XGBoost with a neural network (simplest, loses XGBoost's edge advantages)
- (b) Histogram-based federated XGBoost (active research: see SecureBoost, FedXGB papers)
- (c) Federate only the feature-level statistics, retrain trees centrally
- (d) Use XGBoost locally, federate only a lightweight scoring layer on top

This is the single largest architectural gap in the project.

### 4. The Demo Is a Choreographed Performance

The demo backend reads pre-written transcript files (4 scripts: IRS scam, tech
support scam, bank fraud, legitimate call), streams them one sentence at a time
with `asyncio.sleep(1.5 + random * 1.5)` delays, runs handcrafted feature
extraction (real), scores via heuristic (fake), and sends results over WebSocket.

The frontend receives this stream and renders it beautifully -- phone simulator
UI, real-time score gauge, feature breakdown cards, alert overlays. The UI work
is genuinely polished.

But:
- No audio is captured
- No Whisper transcription occurs
- The fraud score comes from a heuristic, not the trained model
- The `inference_ms` field reported to the frontend is `np.random.uniform(5, 15)` -- randomly generated, not measured
- The privacy demo generates random gradient vectors (`np.random.randn()`), not actual model gradients
- The federated dashboard generates all chart data client-side with `generateRoundData()` -- no hub API connection

### 5. The Android Directory Is Empty

`android/` contains zero files. No Kotlin, no Gradle, no manifest. The
CallScreeningService integration, ONNX Runtime Mobile inference, and on-device
training loop have not been started.

### 6. What IS Good

To be fair, several components are well-engineered:

- **Handcrafted features** (`sentinel_edge/features/handcrafted.py`): 18 features
  with pre-compiled regex patterns, proper word-boundary matching for single words
  vs. substring matching for multi-word phrases. Clean, documented, tested.

- **Ring buffer** (`sentinel_edge/audio/ring_buffer.py`): Correct circular buffer
  with O(1) writes, proper wrap-around reads, sensible memory budget (960KB for
  30s at 16kHz/16-bit). Ready for production use.

- **DP noise injection** (`sentinel_edge/privacy/dp_noise.py`): Correctly
  implements the Gaussian mechanism with proper sigma calibration:
  `sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon`. Gradient clipping
  to bounded L2 norm. Mathematically sound.

- **Model store** (`hub/model_store.py`): Versioned storage with Ed25519 signing
  via PyNaCl, gzip-compressed delta patches for bandwidth efficiency, base64
  encoding for API transport. Well-structured.

- **Alert engine** (`sentinel_edge/classifier/alert_engine.py`): Clean separation
  of concern -- maps scores and features to 5-tier risk levels with human-readable
  reason generation. 13 different reason strings derived from feature values.

- **Hub API** (`hub/server.py`): FastAPI with proper Pydantic schemas, model
  versioning, signature verification endpoint, aggregation status tracking.
  All 7 endpoints are functional.

---

## Known Technical Debt

1. **Demo uses heuristic scorer instead of trained XGBoost model** -- the most
   impactful quick fix in the project (~5-10 lines + loading TF-IDF vectorizer).

2. **`np.trapz` deprecation** -- already fixed to `np.trapezoid` in `hub/validator.py`
   but will break on NumPy < 2.0. No version pin in `requirements.txt`.

3. **CORS allows all origins** -- both `hub/server.py` and `demo/backend/main.py`
   set `allow_origins=["*"]`. Acceptable for local development; unacceptable for
   any deployment.

4. **Hub has zero authentication** -- any `device_id` string is accepted in
   `FederatedUpdate`. No API keys, no mTLS, no JWT. A malicious actor could submit
   poisoned gradient deltas.

5. **No rate limiting on hub endpoints** -- a single client could flood the
   aggregation server with updates.

6. **Hardcoded `/opt/homebrew/bin/python3`** in `.claude/launch.json` -- Mac-specific,
   will fail on Linux or non-Homebrew Mac installs.

7. **Federated simulation convergence is artificially boosted** -- server learning
   rate increases 15% per round. Results should not be cited as evidence of FL
   viability.

8. **No integration tests** -- the 61 unit tests cover individual components but
   nothing tests the full pipeline (feature extraction -> scoring -> accumulation
   -> alert).

9. **No CI/CD pipeline** -- no GitHub Actions, no pre-commit hooks, no linting
   enforcement.

10. **`model_store/signing_key.bin` exists on disk** -- the Ed25519 private key is
    generated and persisted. The `.gitignore` now covers `model_store/` but this
    should be documented prominently.

11. **No TLS requirement documented** -- gradient deltas should only travel over
    HTTPS. No TLS configuration, no certificate pinning documentation.

12. **Synthetic data templates are transparent to the model** -- the XGBoost model
    has memorized the exact phrases from `generate_synthetic_data.py`. It has zero
    generalization to real scam language.

---

## Security Concerns

| Issue | Severity | Status |
|---|---|---|
| CORS `allow_origins=["*"]` on hub and demo | Medium | Open |
| Hub accepts any `device_id` with no auth | High | Open |
| Ed25519 private keys persisted to disk | Medium | Mitigated by `.gitignore` |
| No rate limiting on hub endpoints | Medium | Open |
| No TLS requirement for gradient transport | High | Open |
| Model store directory permissions not locked | Low | Open |
| No input validation on gradient vector dimensions | Low | Mitigated (dimension check in aggregator) |
| Signing key auto-generated silently on first run | Low | By design, but should warn |

---

## Competitive Landscape

| System | Approach | Privacy | FL | Open Source |
|---|---|---|---|---|
| **Google Gemini Nano Scam Detection** | On-device LLM | On-device | No | No |
| **Hiya AI Phone** | Cloud + on-device | Mixed | No | No |
| **Truecaller** | Crowd-sourced DB | Cloud | No | No |
| **RoboKiller** | Pre-answer metadata | Cloud | No | No |
| **SentinelEdge** | Whisper + XGBoost + FL | On-device + DP | Yes (simulated) | Yes |

Google Gemini Nano on Pixel phones is the closest competitor and already ships to
millions of devices. It uses an on-device LLM rather than Whisper + XGBoost, does
not use federated learning, and is a black box. SentinelEdge's differentiation
hinges on (a) federated learning with formal DP guarantees and (b) being open-source
and auditable. Both of these differentiators are currently aspirational, not
demonstrated.

---

## What Is Novel vs. What Is Not

**NOT novel:**
- On-device scam detection (Google ships this today)
- Privacy-first architecture (Google also keeps inference on-device)
- XGBoost for fraud detection (industry standard for tabular fraud)
- Federated learning for fraud (academic papers exist for financial fraud: e.g.,
  "Federated Learning for Credit Card Fraud Detection" and similar)

**POTENTIALLY novel:**
- Federated learning with differential privacy specifically for phone call scam
  detection -- no published paper or shipped product found combining all three
- Open-source auditable system with formal DP guarantee (epsilon=0.3, delta=1e-5)
  vs. Google's black box
- Per-sentence streaming analysis with EMA accumulation (most competitors analyze
  call metadata or caller ID, not real-time transcript content)

The novelty is real but narrow, and currently only exists in the architecture
document, not in working code.

---

## Critical Path to Production

Ordered by dependency and risk:

1. **Resolve the federated XGBoost problem** (Research, 2-4 weeks)
   - Decide: neural net replacement, histogram-based FL, or hybrid approach
   - This blocks all FL claims

2. **Get real training data** (Research + Partnerships, 4-8 weeks)
   - No public labeled phone scam transcript dataset exists
   - Options: FTC complaint narratives (wrong format), LLM-generated transcripts
     (better but still synthetic), telco partnership (ideal but slow)

3. **Wire real XGBoost model into demo backend** (Engineering, 1 day)
   - Load `FraudClassifier` and `TfidfFeatureExtractor` in demo backend
   - Replace `compute_heuristic_score()` with `classifier.predict_proba()`

4. **Connect federated dashboard to real hub API** (Engineering, 2-3 days)
   - Replace `generateRoundData()` with `fetch('/v1/metrics/global')`
   - Replace `generateDeviceData()` with real round status data

5. **Build Android app with CallScreeningService** (Engineering, 4-6 weeks)
   - CallScreeningService for call interception
   - ONNX Runtime Mobile for on-device XGBoost inference
   - Whisper Tiny ONNX for transcription (need latency benchmarks)
   - Test across Samsung, Pixel, Xiaomi -- API varies by OEM

6. **Add authentication to hub** (Engineering, 1 week)
   - Device registration with API keys or mTLS
   - Rate limiting per device

7. **Set up CI/CD** (Engineering, 2-3 days)
   - GitHub Actions: lint, test, type-check
   - Pre-commit hooks for formatting

8. **Real-device testing across Android OEMs** (Testing, 2-4 weeks)
   - VOICE_CALL audio source is blocked by some OEMs
   - Accessibility service fallback needed

---

## Recommendations for New Engineers

1. **Start by running the demo.** It works and looks impressive. Just understand
   what is real behind it and what is choreographed.

2. **Read `sentinel_edge/features/handcrafted.py` first.** It is the best code in
   the repo and sets the quality bar for everything else.

3. **Do not cite the federated simulation metrics.** The linear classifier with
   an inflating learning rate does not prove that federated XGBoost works. It
   proves that FedAvg works on a trivially separable synthetic dataset.

4. **Do not cite the XGBoost F1 = 1.0.** A perfect score on synthetic template
   data tells you nothing about real-world performance.

5. **The fastest wins are:** (a) wire the real model into the demo, (b) connect the
   dashboard to the hub API, (c) add hub authentication. These are each 1-3 day
   tasks that close the gap between what the demo claims and what it does.

6. **The hardest problem is federating XGBoost.** If you solve this cleanly, that
   is a publishable contribution. If you cannot, the project needs to pivot to a
   neural architecture for the federated component.

---

## Summary Verdict

SentinelEdge is a thoughtfully designed system with real engineering in the
components that are built. The NLP features, audio buffer, DP mechanism, model
signing, and hub API are all production-quality code. The React frontend is
polished and demonstrates genuine care for user experience.

The gap is between the demo narrative and the underlying reality. The demo tells
a story of real-time AI fraud detection with federated learning and differential
privacy. The code behind it tells a story of a heuristic scorer running on
pre-written scripts with a federated simulation that does not actually federate
the real model.

This is not a criticism of the engineering -- it is an early-stage prototype and
the scaffolding is sound. But anyone evaluating this project needs to understand
exactly where the real code ends and the simulation begins. This document attempts
to draw that line clearly.
