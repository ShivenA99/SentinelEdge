# SentinelEdge Architecture

> Federated edge AI system for real-time phone call fraud detection.
> Last updated: 2026-03-21

---

## 1. System Overview

SentinelEdge splits fraud detection into two tiers: **edge devices** that run inference locally on Android phones, and a **hub server** that orchestrates federated model improvement without ever seeing user data.

```
┌─────────────────────────────────────────────┐
│                Edge Device (Android)         │
│                                              │
│  Call Audio -> Ring Buffer -> Whisper STT    │
│       -> Sentence Splitter -> Feature Extract│
│       -> XGBoost Classify -> EMA -> Alert    │
│                                              │
│  Local Training Buffer (500 samples, encrypted)
│       -> Fine-tune -> Gradient Delta         │
│       -> DP Noise (e=0.3) -> Upload (~20KB)  │
└──────────────────┬──────────────────────────┘
                   │ TLS (gradient deltas only)
                   v
┌─────────────────────────────────────────────┐
│              Hub Server                      │
│                                              │
│  Collect K updates -> FedAvg -> Validate     │
│       -> Sign (Ed25519) -> Distribute        │
│                                              │
│  Stores: model versions + validation set     │
│  Never sees: audio, text, phone numbers      │
└─────────────────────────────────────────────┘
```

**Design principles:**

1. **Zero raw data leaves the device.** The hub never sees audio, transcripts, phone numbers, or per-call scores.
2. **Inference is fully offline.** No network calls during a phone call. Latency depends only on local compute.
3. **Federated learning improves the model** across the fleet while differential privacy (epsilon=0.3) makes gradient inversion provably infeasible.
4. **Lightweight hub.** A single-core VM with no GPU handles aggregation for thousands of devices.

---

## 2. Edge Device Architecture

### On-Device Components

| Component | Technology | Size | Latency | Purpose |
|---|---|---|---|---|
| Audio capture | `AudioRecord` + `CallScreeningService` | N/A | ~0 ms | 16 kHz mono PCM |
| Ring buffer | In-memory circular buffer | 960 KB | N/A | Holds 30 s of audio, never saved to disk |
| Speech-to-text | whisper-tiny.en, ONNX Mobile | 150 MB | 200 ms -- 2 s | 5 s sliding windows, 1 s hop |
| Sentence splitter | Rule-based punctuation + pause detection | < 1 KB | ~1 ms | Breaks STT output into sentence units |
| Feature extractor | TF-IDF (500 terms) + 18 handcrafted | 2 MB | ~3 ms | Produces 518-dim feature vector |
| Fraud classifier | XGBoost INT8, ONNX runtime | 5 MB | ~8 ms | Outputs fraud probability 0.0 -- 1.0 |
| Score accumulator | EMA (alpha=0.3) | < 1 KB | < 1 ms | Smooths single-sentence noise |
| Alert overlay | Android `SYSTEM_ALERT_WINDOW` | N/A | < 1 ms | Red alert overlay when EMA > 0.75 |

**Total footprint:** ~155 MB disk, ~50 MB RAM during active inference.

### Pipeline Stages and Timing

The on-device pipeline has seven sequential stages per window:

| Stage | Operation | Latency | Notes |
|---|---|---|---|
| 1 | Audio capture into ring buffer | ~0 ms | Continuous background write at 16 kHz |
| 2 | Extract 5 s window (1 s hop) | < 1 ms | Copy from ring buffer, apply Hann window |
| 3 | Mel spectrogram computation | ~10 ms | 80-bin mel filterbank via FFT (NumPy/SciPy) |
| 4 | Whisper inference | 200 ms -- 2 s | Bottleneck. Depends on device SoC |
| 5 | Sentence splitting + feature extraction | ~4 ms | Per completed sentence |
| 6 | XGBoost classification | ~8 ms | INT8 ONNX, constant time per sentence |
| 7 | EMA update + alert decision | < 1 ms | Triggers overlay if threshold crossed |

**Total end-to-end latency: 3 -- 7 s** from speech to alert. The bottleneck is Whisper (stage 4). All other stages combined take under 25 ms.

### Why 5-Second Sliding Windows

Whisper-tiny handles short segments well, and 5 s provides fast incremental updates while capturing complete sentences in most cases. Below 3 s, transcription accuracy degrades noticeably because Whisper loses context. The 1 s hop means 4 s of overlap between consecutive windows, ensuring no speech is clipped at window boundaries. This also allows the system to self-correct if a word was misrecognized at a boundary -- it will appear cleanly in the next window.

### Why Per-Sentence Classification

Each sentence is classified independently in constant time. This has two critical advantages over full-transcript TF-IDF:

1. **Constant-time inference.** Feature extraction and classification cost the same whether the call is 10 seconds or 10 minutes. Full-transcript TF-IDF grows linearly with call length.
2. **Instant scam detection.** The system catches a scam signal the moment a suspicious sentence is spoken, rather than waiting for enough transcript to accumulate. A single sentence like "Your SSN has been compromised" triggers an alert within seconds.

---

## 3. Audio Pipeline Deep Dive

### Data Flow

```
AudioRecord (16 kHz mono PCM)
    |
    v
Ring Buffer (960 KB, holds 30 s)
    |
    v
Windowing (5 s window, 1 s hop, Hann window)
    |
    v
Mel Spectrogram (80-bin filterbank, FFT via NumPy/SciPy)
    |
    v
Whisper-tiny.en (ONNX Mobile runtime)
    |
    v
Sentence Splitter (punctuation + 500 ms pause detection)
    |
    v
Per-sentence feature extraction + classification
```

### Ring Buffer Design

The ring buffer holds exactly 30 seconds of 16 kHz mono PCM audio (480,000 samples, 960 KB). It uses O(1) writes with a head pointer that wraps around. The buffer is never flushed to disk and is zeroed when the call ends. This guarantees that at most 30 seconds of audio exist in memory at any time.

### Windowing Strategy

Windows are extracted every 1 second (the hop size). Each window is 5 seconds (80,000 samples). A Hann window function is applied to reduce spectral leakage before FFT. The windowing module uses pure NumPy and SciPy -- librosa is intentionally excluded to reduce the dependency footprint on mobile.

### Mel Spectrogram

Standard 80-bin mel filterbank applied to the STFT output. Parameters match Whisper's expected input format (16 kHz sample rate, 400-sample FFT window, 160-sample hop). The spectrogram is computed as a log-mel representation and passed directly to the Whisper encoder.

### Sentence Splitter

The sentence splitter operates on streaming Whisper output. It segments on two signals: punctuation tokens (period, question mark, exclamation) and pause gaps exceeding 500 ms detected from Whisper timestamps. Incomplete sentences are held in a buffer until a boundary is detected.

### Key Files

- `sentinel_edge/audio/ring_buffer.py` -- Circular buffer with O(1) writes, zeroed on call end
- `sentinel_edge/audio/windowing.py` -- FFT, mel filterbank (pure NumPy/SciPy, no librosa)
- `sentinel_edge/audio/transcriber.py` -- Whisper ONNX wrapper with lazy model loading
- `sentinel_edge/audio/sentence_splitter.py` -- Streaming sentence segmentation

---

## 4. Fraud Classifier Architecture

### 18 Handcrafted Features

These features are extracted from each sentence independently:

| # | Feature | Type | Scam Signal Strength |
|---|---|---|---|
| 1 | `urgency_count` | int | Very high |
| 2 | `action_count` | int | High |
| 3 | `financial_count` | int | High |
| 4 | `impersonation_count` | int | Very high |
| 5 | `has_url` | binary | Medium |
| 6 | `has_shortened_url` | binary | High |
| 7 | `has_verify_pattern` | binary | Medium |
| 8 | `has_threat` | binary | Very high |
| 9 | `has_prize` | binary | High |
| 10 | `has_account_ref` | binary | High |
| 11 | `dollar_sign` | binary | Medium |
| 12 | `has_phone_number` | binary | Medium |
| 13 | `char_count` | int | Low |
| 14 | `word_count` | int | Low |
| 15 | `avg_word_len` | float | Low |
| 16 | `url_count` | int | Medium |
| 17 | `exclamation_count` | int | Medium |
| 18 | `caps_ratio` | float | Medium |

These 18 features are concatenated with 500 TF-IDF unigram/bigram features for a total of **518 dimensions** per sentence.

The handcrafted features provide interpretable, high-signal indicators. TF-IDF captures vocabulary patterns that do not map neatly to any single handcrafted rule. The combination outperforms either set alone by 4-6% F1 in evaluation.

### XGBoost Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Sufficient tree count for 518 features without excessive model size |
| `max_depth` | 6 | Prevents overfitting on limited per-device sample counts |
| `learning_rate` | 0.1 | Standard rate; balances convergence speed and stability |
| `subsample` | 0.8 | Row subsampling reduces variance |
| `colsample_bytree` | 0.8 | Feature subsampling per tree reduces correlation between trees |
| Quantization | INT8 dynamic range | ~4x compression vs FP32, < 1% accuracy loss |

The classifier supports dual backends: native XGBoost for training and development, and ONNX Runtime for mobile inference. The ONNX model is exported with INT8 dynamic-range quantization, reducing the model from ~20 MB to ~5 MB with negligible accuracy impact.

### Score Accumulation

Raw per-sentence scores are noisy. A single benign sentence in a scam call might score low. The EMA accumulator smooths this:

```
ema_score = alpha * new_score + (1 - alpha) * prev_ema_score
```

With alpha=0.3, the accumulator responds to new evidence within 2-3 sentences while resisting single-sentence false positives. The alert fires when `ema_score > 0.75`.

### Five-Tier Risk Levels

| Risk Level | Score Range | UI Treatment |
|---|---|---|
| `critical` | >= 0.9 | Red overlay + vibration + auto-record suggestion |
| `high` | 0.75 -- 0.9 | Red overlay |
| `medium` | 0.5 -- 0.75 | Yellow banner |
| `low` | 0.25 -- 0.5 | Subtle indicator |
| `safe` | < 0.25 | No UI |

### Key Files

- `sentinel_edge/features/handcrafted.py` -- 18 handcrafted feature extractors
- `sentinel_edge/features/tfidf.py` -- TF-IDF vectorizer wrapper
- `sentinel_edge/features/feature_pipeline.py` -- Combines handcrafted + TF-IDF into 518-dim vector
- `sentinel_edge/classifier/xgb_classifier.py` -- Dual backend (native XGBoost + ONNX Runtime)
- `sentinel_edge/classifier/score_accumulator.py` -- EMA smoothing logic
- `sentinel_edge/classifier/alert_engine.py` -- 5-tier risk levels and alert decisions

---

## 5. Data Lifecycle During a Call

Every data type has a defined lifetime and strict rules about persistence and transmission:

| Data Type | Location | Lifetime | Persisted to Disk? | Sent to Network? |
|---|---|---|---|---|
| Raw audio (PCM) | Ring buffer (RAM) | 30 s max | Never | Never |
| Mel spectrogram | Processing buffer (RAM) | ~200 ms | Never | Never |
| Transcript text | Accumulator (RAM) | Until call ends | Never | Never |
| Feature vectors | Processing buffer (RAM) | ~10 ms | Never | Never |
| Fraud scores | Score buffer (RAM) | Until call ends | Never | Never |
| User action (block/dismiss) | Training label buffer | Until next federated round | Encrypted on disk | Never (label only) |
| Model gradient delta | Computed at round time | Minutes | Never | Yes (DP-noised) |

After a call ends, all in-memory buffers (audio, transcripts, features, scores) are zeroed. The only artifact that survives is the user's block/dismiss action, stored as an encrypted training label paired with the feature vector (never the transcript).

---

## 6. Edge-to-Hub Data Boundary

This is the most critical architectural boundary. It defines what the hub can and cannot know.

### What Stays on Device (Never Transmitted)

- Raw call audio
- Transcript text
- Phone numbers (caller or callee)
- Per-sentence fraud scores
- Feature vectors
- Block/dismiss decisions
- Any data that identifies a specific call or person

### What the Hub Receives (Per Device, Per Federated Round)

| Data | Size | Description |
|---|---|---|
| Model gradient delta | ~20 KB | Weight differences from local fine-tuning, DP-noised |
| Feature importance array | ~2 KB | Per-feature importance scores, DP-noised |
| Sample count | 4 bytes | Number of calls used for local training ("I trained on N calls") |
| Model version | 4 bytes | Which global model version the device was running |

**Total upload per device per round: ~22 KB.**

### Why This Is Safe

The gradient delta is a vector of weight differences averaged over all local training samples. With differential privacy noise at epsilon=0.3, the gradient for any single call is buried under Gaussian noise with standard deviation proportional to `1 / (n_samples * epsilon)`. For a device with 50+ training samples, gradient inversion attacks are provably infeasible -- the signal-to-noise ratio for any individual sample is far below the reconstruction threshold established in the literature.

---

## 7. Hub Server Architecture

### Federated Round Lifecycle

A single federated round proceeds through five stages:

**1. Collect Updates**
Wait until K devices (minimum 5) submit gradient deltas via TLS-authenticated endpoints. Devices authenticate with rotating anonymous device IDs. The hub imposes a deadline; if fewer than 5 devices report before the deadline, the round is skipped.

**2. Secure Aggregation (FedAvg)**
```
global_delta = sum(n_i * G_i) / sum(n_i)
```
Each device's gradient `G_i` is weighted by its sample count `n_i`. A device that trained on 200 calls has proportionally more influence than one that trained on 10. This is standard FedAvg.

**3. Model Validation**
The aggregated delta is applied to the current global model and tested against a held-out validation set maintained by the hub. If the F1 score drops by more than 2% compared to the previous model, the update is rejected. This provides Byzantine fault tolerance: a poisoned gradient from a compromised device will degrade validation metrics and be caught.

**4. Model Distribution**
The validated model is versioned, compressed, and signed with an Ed25519 key. It is distributed as a delta patch (~300 KB) rather than the full model. Edge devices verify the signature before applying the patch, preventing man-in-the-middle model tampering.

**5. Metrics Logging**
Only aggregated statistics are logged: total scams blocked across the fleet, accuracy trend over rounds, number of contributing devices. Zero per-user or per-device metrics are retained.

### Hub Server Specifications

| Property | Value |
|---|---|
| Compute per round | < 100 ms for FedAvg |
| RAM | ~50 MB |
| Storage | ~500 MB (model version history + validation set) |
| Network per device | ~20 KB in, ~300 KB out |
| Hosting requirement | Single-core VM, no GPU |
| Round frequency | Daily or weekly (configurable) |
| Min devices per round | 5 |
| Framework | Flower (`flwr`) for production, custom Python for demo |

### Key Files

- `hub/server.py` -- FastAPI endpoints for gradient submission and model distribution
- `hub/aggregator.py` -- FedAvg with outlier detection and rejection
- `hub/validator.py` -- Model validation against held-out test set
- `hub/model_store.py` -- Model versioning, Ed25519 signing, delta patch generation
- `hub/round_manager.py` -- Round orchestration and deadline management
- `hub/schemas.py` -- Pydantic request/response contracts

---

## 8. On-Device Local Training Loop

### Implicit Labeling from User Behavior

SentinelEdge does not ask users to explicitly label calls. Instead, labels are inferred from post-call behavior:

| User Behavior | Inferred Label | Confidence | Used for Training? |
|---|---|---|---|
| Blocks caller after alert | SCAM | High | Yes |
| Reports as spam after call | SCAM | Medium | Yes |
| Dismisses alert, call continues 30 s+ | UNCERTAIN | Low | Discarded |
| No alert, call lasted 5+ min | LEGIT | High | Yes |
| No alert, call < 10 s | UNCERTAIN | Low | Discarded |

Low-confidence labels are discarded to avoid poisoning the local training set. Only feature vectors (never transcript text) are stored in the local training buffer, which is capped at 500 samples and encrypted at rest with Android Keystore-backed AES-256.

### Local Fine-Tuning

When a federated round is triggered, the device fine-tunes the current global model on its local buffer. The training produces a gradient delta: the difference between the fine-tuned weights and the original global weights.

### DP Noise Injection

Before uploading, the gradient delta is noised:

```
sensitivity = 1.0 / n_local_samples
sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
noised_delta = gradient_delta + Normal(0, sigma)
```

With epsilon=0.3 and delta=1e-5, individual gradients are heavily noised. However, when the hub averages 50+ devices' noised gradients, the noise cancels out (central limit theorem) while the true signal accumulates. This is the fundamental trade-off of federated DP: individual privacy is strong, but collective learning still converges.

### Key Files

- `federated/local_trainer.py` -- On-device fine-tuning loop
- `federated/dp_injector.py` -- DP noise injection orchestration
- `sentinel_edge/privacy/dp_noise.py` -- Core DP math (sensitivity, sigma computation)
- `sentinel_edge/privacy/gradient_delta.py` -- Delta computation between local and global weights

---

## 9. Privacy Guarantees Summary

| Property | Guarantee | Mechanism |
|---|---|---|
| Data locality | 100% on-device | No network calls during inference |
| Differential privacy | epsilon=0.3, delta=1e-5 | Gaussian noise injected into gradient deltas |
| Secure aggregation | Hub sees only the weighted sum | Masking protocol between devices |
| Data minimization | Only ~22 KB transmitted per round | No raw features, text, audio, or metadata |
| Gradient inversion resistance | Provably infeasible at epsilon=0.3 | DP noise floor exceeds reconstruction threshold |
| Byzantine fault tolerance | Poisoned updates detected and rejected | F1 validation gate + statistical outlier rejection |

The system is designed so that even a fully compromised hub server learns nothing about individual calls. The hub receives only DP-noised gradient deltas, which are provably insufficient to reconstruct any input.

---

## 10. API Contracts

### Edge Inference API

The primary interface for on-device fraud detection:

```python
class SentinelEngine:
    def analyze_sms(text: str) -> DetectionResult
    def analyze_url(url: str) -> DetectionResult
    def analyze_call(transcript: str) -> DetectionResult
    def analyze_auto(text: str) -> DetectionResult  # auto-detect channel
```

`analyze_auto` inspects the input to determine the channel (SMS, URL, or call transcript) and routes to the appropriate analyzer. All methods return the same `DetectionResult` schema.

### DetectionResult Schema

```python
@dataclass
class DetectionResult:
    channel: str        # 'sms' | 'url' | 'call'
    is_fraud: bool      # binary verdict
    confidence: float   # 0.0 to 1.0
    risk_level: str     # 'critical' | 'high' | 'medium' | 'low' | 'safe'
    reasons: list[str]  # human-readable explanations, e.g. ["urgency language detected"]
    inference_ms: float # end-to-end latency for this classification
```

### Federated Update Payload (Edge -> Hub)

```python
@dataclass
class FederatedUpdate:
    device_id: str              # anonymous identifier, rotated monthly
    model_version: int          # global model version the device trained against
    gradient_delta: list[float] # DP-noised weight differences
    feature_importances: list[float]  # DP-noised per-feature importance
    n_samples: int              # number of local training samples used
    dp_epsilon: float           # privacy budget spent this round
    dp_sigma: float             # noise standard deviation applied
```

### Hub Response (Hub -> Edge)

```python
@dataclass
class ModelUpdate:
    model_version: int          # new global model version number
    model_delta: bytes          # compressed weight delta (ZSTD)
    signature: bytes            # Ed25519 signature over model_delta
    n_contributing_devices: int # how many devices contributed to this round
    round_accuracy: float       # F1 score on validation set
```

Edge devices verify `signature` against the hub's known public key before applying `model_delta`. If verification fails, the update is rejected and the device continues on its current model version.
