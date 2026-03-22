# SentinelEdge

**Federated Edge AI for Real-Time Phone Call Fraud Detection**

SentinelEdge is a privacy-preserving system that detects scam calls in real-time using on-device machine learning. No call audio, transcripts, or personal data ever leaves the user's device.

---

## Key Properties

| Property | Value |
|---|---|
| Model architecture | Whisper Tiny (transcription) + XGBoost (classification) |
| On-device model size | ~155 MB total (150 MB Whisper + 5 MB classifier) |
| End-to-end latency | 3-7 seconds (mic to alert) |
| Privacy guarantee | Differential privacy, epsilon = 0.3 |
| Federated protocol | FedAvg with secure aggregation |
| Target platform | Android 7+ (CallScreeningService API) |

---

## Architecture

```
Phone Call -> Audio Capture -> Ring Buffer -> Whisper STT -> Sentence Splitter
    -> Feature Extraction (18 handcrafted + 500 TF-IDF = 518 dims)
    -> XGBoost Classifier -> EMA Smoothing -> Alert Decision

Federated Learning:
    Edge Device -> Local Training -> DP Noise -> Hub Server -> FedAvg -> New Global Model
    (only ~20KB gradient delta crosses the network, never any user data)
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/SentinelEdge.git
cd SentinelEdge
pip install -r requirements.txt

# Generate training data and train model
python3 training/generate_synthetic_data.py
python3 training/prepare_datasets.py
python3 training/fit_tfidf.py
python3 training/train_call_classifier.py

# Run the demo
python3 demo/backend/main.py &          # Backend on :8000
cd demo/frontend && npm install && npm run dev  # Frontend on :5173

# Run federated simulation
python3 -m federated.simulate --devices 5 --rounds 5

# Run tests
python3 -m pytest tests/ -v
```

---

## Project Structure

```
SentinelEdge/
├── sentinel_edge/          # Core ML package
│   ├── features/           # Feature extraction (18 handcrafted + TF-IDF)
│   ├── classifier/         # XGBoost inference, EMA scoring, alert engine
│   ├── audio/              # Ring buffer, Whisper transcription, windowing
│   └── privacy/            # Differential privacy noise injection
├── training/               # Model training pipeline
├── hub/                    # Federated aggregation server (FastAPI)
├── federated/              # Federated learning simulation
├── demo/
│   ├── backend/            # WebSocket server for real-time demo
│   └── frontend/           # React phone simulator UI
├── tests/                  # 61 unit tests
└── docs/                   # Architecture documentation
```

---

## How It Works

SentinelEdge runs a 7-stage pipeline entirely on-device:

1. **Audio Capture** -- Raw 16kHz PCM audio is read from the microphone into a circular ring buffer. No audio is written to disk at any point.

2. **Windowing** -- A 5-second sliding window with a 1-second hop extracts overlapping audio chunks for continuous analysis.

3. **Whisper Transcription** -- Each audio window is fed through Whisper Tiny (running via ONNX Runtime) to produce a text transcript in real time.

4. **Sentence Segmentation** -- The raw transcript stream is split into coherent sentence boundaries for downstream feature extraction.

5. **Feature Extraction** -- Each sentence is transformed into a 518-dimensional vector: 18 handcrafted linguistic features (urgency cues, financial keywords, pressure tactics, etc.) concatenated with 500 TF-IDF dimensions.

6. **XGBoost Classification** -- The feature vector is scored by a lightweight XGBoost model that outputs a fraud probability between 0 and 1.

7. **Alert Decision** -- An exponential moving average (EMA) smooths per-sentence scores across the call. When the EMA exceeds 0.75, a red alert is triggered to warn the user.

---

## Privacy Guarantees

| Data | Stays on device | Crosses the network |
|---|---|---|
| Raw audio | Never sent, never saved to disk | -- |
| Transcript text | Never sent, held in RAM only | -- |
| Feature vectors | Never sent | -- |
| Fraud scores | Never sent | -- |
| Model gradient delta (~20KB) | -- | Sent with DP noise (epsilon = 0.3) |

The only data that ever leaves the device is a small gradient update (~20KB) protected by calibrated differential privacy noise. An attacker intercepting this payload cannot reconstruct any individual call, transcript, or feature vector.

---

## Detection Channels

| Channel | Model | Data Source | Status |
|---|---|---|---|
| Phone call scam | Whisper + XGBoost | Synthetic transcripts | Working |
| SMS phishing | TF-IDF + XGBoost | Kaggle SMS Spam | Planned |
| Phishing URLs | URL features + XGBoost | PhishTank | Planned |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| ML / Inference | Python 3.10+, XGBoost, scikit-learn, ONNX Runtime, Whisper |
| Federated Server | FastAPI, PyNaCl (secure aggregation) |
| Demo Frontend | React 18, TypeScript, Tailwind CSS, Recharts |
| Testing | pytest (61 unit tests) |

---

## Current Status

SentinelEdge is a **prototype and proof-of-concept demo**. It demonstrates the full pipeline from audio capture through federated learning, but it has not been deployed to production or tested on real phone networks.

The Android UI prototype now lives under `android/` and is intentionally separate from the web demo in `demo/frontend`.

Key limitations:

- The training data is synthetic (LLM-generated scam transcripts), not real-world call recordings.
- The federated learning loop has been validated in simulation only, not across physical devices.
- The Android integration targets the CallScreeningService API but has not been packaged as an APK.
- Model accuracy metrics reflect synthetic data performance and may not generalize to real scam calls.

See [ASSESSMENT.md](docs/ASSESSMENT.md) for the full honest technical breakdown, including what works, what does not, and what would be needed to move toward production.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, feature requests, and pull requests.

---

## License

This project is licensed under the [MIT License](LICENSE).
