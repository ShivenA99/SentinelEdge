# SentinelEdge - AI Coding Guide

## What This Project Is
Federated edge AI for real-time phone call fraud detection. On-device ML pipeline: Whisper STT -> NLP features -> XGBoost classifier -> alert. Federated learning with differential privacy (epsilon=0.3) so no user data ever leaves the device.

## Key Commands
```bash
# Run demo (backend + frontend)
python3 demo/backend/main.py              # FastAPI on :8000
cd demo/frontend && npm run dev            # React on :5173

# Run hub server
python3 -m uvicorn hub.server:app --port 8080

# Run federated simulation
python3 -m federated.simulate --devices 5 --rounds 5

# Train models (from project root)
python3 training/generate_synthetic_data.py
python3 training/prepare_datasets.py
python3 training/fit_tfidf.py
python3 training/train_call_classifier.py

# Run tests
python3 -m pytest tests/ -v
```

## Directory Structure
```
sentinel_edge/           Core ML package (features, classifier, audio, privacy)
  features/              18 handcrafted + 500 TF-IDF features -> 518-dim vector
  classifier/            XGBoost inference, EMA scoring, alert engine
  audio/                 Ring buffer, Whisper STT, mel spectrogram, sentence splitter
  privacy/               DP noise injection, gradient delta computation
  engine.py              Top-level SentinelEngine API

training/                Data generation + model training scripts
hub/                     FastAPI federated aggregation server
federated/               FL simulation (N devices, M rounds)
demo/backend/            WebSocket server streaming real-time detection
demo/frontend/           React + Tailwind phone simulator UI
tests/                   61 unit tests (pytest)
```

## Conventions
- Python 3.10+, type hints on all functions
- numpy arrays for numerical data (not lists)
- Pydantic models for API schemas (hub/)
- React 18 + TypeScript strict mode (frontend)
- Tailwind CSS for styling (no CSS-in-JS)

## Important Patterns
- **Feature pipeline**: `extract_handcrafted_features(text)` returns dict of 18 features. `FeaturePipeline.extract(text)` returns 518-dim numpy array.
- **EMA scoring**: `ScoreAccumulator(alpha=0.3)` smooths per-sentence scores. First update bootstraps to the value.
- **Alert thresholds**: >0.75 = CRITICAL (red overlay), 0.5-0.75 = HIGH (amber), 0.3-0.5 = MEDIUM, <0.3 = SAFE
- **DP noise**: `sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon`. Sensitivity = 1/n_samples.

## Do NOT
- Commit `model_store/` directory (contains Ed25519 private keys)
- Use `np.trapz()` (removed in NumPy 2.0, use `np.trapezoid()`)
- Commit `.env` files or API keys
- Skip type hints on new Python code
- Add dependencies without updating both `requirements.txt` and `pyproject.toml`

## Before Submitting a PR
1. All existing tests pass: `python3 -m pytest tests/ -v`
2. Add tests for new functionality
3. No secrets in committed files
4. Type hints on all new functions
