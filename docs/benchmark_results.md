# SentinelEdge Mobile Benchmark Results

Generated: 2026-03-21 15:52:27
Iterations: 100

## 1. Model Size Analysis

| Model | Size | Status |
|-------|------|--------|
| XGBoost (call_fraud_xgb.json) | 210.63 KB | Found |
| TF-IDF (tfidf_call_vectorizer.pkl) | 18.79 KB | Found |
| MLP (call_fraud_mlp.npz) | 240.00 KB | Estimated |
| MiniLM ONNX (all-MiniLM-L6-v2) | 22.00 MB | Estimated |
| Whisper Tiny ONNX | 150.00 MB | Estimated |

**Total estimated on-device footprint:** ~172.5 MB
**Budget:** 300 MB
**Status:** WITHIN BUDGET

## 2. Inference Latency

| Component | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|-----------|-----------|----------|----------|----------|
| Handcrafted features (18 dims) | 0.055 | 0.048 | 0.090 | 0.353 | (measured)
| TF-IDF vectorize (500 dims) | 0.081 | 0.077 | 0.088 | 0.330 | (measured)
| XGBoost classify | 0.169 | 0.160 | 0.188 | 0.971 | (measured)
| MLP classify (402->128->64->1) | 0.011 | 0.011 | 0.011 | 0.017 | (measured)
| MiniLM embed (384 dims) | 9.283 | 8.267 | 21.168 | 24.873 | (measured)

## 3. Memory Profiling

| Component | Memory |
|-----------|--------|
| Handcrafted features | 14.94 KB |
| TF-IDF vectorizer | 91.73 KB |
| XGBoost model | 9.73 KB |
| MLP model | 470.59 KB |
| MiniLM sentence-transformer | 3.92 MB |
| ONNX Runtime Mobile (base) | 15.00 MB |

## 4. Android Performance Estimation

Scaling factors (Snapdragon 6 Gen 1 mid-range vs. laptop CPU):
- CPU workloads: 3.0x - 5.0x slower
- NPU-accelerated: 1.5x - 2.0x slower

| Component | Desktop (ms) | Android CPU (ms) | Android NPU (ms) |
|-----------|-------------|------------------|-------------------|
| Handcrafted features (18 dims) | 0.055 | 0.2 - 0.3 | N/A |
| TF-IDF vectorize (500 dims) | 0.081 | 0.2 - 0.4 | N/A |
| XGBoost classify | 0.169 | 0.5 - 0.8 | N/A |
| MLP classify (402->128->64->1) | 0.011 | 0.0 - 0.1 | N/A |
| MiniLM embed (384 dims) | 9.283 | 27.8 - 46.4 | 13.9 - 18.6 |

### End-to-End Pipeline Estimate

| Pipeline Stage | Desktop (ms) | Android (ms) |
|---------------|-------------|-------------|
| Classification pipeline | 9.6 | 29 - 48 |
| + Whisper Tiny (10s audio) | 2000 | 3000 - 7000 |
| **Grand Total** | **2010** | **3029 - 7048** |

## 5. ONNX Export Verification

- Exported: No
- Verified: No

## 6. Reproduction Instructions

### Desktop Benchmark
```bash
cd SentinelEdge
python3 training/benchmark_mobile.py
```

### Android Device Benchmark

To reproduce on a real Android device:

1. **Export models to ONNX:**
   ```bash
   python3 training/export_onnx.py
   ```

2. **Build the Android benchmark app:**
   - Open `android/` project in Android Studio
   - Copy ONNX models to `app/src/main/assets/`
   - Run the benchmark Activity on device

3. **Use ONNX Runtime Mobile:**
   ```kotlin
   // In your Android benchmark code:
   val session = OrtEnvironment.getEnvironment()
   val options = OrtSession.SessionOptions()
   // Enable NNAPI for NPU acceleration:
   options.addNnapi()
   val ortSession = session.createSession(modelPath, options)
   ```

4. **Measure latency per component:**
   - Time each ONNX session.run() call
   - Run 100+ iterations, discard first 10 (warmup)
   - Report mean, P50, P95, P99

5. **Compare with/without NNAPI:**
   - CPU-only: remove `addNnapi()` call
   - NNAPI (NPU): include `addNnapi()` call
   - Some models may not benefit from NPU

### Key Android Considerations

- **Thermal throttling:** Run benchmarks with device at room temp
- **Background processes:** Use airplane mode, close other apps
- **Battery state:** Benchmark at >50% battery (avoid power-saving mode)
- **Memory pressure:** Monitor `adb shell dumpsys meminfo` during benchmark
- **CPU governor:** Lock to performance mode if rooted:
  ```bash
  adb shell 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
  ```

## Methodology Notes

- Desktop benchmarks run 100 iterations with 10-iteration warmup
- Android estimates use published CPU scaling factors:
  - Mid-range (Snapdragon 6 Gen 1): 3.0x - 5.0x vs. laptop
  - NPU acceleration: 1.5x - 2.0x vs. laptop
- Whisper Tiny estimates from published ONNX Runtime Mobile benchmarks
- MiniLM ONNX size estimate (~22 MB) from HuggingFace model card
- Memory profiling uses Python tracemalloc (may underestimate native allocations)
