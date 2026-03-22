"""Benchmark MiniLM + MLP inference for mobile deployment.

Measures latency, memory, and model sizes on the current machine,
then estimates Android performance based on published benchmarks.

Run: python3 training/benchmark_mobile.py
"""

from __future__ import annotations

import datetime
import gc
import os
import statistics
import sys
import time
import tracemalloc

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ITERATIONS = 100
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(_PROJECT_ROOT, "docs")

# Android scaling factors (based on published benchmarks)
# Snapdragon 6 Gen 1 mid-range vs. typical laptop (M-series / x86-64)
ANDROID_CPU_SLOW = 3.0   # conservative: 3x slower than laptop
ANDROID_CPU_FAST = 5.0   # worst-case: 5x slower than laptop
ANDROID_NPU_SLOW = 1.5   # NPU-accelerated: 1.5x slower
ANDROID_NPU_FAST = 2.0   # NPU-accelerated: 2x slower

# Whisper Tiny estimates for Android (published benchmarks)
WHISPER_TINY_ONNX_SIZE_MB = 150.0
WHISPER_ANDROID_LATENCY_LOW_MS = 3000.0
WHISPER_ANDROID_LATENCY_HIGH_MS = 7000.0
WHISPER_DESKTOP_LATENCY_MS = 2000.0

# MiniLM ONNX model size estimate (from HuggingFace model card)
MINILM_ONNX_SIZE_MB = 22.0

# Budget constraints
TOTAL_SIZE_BUDGET_MB = 300.0

# Sample texts for benchmarking (representative of fraud detection inputs)
SAMPLE_TEXTS = [
    "Your account has been compromised. Call us immediately at 1-800-555-0123 to verify your identity.",
    "Hi, this is a reminder about your dental appointment tomorrow at 3pm.",
    "URGENT: Your social security number has been suspended. Press 1 to speak with an agent now!",
    "Hey, just wanted to check if you're coming to dinner tonight.",
    "Congratulations! You've been selected as the winner of a $10,000 prize. Click here to claim: bit.ly/abc123",
    "The quarterly report is due by Friday. Please submit it to the shared drive.",
    "This is the IRS. You owe back taxes of $5,000. Pay immediately or face arrest.",
    "Can you pick up milk on your way home? Thanks!",
    "Your Amazon account will be locked unless you verify your payment information right now.",
    "The meeting has been rescheduled to 2pm. See you there.",
]


# ======================================================================
# Utility functions
# ======================================================================

def format_size(size_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def compute_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Compute mean, p50, p95, p99 from a list of latencies in ms."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    return {
        "mean": statistics.mean(sorted_lat),
        "p50": sorted_lat[int(n * 0.50)],
        "p95": sorted_lat[min(int(n * 0.95), n - 1)],
        "p99": sorted_lat[min(int(n * 0.99), n - 1)],
    }


def estimate_android(latency_ms: float) -> tuple[float, float]:
    """Estimate Android CPU latency range from desktop measurement."""
    return (latency_ms * ANDROID_CPU_SLOW, latency_ms * ANDROID_CPU_FAST)


def measure_memory_delta(func, *args, **kwargs):
    """Measure peak memory delta during function execution.

    Returns (result, delta_bytes).
    """
    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    result = func(*args, **kwargs)
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compute delta
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    delta = sum(s.size_diff for s in stats if s.size_diff > 0)
    return result, delta


# ======================================================================
# 1. Model Size Analysis
# ======================================================================

def analyze_model_sizes() -> dict[str, dict]:
    """Measure on-disk sizes of all model files."""
    print("\n" + "=" * 70)
    print("1. MODEL SIZE ANALYSIS")
    print("=" * 70)

    model_files = {
        "XGBoost (call_fraud_xgb.json)": os.path.join(MODELS_DIR, "call_fraud_xgb.json"),
        "TF-IDF (tfidf_call_vectorizer.pkl)": os.path.join(MODELS_DIR, "tfidf_call_vectorizer.pkl"),
        "MLP (call_fraud_mlp.npz)": os.path.join(MODELS_DIR, "call_fraud_mlp.npz"),
    }

    results = {}
    total_measured = 0

    for name, path in model_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            results[name] = {"size_bytes": size, "exists": True}
            total_measured += size
            print(f"  {name:<45} {format_size(size):>12}  [FOUND]")
        else:
            # Provide estimates for missing models
            if "mlp" in name.lower():
                est_size = 240 * 1024  # ~240 KB for MLP with 59,969 params
                results[name] = {"size_bytes": est_size, "exists": False}
                print(f"  {name:<45} ~{format_size(est_size):>11}  [ESTIMATED]")
            else:
                results[name] = {"size_bytes": 0, "exists": False}
                print(f"  {name:<45} {'N/A':>12}  [NOT FOUND]")

    # MiniLM ONNX estimate
    minilm_bytes = int(MINILM_ONNX_SIZE_MB * 1024 * 1024)
    results["MiniLM ONNX (all-MiniLM-L6-v2)"] = {
        "size_bytes": minilm_bytes,
        "exists": False,
    }
    print(f"  {'MiniLM ONNX (all-MiniLM-L6-v2)':<45} ~{format_size(minilm_bytes):>11}  [ESTIMATED]")

    # Whisper Tiny estimate
    whisper_bytes = int(WHISPER_TINY_ONNX_SIZE_MB * 1024 * 1024)
    results["Whisper Tiny ONNX"] = {
        "size_bytes": whisper_bytes,
        "exists": False,
    }
    print(f"  {'Whisper Tiny ONNX':<45} ~{format_size(whisper_bytes):>11}  [ESTIMATED]")

    # Total footprint
    total_all = sum(r["size_bytes"] for r in results.values())
    print(f"\n  {'Total on-device footprint (estimated)':<45} ~{format_size(total_all):>11}")
    print(f"  {'Budget':<45} {format_size(int(TOTAL_SIZE_BUDGET_MB * 1024 * 1024)):>12}")

    if total_all <= TOTAL_SIZE_BUDGET_MB * 1024 * 1024:
        print(f"  STATUS: WITHIN BUDGET ({total_all / (1024*1024):.1f} MB < {TOTAL_SIZE_BUDGET_MB:.0f} MB)")
    else:
        print(f"  STATUS: OVER BUDGET ({total_all / (1024*1024):.1f} MB > {TOTAL_SIZE_BUDGET_MB:.0f} MB)")

    return results


# ======================================================================
# 2. Inference Latency Benchmarks
# ======================================================================

def benchmark_handcrafted_features() -> dict:
    """Benchmark handcrafted feature extraction (18 features)."""
    from sentinel_edge.features.handcrafted import extract_handcrafted_features

    latencies = []
    for i in range(N_ITERATIONS):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        _ = extract_handcrafted_features(text)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

    stats = compute_stats(latencies)
    return {"component": "Handcrafted features (18 dims)", "latencies_ms": latencies, "stats": stats}


def benchmark_tfidf() -> dict:
    """Benchmark TF-IDF vectorization (500 features)."""
    from sentinel_edge.features.tfidf import TfidfFeatureExtractor

    tfidf_path = os.path.join(MODELS_DIR, "tfidf_call_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        print("  WARNING: TF-IDF vectorizer not found, skipping benchmark.")
        return {"component": "TF-IDF vectorize (500 dims)", "latencies_ms": [], "stats": compute_stats([])}

    tfidf = TfidfFeatureExtractor.load(tfidf_path)

    latencies = []
    for i in range(N_ITERATIONS):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        _ = tfidf.transform(text)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    stats = compute_stats(latencies)
    return {"component": "TF-IDF vectorize (500 dims)", "latencies_ms": latencies, "stats": stats}


def benchmark_xgboost() -> dict:
    """Benchmark XGBoost inference."""
    xgb_path = os.path.join(MODELS_DIR, "call_fraud_xgb.json")
    if not os.path.exists(xgb_path):
        print("  WARNING: XGBoost model not found, skipping benchmark.")
        return {"component": "XGBoost classify", "latencies_ms": [], "stats": compute_stats([])}

    try:
        import xgboost as xgb
    except ImportError:
        print("  WARNING: xgboost not installed, skipping benchmark.")
        return {"component": "XGBoost classify", "latencies_ms": [], "stats": compute_stats([])}

    model = xgb.XGBClassifier()
    model.load_model(xgb_path)

    # Prepare a dummy feature vector (518 dims: 18 handcrafted + 500 TF-IDF)
    rng = np.random.default_rng(42)
    dummy_features = rng.standard_normal((1, 518)).astype(np.float32)

    # Warmup
    for _ in range(10):
        model.predict(dummy_features)

    latencies = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        _ = model.predict(dummy_features)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    stats = compute_stats(latencies)
    return {"component": "XGBoost classify", "latencies_ms": latencies, "stats": stats}


def benchmark_mlp() -> dict:
    """Benchmark MLP inference (numpy forward pass)."""
    from sentinel_edge.classifier.mlp_classifier import MLPClassifier

    mlp_path = os.path.join(MODELS_DIR, "call_fraud_mlp.npz")

    if os.path.exists(mlp_path):
        model = MLPClassifier.load(mlp_path)
        print(f"  Loaded MLP from {mlp_path}: {model}")
    else:
        # Use a freshly initialized model (random weights) for latency measurement
        model = MLPClassifier(input_dim=402)
        print(f"  MLP model file not found; using initialized model: {model}")

    # Prepare a dummy feature vector (402 dims: 18 handcrafted + 384 embedding)
    # Use batch format (1, 402) which matches real deployment pattern
    rng = np.random.default_rng(42)
    dummy_features = rng.standard_normal((1, 402)).astype(np.float32)

    # Warmup
    for _ in range(10):
        model.forward_batch(dummy_features)

    latencies = []
    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        _ = model.forward_batch(dummy_features)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    stats = compute_stats(latencies)
    return {"component": "MLP classify (402->128->64->1)", "latencies_ms": latencies, "stats": stats}


def benchmark_embedding() -> dict:
    """Benchmark MiniLM embedding extraction."""
    try:
        from sentence_transformers import SentenceTransformer

        print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Warmup
        for _ in range(5):
            st_model.encode("warmup text", convert_to_numpy=True)

        latencies = []
        for i in range(N_ITERATIONS):
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            t0 = time.perf_counter()
            _ = st_model.encode(text, convert_to_numpy=True)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        stats = compute_stats(latencies)
        return {
            "component": "MiniLM embed (384 dims)",
            "latencies_ms": latencies,
            "stats": stats,
            "available": True,
        }

    except ImportError:
        print("  sentence-transformers not available; using estimate.")
        # Estimate based on published benchmarks: ~30ms on laptop CPU
        est_ms = 30.0
        return {
            "component": "MiniLM embed (384 dims)",
            "latencies_ms": [],
            "stats": {"mean": est_ms, "p50": est_ms, "p95": est_ms * 1.2, "p99": est_ms * 1.5},
            "available": False,
        }


def run_latency_benchmarks() -> list[dict]:
    """Run all latency benchmarks and return results."""
    print("\n" + "=" * 70)
    print(f"2. INFERENCE LATENCY BENCHMARKS ({N_ITERATIONS} iterations)")
    print("=" * 70)

    results = []

    print("\n  [A] Handcrafted feature extraction...")
    r = benchmark_handcrafted_features()
    results.append(r)
    print(f"      mean={r['stats']['mean']:.3f}ms  p50={r['stats']['p50']:.3f}ms  "
          f"p95={r['stats']['p95']:.3f}ms  p99={r['stats']['p99']:.3f}ms")

    print("\n  [B] TF-IDF vectorization...")
    r = benchmark_tfidf()
    results.append(r)
    if r["latencies_ms"]:
        print(f"      mean={r['stats']['mean']:.3f}ms  p50={r['stats']['p50']:.3f}ms  "
              f"p95={r['stats']['p95']:.3f}ms  p99={r['stats']['p99']:.3f}ms")

    print("\n  [C] XGBoost inference...")
    r = benchmark_xgboost()
    results.append(r)
    if r["latencies_ms"]:
        print(f"      mean={r['stats']['mean']:.3f}ms  p50={r['stats']['p50']:.3f}ms  "
              f"p95={r['stats']['p95']:.3f}ms  p99={r['stats']['p99']:.3f}ms")

    print("\n  [D] MLP inference...")
    r = benchmark_mlp()
    results.append(r)
    if r["latencies_ms"]:
        print(f"      mean={r['stats']['mean']:.3f}ms  p50={r['stats']['p50']:.3f}ms  "
              f"p95={r['stats']['p95']:.3f}ms  p99={r['stats']['p99']:.3f}ms")

    print("\n  [E] MiniLM embedding extraction...")
    r = benchmark_embedding()
    results.append(r)
    available_tag = "(measured)" if r.get("available", False) else "(estimated)"
    print(f"      mean={r['stats']['mean']:.3f}ms  p50={r['stats']['p50']:.3f}ms  "
          f"p95={r['stats']['p95']:.3f}ms  p99={r['stats']['p99']:.3f}ms  {available_tag}")

    return results


# ======================================================================
# 3. Memory Profiling
# ======================================================================

def profile_memory() -> dict[str, int]:
    """Measure memory usage for loading each model."""
    print("\n" + "=" * 70)
    print("3. MEMORY PROFILING")
    print("=" * 70)

    mem_results = {}

    # --- Handcrafted features (function, no model to load) ---
    from sentinel_edge.features.handcrafted import extract_handcrafted_features

    def _run_handcrafted():
        return extract_handcrafted_features("Test sentence for memory profiling.")

    _, delta = measure_memory_delta(_run_handcrafted)
    mem_results["Handcrafted features"] = delta
    print(f"  {'Handcrafted features':<40} {format_size(delta):>12}")

    # --- TF-IDF vectorizer ---
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_call_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        from sentinel_edge.features.tfidf import TfidfFeatureExtractor

        def _load_tfidf():
            return TfidfFeatureExtractor.load(tfidf_path)

        _, delta = measure_memory_delta(_load_tfidf)
        mem_results["TF-IDF vectorizer"] = delta
        print(f"  {'TF-IDF vectorizer':<40} {format_size(delta):>12}")
    else:
        print(f"  {'TF-IDF vectorizer':<40} {'N/A (not found)':>12}")

    # --- XGBoost model ---
    xgb_path = os.path.join(MODELS_DIR, "call_fraud_xgb.json")
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb

            def _load_xgb():
                m = xgb.XGBClassifier()
                m.load_model(xgb_path)
                return m

            _, delta = measure_memory_delta(_load_xgb)
            mem_results["XGBoost model"] = delta
            print(f"  {'XGBoost model':<40} {format_size(delta):>12}")
        except ImportError:
            print(f"  {'XGBoost model':<40} {'N/A (xgboost not installed)':>12}")
    else:
        print(f"  {'XGBoost model':<40} {'N/A (not found)':>12}")

    # --- MLP model ---
    from sentinel_edge.classifier.mlp_classifier import MLPClassifier

    def _load_mlp():
        mlp_path = os.path.join(MODELS_DIR, "call_fraud_mlp.npz")
        if os.path.exists(mlp_path):
            return MLPClassifier.load(mlp_path)
        return MLPClassifier(input_dim=402)

    _, delta = measure_memory_delta(_load_mlp)
    mem_results["MLP model"] = delta
    print(f"  {'MLP model':<40} {format_size(delta):>12}")

    # --- MiniLM estimate ---
    try:
        from sentence_transformers import SentenceTransformer

        def _load_minilm():
            return SentenceTransformer("all-MiniLM-L6-v2")

        _, delta = measure_memory_delta(_load_minilm)
        mem_results["MiniLM sentence-transformer"] = delta
        print(f"  {'MiniLM sentence-transformer':<40} {format_size(delta):>12}  [measured]")
    except ImportError:
        est_mem = 90 * 1024 * 1024  # ~90 MB estimated runtime memory
        mem_results["MiniLM sentence-transformer"] = est_mem
        print(f"  {'MiniLM sentence-transformer':<40} ~{format_size(est_mem):>11}  [estimated]")

    # --- ONNX Runtime Mobile overhead estimate ---
    ort_overhead = 15 * 1024 * 1024  # ~15 MB for ONNX Runtime Mobile base
    mem_results["ONNX Runtime Mobile (base)"] = ort_overhead
    print(f"  {'ONNX Runtime Mobile (base)':<40} ~{format_size(ort_overhead):>11}  [estimated]")

    total_mem = sum(mem_results.values())
    print(f"\n  {'Total estimated runtime memory':<40} ~{format_size(total_mem):>11}")

    return mem_results


# ======================================================================
# 4. Android Estimation
# ======================================================================

def estimate_android_performance(latency_results: list[dict]) -> list[dict]:
    """Estimate Android latency based on desktop measurements and scaling factors."""
    print("\n" + "=" * 70)
    print("4. ANDROID PERFORMANCE ESTIMATION")
    print("=" * 70)

    print(f"\n  Scaling factors (Snapdragon 6 Gen 1 mid-range vs. laptop):")
    print(f"    CPU: {ANDROID_CPU_SLOW}x - {ANDROID_CPU_FAST}x slower")
    print(f"    NPU: {ANDROID_NPU_SLOW}x - {ANDROID_NPU_FAST}x slower")

    android_estimates = []
    print(f"\n  {'Component':<35} {'Desktop (ms)':>14} {'Android CPU (ms)':>18} {'Android NPU (ms)':>18}")
    print(f"  {'-'*35} {'-'*14} {'-'*18} {'-'*18}")

    for r in latency_results:
        mean_ms = r["stats"]["mean"]
        cpu_low, cpu_high = estimate_android(mean_ms)

        # NPU estimate (only applicable for MiniLM)
        if "MiniLM" in r["component"]:
            npu_low = mean_ms * ANDROID_NPU_SLOW
            npu_high = mean_ms * ANDROID_NPU_FAST
            npu_str = f"{npu_low:.1f} - {npu_high:.1f}"
        else:
            npu_low = npu_high = None
            npu_str = "N/A (CPU only)"

        android_estimates.append({
            "component": r["component"],
            "desktop_ms": mean_ms,
            "android_cpu_low": cpu_low,
            "android_cpu_high": cpu_high,
            "android_npu_low": npu_low,
            "android_npu_high": npu_high,
        })

        print(f"  {r['component']:<35} {mean_ms:>14.3f} {cpu_low:>7.1f} - {cpu_high:<7.1f}  {npu_str:>18}")

    # Whisper Tiny estimate (not measured, from published data)
    print(f"\n  Whisper Tiny (speech-to-text):")
    print(f"    Desktop estimate:  ~{WHISPER_DESKTOP_LATENCY_MS:.0f} ms (per 10s audio)")
    print(f"    Android estimate:  ~{WHISPER_ANDROID_LATENCY_LOW_MS:.0f} - {WHISPER_ANDROID_LATENCY_HIGH_MS:.0f} ms")
    print(f"    Note: Whisper runs in a streaming fashion; latency is amortized.")

    return android_estimates


# ======================================================================
# 5. ONNX Export Verification
# ======================================================================

def verify_onnx_export() -> dict:
    """Try to export MLP to ONNX and verify outputs match."""
    print("\n" + "=" * 70)
    print("5. ONNX EXPORT VERIFICATION")
    print("=" * 70)

    from sentinel_edge.classifier.mlp_classifier import MLPClassifier

    mlp_path = os.path.join(MODELS_DIR, "call_fraud_mlp.npz")
    if os.path.exists(mlp_path):
        model = MLPClassifier.load(mlp_path)
        print(f"  Loaded MLP from {mlp_path}")
    else:
        model = MLPClassifier(input_dim=402)
        print(f"  Using initialized MLP (model file not found)")

    onnx_result = {"exported": False, "verified": False, "onnx_latency_ms": None}

    # Check for required libraries
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"  SKIP: Required libraries not available ({e})")
        print("  Install with: pip install onnx onnxruntime")
        return onnx_result

    # Create dummy input
    rng = np.random.default_rng(42)
    dummy_input = rng.standard_normal((1, model.input_dim)).astype(np.float32)

    # Get numpy reference output
    ref_output = model.forward_batch(dummy_input)
    print(f"  Numpy reference output: {ref_output[0]:.6f}")

    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "call_fraud_mlp_benchmark.onnx")
    try:
        # Build ONNX model manually from MLP weights
        from onnx import TensorProto, helper, numpy_helper

        # Define inputs and outputs
        X_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, model.input_dim])
        Y_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

        # Create weight initializers
        W1_init = numpy_helper.from_array(model.W1, name="W1")
        b1_init = numpy_helper.from_array(model.b1.reshape(1, -1), name="b1")
        W2_init = numpy_helper.from_array(model.W2, name="W2")
        b2_init = numpy_helper.from_array(model.b2.reshape(1, -1), name="b2")
        W3_init = numpy_helper.from_array(model.W3, name="W3")
        b3_init = numpy_helper.from_array(model.b3.reshape(1, -1), name="b3")

        # Build graph: input -> MatMul+Add -> Relu -> MatMul+Add -> Relu -> MatMul+Add -> Sigmoid
        # Layer 1
        matmul1 = helper.make_node("MatMul", ["input", "W1"], ["z1_pre"])
        add1 = helper.make_node("Add", ["z1_pre", "b1"], ["z1"])
        relu1 = helper.make_node("Relu", ["z1"], ["h1"])

        # Layer 2
        matmul2 = helper.make_node("MatMul", ["h1", "W2"], ["z2_pre"])
        add2 = helper.make_node("Add", ["z2_pre", "b2"], ["z2"])
        relu2 = helper.make_node("Relu", ["z2"], ["h2"])

        # Layer 3 (output)
        matmul3 = helper.make_node("MatMul", ["h2", "W3"], ["z3_pre"])
        add3 = helper.make_node("Add", ["z3_pre", "b3"], ["z3"])
        sigmoid = helper.make_node("Sigmoid", ["z3"], ["output"])

        graph = helper.make_graph(
            nodes=[matmul1, add1, relu1, matmul2, add2, relu2, matmul3, add3, sigmoid],
            name="mlp_fraud_classifier",
            inputs=[X_input],
            outputs=[Y_output],
            initializer=[W1_init, b1_init, W2_init, b2_init, W3_init, b3_init],
        )

        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx_model.ir_version = 7

        # Validate and save
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, onnx_path)
        onnx_size = os.path.getsize(onnx_path)
        print(f"  Exported ONNX model: {format_size(onnx_size)}")
        onnx_result["exported"] = True

        # Verify with ONNX Runtime
        sess = ort.InferenceSession(onnx_path)
        onnx_output = sess.run(None, {"input": dummy_input})
        onnx_prob = float(onnx_output[0][0][0])
        print(f"  ONNX output:         {onnx_prob:.6f}")

        max_diff = abs(ref_output[0] - onnx_prob)
        print(f"  Max difference:      {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  VERIFICATION: PASSED (outputs match within tolerance)")
            onnx_result["verified"] = True
        else:
            print(f"  VERIFICATION: WARNING (difference {max_diff:.2e} exceeds 1e-5)")

        # Benchmark ONNX inference
        # Warmup
        for _ in range(10):
            sess.run(None, {"input": dummy_input})

        onnx_latencies = []
        for _ in range(N_ITERATIONS):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy_input})
            elapsed = (time.perf_counter() - t0) * 1000
            onnx_latencies.append(elapsed)

        onnx_stats = compute_stats(onnx_latencies)
        onnx_result["onnx_latency_ms"] = onnx_stats["mean"]
        print(f"\n  ONNX inference latency:")
        print(f"    mean={onnx_stats['mean']:.3f}ms  p50={onnx_stats['p50']:.3f}ms  "
              f"p95={onnx_stats['p95']:.3f}ms  p99={onnx_stats['p99']:.3f}ms")

        # Compare with numpy
        numpy_latencies = []
        for _ in range(N_ITERATIONS):
            t0 = time.perf_counter()
            model.forward_batch(dummy_input)
            elapsed = (time.perf_counter() - t0) * 1000
            numpy_latencies.append(elapsed)

        numpy_stats = compute_stats(numpy_latencies)
        print(f"  Numpy inference latency:")
        print(f"    mean={numpy_stats['mean']:.3f}ms  p50={numpy_stats['p50']:.3f}ms  "
              f"p95={numpy_stats['p95']:.3f}ms  p99={numpy_stats['p99']:.3f}ms")

        speedup = numpy_stats["mean"] / onnx_stats["mean"] if onnx_stats["mean"] > 0 else 0
        print(f"  ONNX vs Numpy speedup: {speedup:.2f}x")

        # Cleanup benchmark ONNX file
        try:
            os.remove(onnx_path)
        except OSError:
            pass

    except Exception as e:
        print(f"  ONNX export failed: {e}")
        traceback_str = tracemalloc.format_exc() if hasattr(tracemalloc, 'format_exc') else ""
        if traceback_str:
            print(f"  {traceback_str}")

    return onnx_result


# ======================================================================
# 6. Report Generation
# ======================================================================

def generate_report(
    size_results: dict,
    latency_results: list[dict],
    memory_results: dict,
    android_estimates: list[dict],
    onnx_result: dict,
) -> str:
    """Generate a formatted report and save to docs/benchmark_results.md."""
    print("\n" + "=" * 70)
    print("6. SUMMARY REPORT")
    print("=" * 70)

    # --- Console summary table ---
    print(f"\n  {'Component':<25} | {'Size':>9} | {'Latency (ms)':>14} | {'Est. Android (ms)':>20}")
    print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*14}-+-{'-'*20}")

    summary_rows = []

    for ae in android_estimates:
        # Find size for this component
        size_str = "<1 KB"
        if "Handcrafted" in ae["component"]:
            size_str = "<1 KB"
        elif "TF-IDF" in ae["component"]:
            tfidf_path = os.path.join(MODELS_DIR, "tfidf_call_vectorizer.pkl")
            if os.path.exists(tfidf_path):
                size_str = format_size(os.path.getsize(tfidf_path))
            else:
                size_str = "~19 KB"
        elif "XGBoost" in ae["component"]:
            xgb_path = os.path.join(MODELS_DIR, "call_fraud_xgb.json")
            if os.path.exists(xgb_path):
                size_str = format_size(os.path.getsize(xgb_path))
            else:
                size_str = "~211 KB"
        elif "MLP" in ae["component"]:
            mlp_path = os.path.join(MODELS_DIR, "call_fraud_mlp.npz")
            if os.path.exists(mlp_path):
                size_str = format_size(os.path.getsize(mlp_path))
            else:
                size_str = "~240 KB"
        elif "MiniLM" in ae["component"]:
            size_str = f"~{MINILM_ONNX_SIZE_MB:.0f} MB"

        android_str = f"{ae['android_cpu_low']:.1f} - {ae['android_cpu_high']:.1f}"

        summary_rows.append({
            "component": ae["component"],
            "size": size_str,
            "latency": f"{ae['desktop_ms']:.3f}",
            "android": android_str,
        })

        # Truncate component name for display
        display_name = ae["component"][:25]
        print(f"  {display_name:<25} | {size_str:>9} | {ae['desktop_ms']:>14.3f} | {android_str:>20}")

    # Totals
    total_desktop = sum(ae["desktop_ms"] for ae in android_estimates)
    total_android_low = sum(ae["android_cpu_low"] for ae in android_estimates)
    total_android_high = sum(ae["android_cpu_high"] for ae in android_estimates)

    print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*14}-+-{'-'*20}")
    print(f"  {'TOTAL (excl. Whisper)':<25} | {'':>9} | {total_desktop:>14.3f} | "
          f"{total_android_low:.1f} - {total_android_high:.1f}")
    print(f"  {'+ Whisper Tiny':<25} | {'~150 MB':>9} | {WHISPER_DESKTOP_LATENCY_MS:>14.0f} | "
          f"{WHISPER_ANDROID_LATENCY_LOW_MS:.0f} - {WHISPER_ANDROID_LATENCY_HIGH_MS:.0f}")

    grand_desktop = total_desktop + WHISPER_DESKTOP_LATENCY_MS
    grand_android_low = total_android_low + WHISPER_ANDROID_LATENCY_LOW_MS
    grand_android_high = total_android_high + WHISPER_ANDROID_LATENCY_HIGH_MS

    print(f"  {'GRAND TOTAL':<25} |           | {grand_desktop:>14.1f} | "
          f"{grand_android_low:.0f} - {grand_android_high:.0f}")

    # --- Generate markdown report ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = [
        "# SentinelEdge Mobile Benchmark Results",
        "",
        f"Generated: {now}",
        f"Iterations: {N_ITERATIONS}",
        "",
        "## 1. Model Size Analysis",
        "",
        "| Model | Size | Status |",
        "|-------|------|--------|",
    ]

    for name, info in size_results.items():
        status = "Found" if info["exists"] else "Estimated"
        md_lines.append(f"| {name} | {format_size(info['size_bytes'])} | {status} |")

    total_size = sum(r["size_bytes"] for r in size_results.values())
    md_lines.extend([
        "",
        f"**Total estimated on-device footprint:** ~{total_size / (1024*1024):.1f} MB",
        f"**Budget:** {TOTAL_SIZE_BUDGET_MB:.0f} MB",
        f"**Status:** {'WITHIN BUDGET' if total_size <= TOTAL_SIZE_BUDGET_MB * 1024 * 1024 else 'OVER BUDGET'}",
        "",
    ])

    md_lines.extend([
        "## 2. Inference Latency",
        "",
        "| Component | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |",
        "|-----------|-----------|----------|----------|----------|",
    ])

    for r in latency_results:
        s = r["stats"]
        available = "(measured)" if r.get("available", True) and r["latencies_ms"] else "(estimated)"
        md_lines.append(
            f"| {r['component']} | {s['mean']:.3f} | {s['p50']:.3f} | "
            f"{s['p95']:.3f} | {s['p99']:.3f} | {available}"
        )

    md_lines.extend([
        "",
        "## 3. Memory Profiling",
        "",
        "| Component | Memory |",
        "|-----------|--------|",
    ])
    for name, mem_bytes in memory_results.items():
        md_lines.append(f"| {name} | {format_size(mem_bytes)} |")

    md_lines.extend([
        "",
        "## 4. Android Performance Estimation",
        "",
        f"Scaling factors (Snapdragon 6 Gen 1 mid-range vs. laptop CPU):",
        f"- CPU workloads: {ANDROID_CPU_SLOW}x - {ANDROID_CPU_FAST}x slower",
        f"- NPU-accelerated: {ANDROID_NPU_SLOW}x - {ANDROID_NPU_FAST}x slower",
        "",
        "| Component | Desktop (ms) | Android CPU (ms) | Android NPU (ms) |",
        "|-----------|-------------|------------------|-------------------|",
    ])

    for ae in android_estimates:
        npu_str = "N/A"
        if ae.get("android_npu_low") is not None:
            npu_str = f"{ae['android_npu_low']:.1f} - {ae['android_npu_high']:.1f}"
        md_lines.append(
            f"| {ae['component']} | {ae['desktop_ms']:.3f} | "
            f"{ae['android_cpu_low']:.1f} - {ae['android_cpu_high']:.1f} | "
            f"{npu_str} |"
        )

    md_lines.extend([
        "",
        "### End-to-End Pipeline Estimate",
        "",
        "| Pipeline Stage | Desktop (ms) | Android (ms) |",
        "|---------------|-------------|-------------|",
        f"| Classification pipeline | {total_desktop:.1f} | {total_android_low:.0f} - {total_android_high:.0f} |",
        f"| + Whisper Tiny (10s audio) | {WHISPER_DESKTOP_LATENCY_MS:.0f} | "
        f"{WHISPER_ANDROID_LATENCY_LOW_MS:.0f} - {WHISPER_ANDROID_LATENCY_HIGH_MS:.0f} |",
        f"| **Grand Total** | **{grand_desktop:.0f}** | "
        f"**{grand_android_low:.0f} - {grand_android_high:.0f}** |",
        "",
    ])

    md_lines.extend([
        "## 5. ONNX Export Verification",
        "",
        f"- Exported: {'Yes' if onnx_result['exported'] else 'No'}",
        f"- Verified: {'Yes' if onnx_result['verified'] else 'No'}",
    ])
    if onnx_result.get("onnx_latency_ms") is not None:
        md_lines.append(f"- ONNX inference latency: {onnx_result['onnx_latency_ms']:.3f} ms")

    md_lines.extend([
        "",
        "## 6. Reproduction Instructions",
        "",
        "### Desktop Benchmark",
        "```bash",
        "cd SentinelEdge",
        "python3 training/benchmark_mobile.py",
        "```",
        "",
        "### Android Device Benchmark",
        "",
        "To reproduce on a real Android device:",
        "",
        "1. **Export models to ONNX:**",
        "   ```bash",
        "   python3 training/export_onnx.py",
        "   ```",
        "",
        "2. **Build the Android benchmark app:**",
        "   - Open `android/` project in Android Studio",
        "   - Copy ONNX models to `app/src/main/assets/`",
        "   - Run the benchmark Activity on device",
        "",
        "3. **Use ONNX Runtime Mobile:**",
        "   ```kotlin",
        "   // In your Android benchmark code:",
        "   val session = OrtEnvironment.getEnvironment()",
        "   val options = OrtSession.SessionOptions()",
        "   // Enable NNAPI for NPU acceleration:",
        "   options.addNnapi()",
        "   val ortSession = session.createSession(modelPath, options)",
        "   ```",
        "",
        "4. **Measure latency per component:**",
        "   - Time each ONNX session.run() call",
        "   - Run 100+ iterations, discard first 10 (warmup)",
        "   - Report mean, P50, P95, P99",
        "",
        "5. **Compare with/without NNAPI:**",
        "   - CPU-only: remove `addNnapi()` call",
        "   - NNAPI (NPU): include `addNnapi()` call",
        "   - Some models may not benefit from NPU",
        "",
        "### Key Android Considerations",
        "",
        "- **Thermal throttling:** Run benchmarks with device at room temp",
        "- **Background processes:** Use airplane mode, close other apps",
        "- **Battery state:** Benchmark at >50% battery (avoid power-saving mode)",
        "- **Memory pressure:** Monitor `adb shell dumpsys meminfo` during benchmark",
        "- **CPU governor:** Lock to performance mode if rooted:",
        "  ```bash",
        "  adb shell 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'",
        "  ```",
        "",
        "## Methodology Notes",
        "",
        f"- Desktop benchmarks run {N_ITERATIONS} iterations with 10-iteration warmup",
        "- Android estimates use published CPU scaling factors:",
        f"  - Mid-range (Snapdragon 6 Gen 1): {ANDROID_CPU_SLOW}x - {ANDROID_CPU_FAST}x vs. laptop",
        f"  - NPU acceleration: {ANDROID_NPU_SLOW}x - {ANDROID_NPU_FAST}x vs. laptop",
        "- Whisper Tiny estimates from published ONNX Runtime Mobile benchmarks",
        "- MiniLM ONNX size estimate (~22 MB) from HuggingFace model card",
        "- Memory profiling uses Python tracemalloc (may underestimate native allocations)",
        "",
    ])

    md_content = "\n".join(md_lines)

    # Save to docs/
    os.makedirs(DOCS_DIR, exist_ok=True)
    results_path = os.path.join(DOCS_DIR, "benchmark_results.md")
    with open(results_path, "w") as f:
        f.write(md_content)
    print(f"\n  Report saved to: {results_path}")

    return md_content


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    """Run the full mobile deployment benchmark suite."""
    print("=" * 70)
    print("SentinelEdge Mobile Deployment Benchmark")
    print("=" * 70)
    print(f"  Project root: {_PROJECT_ROOT}")
    print(f"  Models dir:   {MODELS_DIR}")
    print(f"  Iterations:   {N_ITERATIONS}")
    print(f"  Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Model sizes
    size_results = analyze_model_sizes()

    # 2. Latency benchmarks
    latency_results = run_latency_benchmarks()

    # 3. Memory profiling
    memory_results = profile_memory()

    # 4. Android estimates
    android_estimates = estimate_android_performance(latency_results)

    # 5. ONNX export verification
    onnx_result = verify_onnx_export()

    # 6. Report
    generate_report(size_results, latency_results, memory_results, android_estimates, onnx_result)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
