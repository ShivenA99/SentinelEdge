"""Export XGBoost models to ONNX format with INT8 quantization.

Converts trained XGBoost .json models to ONNX for efficient edge inference
using ONNX Runtime.  Applies INT8 static quantization to reduce model size
and improve inference speed on resource-constrained devices.

Usage
-----
    python training/export_onnx.py [--models-dir models]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def export_to_onnx(
    model_path: str,
    output_path: str,
    n_features: int,
    quantize: bool = True,
) -> None:
    """Export an XGBoost model to ONNX format with optional INT8 quantization.

    Parameters
    ----------
    model_path : str
        Path to the XGBoost model file (.json).
    output_path : str
        Destination path for the ONNX model (.onnx).
    n_features : int
        Number of input features the model expects.
    quantize : bool
        Whether to apply INT8 quantization (default: True).
    """
    import xgboost as xgb

    if not os.path.exists(model_path):
        print(f"  WARNING: Model file {model_path} not found, skipping.")
        return

    print(f"\nExporting {model_path} -> {output_path}")
    print(f"  Input features: {n_features}")

    # ---- Load XGBoost model ----
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()

    # ---- Convert to ONNX using onnxmltools ----
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxconverter_common import FloatTensorType
    except ImportError:
        print("  ERROR: onnxmltools and onnxconverter-common are required.")
        print("  Install with: pip install onnxmltools onnxconverter-common")
        # Fallback: try using skl2onnx
        _export_via_skl2onnx(model, output_path, n_features, quantize)
        return

    # Define input type
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    # Convert
    print("  Converting XGBoost model to ONNX ...")
    onnx_model = convert_xgboost(
        booster,
        initial_types=initial_type,
        target_opset=13,
    )

    # ---- Save unquantized model first ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if quantize:
        _save_and_quantize(onnx_model, output_path, n_features)
    else:
        import onnx
        onnx.save_model(onnx_model, output_path)
        _report_size(output_path, "ONNX (float32)")

    # ---- Validate the exported model ----
    _validate_onnx(output_path, n_features)


def _export_via_skl2onnx(
    model,
    output_path: str,
    n_features: int,
    quantize: bool,
) -> None:
    """Fallback export path using skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("  ERROR: skl2onnx is required for fallback conversion.")
        print("  Install with: pip install skl2onnx")
        return

    print("  Using skl2onnx fallback conversion ...")
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if quantize:
        _save_and_quantize(onnx_model, output_path, n_features)
    else:
        import onnx
        onnx.save_model(onnx_model, output_path)
        _report_size(output_path, "ONNX (float32)")

    _validate_onnx(output_path, n_features)


def _save_and_quantize(
    onnx_model,
    output_path: str,
    n_features: int,
) -> None:
    """Save ONNX model, then apply INT8 static quantization.

    Saves both the full-precision model (with _fp32 suffix) and the
    quantized model at the requested output_path.
    """
    import onnx

    # Save the float32 model to a temporary file for quantization input
    fp32_path = output_path.replace(".onnx", "_fp32.onnx")
    onnx.save_model(onnx_model, fp32_path)
    _report_size(fp32_path, "ONNX (float32)")

    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError:
        print("  WARNING: onnxruntime quantization module not available.")
        print("  Saving float32 model only.")
        os.rename(fp32_path, output_path)
        return

    # ------------------------------------------------------------------
    # Dynamic quantization (simpler, no calibration data needed)
    # ------------------------------------------------------------------
    print("  Applying INT8 dynamic quantization ...")
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        _report_size(output_path, "ONNX (INT8 quantized)")
    except Exception as e:
        print(f"  WARNING: Dynamic quantization failed: {e}")
        print("  Trying static quantization with synthetic calibration data ...")

        # ----------------------------------------------------------
        # Static quantization with synthetic calibration data
        # ----------------------------------------------------------
        class SyntheticCalibrationReader(CalibrationDataReader):
            """Generate random calibration data for static quantization."""

            def __init__(self, n_features: int, n_samples: int = 100):
                self.data = iter([
                    {"float_input": np.random.randn(1, n_features).astype(np.float32)}
                    for _ in range(n_samples)
                ])

            def get_next(self):
                return next(self.data, None)

        try:
            calibration_reader = SyntheticCalibrationReader(n_features)
            quantize_static(
                model_input=fp32_path,
                model_output=output_path,
                calibration_data_reader=calibration_reader,
                quant_format=QuantType.QInt8,
            )
            _report_size(output_path, "ONNX (INT8 static quantized)")
        except Exception as e2:
            print(f"  WARNING: Static quantization also failed: {e2}")
            print("  Falling back to float32 model.")
            if os.path.exists(fp32_path):
                os.rename(fp32_path, output_path)
            return

    # Clean up fp32 model if quantized version was created successfully
    if os.path.exists(output_path) and os.path.exists(fp32_path):
        # Keep both for comparison
        print(f"  Float32 model preserved at {fp32_path}")


def _report_size(path: str, label: str) -> None:
    """Print the file size of a model."""
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024*1024):.2f} MB"
        else:
            size_str = f"{size_bytes / 1024:.2f} KB"
        print(f"  {label}: {size_str}")


def _validate_onnx(output_path: str, n_features: int) -> None:
    """Validate the ONNX model by running a test inference."""
    if not os.path.exists(output_path):
        print("  Validation skipped: output file not found.")
        return

    try:
        import onnxruntime as ort

        print("  Validating ONNX model ...")
        sess = ort.InferenceSession(output_path)

        # Get input/output info
        input_info = sess.get_inputs()
        output_info = sess.get_outputs()
        print(f"    Inputs:  {[(i.name, i.shape, i.type) for i in input_info]}")
        print(f"    Outputs: {[(o.name, o.shape, o.type) for o in output_info]}")

        # Run test inference
        test_input = np.random.randn(1, n_features).astype(np.float32)
        input_name = input_info[0].name
        results = sess.run(None, {input_name: test_input})

        print(f"    Test inference output shapes: {[r.shape for r in results]}")
        print("  Validation PASSED")

    except Exception as e:
        print(f"  Validation WARNING: {e}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export XGBoost models to ONNX with INT8 quantization."
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing XGBoost models (default: models)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization (export float32 only)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir
    quantize = not args.no_quantize

    print("=" * 60)
    print("Exporting XGBoost models to ONNX")
    print("=" * 60)

    # Define models to export: (json_path, onnx_path, n_features)
    exports = [
        (
            os.path.join(models_dir, "call_fraud_xgb.json"),
            os.path.join(models_dir, "call_fraud_xgb.onnx"),
            518,  # 18 handcrafted + 500 TF-IDF
        ),
        (
            os.path.join(models_dir, "sms_fraud_xgb.json"),
            os.path.join(models_dir, "sms_fraud_xgb.onnx"),
            518,  # 18 handcrafted + 500 TF-IDF
        ),
        (
            os.path.join(models_dir, "url_fraud_xgb.json"),
            os.path.join(models_dir, "url_fraud_xgb.onnx"),
            14,   # 14 URL structural features
        ),
    ]

    for json_path, onnx_path, n_features in exports:
        export_to_onnx(json_path, onnx_path, n_features, quantize=quantize)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    # List output files
    print("\nGenerated ONNX files:")
    for _, onnx_path, _ in exports:
        if os.path.exists(onnx_path):
            _report_size(onnx_path, f"  {onnx_path}")
        # Also check for fp32 version
        fp32_path = onnx_path.replace(".onnx", "_fp32.onnx")
        if os.path.exists(fp32_path):
            _report_size(fp32_path, f"  {fp32_path}")


if __name__ == "__main__":
    main()
