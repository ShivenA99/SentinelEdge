"""XGBoost fraud classifier with native and ONNX inference support.

Loads a trained binary classifier from either:
  - A native XGBoost JSON checkpoint (``.json``)
  - An ONNX model exported from XGBoost (``.onnx``)

and exposes a simple ``predict`` / ``predict_proba`` interface suitable
for real-time, single-sample inference on the edge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FraudClassifier:
    """Lightweight inference wrapper around a trained XGBoost model.

    Parameters
    ----------
    model_path : str
        Path to the model file.  The backend is chosen by extension:
        - ``.json`` -- native XGBoost (``xgboost.Booster``)
        - ``.onnx`` -- ONNX Runtime inference session
    """

    def __init__(self, model_path: str) -> None:
        self._path = Path(model_path)
        self._backend: str = ""  # "xgboost" | "onnx"
        self._model = None  # xgboost.Booster or ort.InferenceSession

        self._load(self._path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Dispatch loading to the correct backend based on file extension."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            self._load_xgboost(path)
        elif suffix == ".onnx":
            self._load_onnx(path)
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}'. "
                "Expected '.json' (native XGBoost) or '.onnx'."
            )

    def _load_xgboost(self, path: Path) -> None:
        """Load a native XGBoost Booster from a JSON checkpoint."""
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        self._model = booster
        self._backend = "xgboost"
        logger.info("Loaded native XGBoost model from %s", path)

    def _load_onnx(self, path: Path) -> None:
        """Load an ONNX model via ONNX Runtime."""
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self._model = ort.InferenceSession(
            str(path), sess_options=sess_opts
        )
        self._backend = "onnx"
        logger.info("Loaded ONNX model from %s", path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> float:
        """Return the fraud probability for a single feature vector.

        Parameters
        ----------
        features : np.ndarray
            1-D feature vector (e.g. 518-dim from ``FeaturePipeline``).

        Returns
        -------
        float
            Probability of the positive (fraud) class in ``[0.0, 1.0]``.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._backend == "xgboost":
            return self._predict_xgb(features)
        elif self._backend == "onnx":
            return self._predict_onnx(features)
        else:
            raise RuntimeError("Model not loaded.")

    def predict(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """Binary prediction with confidence score.

        Parameters
        ----------
        features : np.ndarray
            1-D feature vector.
        threshold : float
            Decision threshold.  Scores >= threshold are classified as
            fraud.

        Returns
        -------
        tuple[bool, float]
            ``(is_fraud, confidence)`` where *confidence* is the raw
            fraud probability.
        """
        prob = self.predict_proba(features)
        return (prob >= threshold, prob)

    # ------------------------------------------------------------------
    # Backend-specific prediction
    # ------------------------------------------------------------------

    def _predict_xgb(self, features_2d: np.ndarray) -> float:
        """Run prediction through the native XGBoost Booster."""
        import xgboost as xgb

        dmat = xgb.DMatrix(features_2d)
        preds = self._model.predict(dmat)  # shape (1,) for single sample
        return float(preds[0])

    def _predict_onnx(self, features_2d: np.ndarray) -> float:
        """Run prediction through ONNX Runtime."""
        input_name = self._model.get_inputs()[0].name
        output = self._model.run(None, {input_name: features_2d})

        # ONNX XGBoost classifiers typically output:
        #   [0] -> predicted labels, [1] -> class probabilities
        if len(output) >= 2:
            # output[1] is a list of dicts [{0: p0, 1: p1}] or ndarray
            probs = output[1]
            if isinstance(probs, list):
                # List of dicts: [{0: 0.2, 1: 0.8}]
                return float(probs[0][1])
            else:
                # ndarray of shape (1, 2)
                return float(probs[0, 1])
        else:
            # Single output: raw score (sigmoid already applied for
            # binary:logistic objective)
            return float(output[0][0])
