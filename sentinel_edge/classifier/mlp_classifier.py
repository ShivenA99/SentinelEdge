"""MLP fraud classifier -- differentiable for federated learning.

Architecture: input_dim -> 128 -> ReLU -> 64 -> ReLU -> 1 -> Sigmoid

The classifier exposes ``get_weights`` / ``set_weights`` so that the
federated learning protocol can exchange flat weight vectors between
edge devices and the central hub.

For input_dim=402 (18 handcrafted + 384 embedding):
  Total params: 402*128 + 128 + 128*64 + 64 + 64*1 + 1 = 59,969
  Size:         ~240 KB (float32)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MLPClassifier:
    """Numpy-based MLP binary classifier with federated-learning support.

    Parameters
    ----------
    input_dim : int
        Number of input features.  Default 402 matches the embedding
        pipeline (18 handcrafted + 384 sentence embedding).
    """

    def __init__(self, input_dim: int = 402) -> None:
        self.input_dim = input_dim
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (He / Kaiming)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights using He initialisation."""
        rng = np.random.default_rng(seed=42)

        self.W1 = (
            rng.standard_normal((self.input_dim, 128)).astype(np.float32)
            * np.sqrt(2.0 / self.input_dim)
        )
        self.b1 = np.zeros(128, dtype=np.float32)

        self.W2 = (
            rng.standard_normal((128, 64)).astype(np.float32)
            * np.sqrt(2.0 / 128)
        )
        self.b2 = np.zeros(64, dtype=np.float32)

        self.W3 = (
            rng.standard_normal((64, 1)).astype(np.float32)
            * np.sqrt(2.0 / 64)
        )
        self.b3 = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> float:
        """Run the forward pass for a single sample.

        Parameters
        ----------
        x : np.ndarray
            1-D feature vector of length ``input_dim``.

        Returns
        -------
        float
            Fraud probability in ``[0, 1]``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return float(self.forward_batch(x)[0])
        h1 = np.maximum(0, x @ self.W1 + self.b1)          # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)         # ReLU
        logit = float(h2 @ self.W3 + self.b3)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))  # Sigmoid

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Run the forward pass for a batch of samples.

        Parameters
        ----------
        X : np.ndarray
            2-D array of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            1-D array of fraud probabilities, shape ``(n_samples,)``.
        """
        X = np.asarray(X, dtype=np.float32)
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = (h2 @ self.W3 + self.b3).ravel()
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    # ------------------------------------------------------------------
    # Prediction interface (drop-in compatible with FraudClassifier)
    # ------------------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> float:
        """Return fraud probability for a single feature vector.

        Same interface as ``FraudClassifier.predict_proba`` for
        drop-in replacement in the engine.
        """
        return float(self.forward(features))

    def predict(
        self, features: np.ndarray, threshold: float = 0.5
    ) -> tuple[bool, float]:
        """Binary prediction with confidence score.

        Parameters
        ----------
        features : np.ndarray
            1-D feature vector.
        threshold : float
            Decision threshold.

        Returns
        -------
        tuple[bool, float]
            ``(is_fraud, confidence)``.
        """
        prob = self.predict_proba(features)
        return prob >= threshold, prob

    # ------------------------------------------------------------------
    # Federated learning: weight serialisation
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a single 1-D vector.

        Order: W1, b1, W2, b2, W3, b3.

        Returns
        -------
        np.ndarray
            1-D float32 vector of length ``n_params``.
        """
        return np.concatenate(
            [
                self.W1.ravel(),
                self.b1,
                self.W2.ravel(),
                self.b2,
                self.W3.ravel(),
                self.b3,
            ]
        )

    def set_weights(self, weights: np.ndarray) -> None:
        """Set all weights from a flat vector (inverse of ``get_weights``).

        Parameters
        ----------
        weights : np.ndarray
            1-D vector of length ``n_params``.
        """
        weights = np.asarray(weights, dtype=np.float32)
        idx = 0

        # W1: (input_dim, 128)
        n = self.input_dim * 128
        self.W1 = weights[idx : idx + n].reshape(self.input_dim, 128)
        idx += n

        # b1: (128,)
        self.b1 = weights[idx : idx + 128].copy()
        idx += 128

        # W2: (128, 64)
        n = 128 * 64
        self.W2 = weights[idx : idx + n].reshape(128, 64)
        idx += n

        # b2: (64,)
        self.b2 = weights[idx : idx + 64].copy()
        idx += 64

        # W3: (64, 1)
        n = 64
        self.W3 = weights[idx : idx + n].reshape(64, 1)
        idx += n

        # b3: (1,)
        self.b3 = weights[idx : idx + 1].copy()

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return (
            self.input_dim * 128
            + 128
            + 128 * 64
            + 64
            + 64 * 1
            + 1
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a ``.npz`` file.

        Saves via the flat weight vector so that ``load`` can restore
        using ``set_weights``, keeping serialisation consistent with
        the federated learning protocol.

        Parameters
        ----------
        path : str
            Destination path (e.g. ``models/call_fraud_mlp.npz``).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest,
            weights=self.get_weights(),
            input_dim=np.array(self.input_dim),
        )
        logger.info(
            "Saved MLP classifier (%d params) to %s", self.n_params, path
        )

    @classmethod
    def load(cls, path: str) -> MLPClassifier:
        """Load model weights from a ``.npz`` file.

        Parameters
        ----------
        path : str
            Path to a previously saved ``.npz`` file.

        Returns
        -------
        MLPClassifier
            Restored classifier instance.
        """
        data = np.load(path)
        model = cls(input_dim=int(data["input_dim"]))
        model.set_weights(data["weights"])
        logger.info(
            "Loaded MLP classifier (%d params) from %s",
            model.n_params,
            path,
        )
        return model

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MLPClassifier(input_dim={self.input_dim}, "
            f"arch=[{self.input_dim}->128->64->1], "
            f"params={self.n_params:,})"
        )
