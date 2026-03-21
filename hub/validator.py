"""Model validator for SentinelEdge Hub.

Validates new global model weights against a held-out validation set
derived from public FTC fraud report data. Implements Byzantine fault
tolerance by rejecting model updates that degrade F1 by more than 2%.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Number of features the model expects (phone call fraud features).
DEFAULT_N_FEATURES = 25


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _generate_synthetic_validation_data(
    n_samples: int = 2000,
    n_features: int = DEFAULT_N_FEATURES,
    fraud_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic validation set mimicking FTC fraud report patterns.

    This produces a reproducible dataset with realistic class imbalance
    (~15% fraud) and feature distributions inspired by phone call fraud
    indicators: call duration, time of day, caller ID mismatch, etc.

    Args:
        n_samples: Total number of validation samples.
        n_features: Number of features per sample.
        fraud_ratio: Fraction of positive (fraud) samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    rng = np.random.RandomState(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate calls: features drawn from a "normal" distribution
    X_legit = rng.randn(n_legit, n_features) * 0.8

    # Fraud calls: shifted distribution -- certain features are elevated
    X_fraud = rng.randn(n_fraud, n_features) * 1.0
    # Fraud indicators: features 0-4 tend to be elevated in fraud calls
    X_fraud[:, 0] += 2.0   # call duration anomaly
    X_fraud[:, 1] += 1.5   # time-of-day anomaly
    X_fraud[:, 2] += 1.8   # caller-id mismatch score
    X_fraud[:, 3] += 1.2   # rapid callback pattern
    X_fraud[:, 4] += 1.0   # geographic anomaly

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    return X, y


class ModelValidator:
    """Validates global model against held-out validation data.

    If a validation_data_path is provided and exists, loads it as a
    numpy .npz file with keys 'X' and 'y'. Otherwise, generates a
    reproducible synthetic dataset.
    """

    def __init__(
        self,
        validation_data_path: str = "validation_data.npz",
        n_features: int = DEFAULT_N_FEATURES,
    ):
        """Load or generate validation dataset.

        Args:
            validation_data_path: Path to .npz file with 'X' and 'y' keys.
            n_features: Expected number of features (used for synthetic data).
        """
        self.n_features = n_features
        self.X: np.ndarray
        self.y: np.ndarray

        if os.path.exists(validation_data_path):
            try:
                data = np.load(validation_data_path)
                self.X = data["X"]
                self.y = data["y"]
                logger.info(
                    "Loaded validation data from %s: %d samples",
                    validation_data_path,
                    len(self.y),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load %s (%s), generating synthetic data",
                    validation_data_path,
                    e,
                )
                self.X, self.y = _generate_synthetic_validation_data(
                    n_features=n_features
                )
        else:
            logger.info(
                "Validation data file not found at %s, "
                "generating synthetic validation set",
                validation_data_path,
            )
            self.X, self.y = _generate_synthetic_validation_data(
                n_features=n_features
            )

        logger.info(
            "Validation set: %d samples, %.1f%% fraud",
            len(self.y),
            100.0 * np.mean(self.y),
        )

    def _predict(self, model_weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Run logistic regression prediction with the given weights.

        The model is a simple linear model: y_hat = sigmoid(X @ w).
        Weights vector is expected to be of length n_features (no bias)
        or n_features+1 (last element is bias).

        Args:
            model_weights: Weight vector.
            threshold: Classification threshold.

        Returns:
            Binary predictions array.
        """
        w = np.array(model_weights, dtype=np.float64)

        if len(w) == self.X.shape[1] + 1:
            # Last element is bias
            logits = self.X @ w[:-1] + w[-1]
        elif len(w) == self.X.shape[1]:
            logits = self.X @ w
        else:
            # Dimension mismatch: truncate or pad with zeros
            logger.warning(
                "Weight dimension %d != feature dimension %d, adjusting",
                len(w),
                self.X.shape[1],
            )
            adjusted = np.zeros(self.X.shape[1], dtype=np.float64)
            n = min(len(w), self.X.shape[1])
            adjusted[:n] = w[:n]
            logits = self.X @ adjusted

        probs = _sigmoid(logits)
        return (probs >= threshold).astype(np.float64)

    def validate(
        self, model_weights: np.ndarray, previous_f1: float
    ) -> tuple[float, bool]:
        """Test model, return (f1_score, should_accept).

        Reject if F1 drops more than 2% from previous round
        (Byzantine fault tolerance).

        Args:
            model_weights: New global model weight vector.
            previous_f1: F1 score from the previous round.

        Returns:
            Tuple of (f1_score, should_accept).
        """
        metrics = self.compute_metrics(model_weights)
        f1 = metrics["f1"]

        # Accept if F1 is within 2% of previous, or improved
        max_allowed_drop = 0.02
        should_accept = f1 >= (previous_f1 - max_allowed_drop)

        if not should_accept:
            logger.warning(
                "Model REJECTED: F1=%.4f, previous=%.4f, drop=%.4f > %.4f",
                f1,
                previous_f1,
                previous_f1 - f1,
                max_allowed_drop,
            )
        else:
            logger.info(
                "Model ACCEPTED: F1=%.4f (previous=%.4f, delta=%+.4f)",
                f1,
                previous_f1,
                f1 - previous_f1,
            )

        return f1, should_accept

    def compute_metrics(self, model_weights: np.ndarray) -> dict:
        """Compute full classification metrics.

        Args:
            model_weights: Model weight vector.

        Returns:
            Dict with keys: accuracy, precision, recall, f1, auc.
        """
        y_pred = self._predict(model_weights)
        y_true = self.y

        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        tn = float(np.sum((y_pred == 0) & (y_true == 0)))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1e-10)
        recall = tp / max(tp + fn, 1e-10)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-10)
        )

        # Approximate AUC using the trapezoidal rule across thresholds
        auc = self._compute_auc(model_weights)

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
        }

    def _compute_auc(self, model_weights: np.ndarray) -> float:
        """Compute approximate AUC-ROC.

        Uses 100 threshold values to build the ROC curve and computes
        area under curve via the trapezoidal rule.
        """
        w = np.array(model_weights, dtype=np.float64)

        if len(w) == self.X.shape[1] + 1:
            logits = self.X @ w[:-1] + w[-1]
        elif len(w) == self.X.shape[1]:
            logits = self.X @ w
        else:
            adjusted = np.zeros(self.X.shape[1], dtype=np.float64)
            n = min(len(w), self.X.shape[1])
            adjusted[:n] = w[:n]
            logits = self.X @ adjusted

        probs = _sigmoid(logits)
        y_true = self.y

        thresholds = np.linspace(0, 1, 101)
        tpr_list = []
        fpr_list = []

        total_pos = max(np.sum(y_true == 1), 1)
        total_neg = max(np.sum(y_true == 0), 1)

        for t in thresholds:
            y_pred = (probs >= t).astype(np.float64)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tpr_list.append(tp / total_pos)
            fpr_list.append(fp / total_neg)

        # Sort by FPR for proper AUC calculation
        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)
        sorted_idx = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[sorted_idx]
        tpr_sorted = tpr_arr[sorted_idx]

        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
        return max(0.0, min(1.0, abs(auc)))
