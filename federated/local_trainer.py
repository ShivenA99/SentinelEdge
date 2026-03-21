"""On-device local training simulation."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LocalTrainingBuffer:
    """Encrypted local label store. Stores feature vectors + labels only."""
    features: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    max_samples: int = 500

    def add(self, feature_vector: np.ndarray, label: int):
        """Add a training sample. Auto-evicts oldest if at capacity."""
        if len(self.labels) >= self.max_samples:
            self.features.pop(0)
            self.labels.pop(0)
        self.features.append(feature_vector.copy())
        self.labels.append(label)

    def size(self) -> int:
        return len(self.labels)

    def to_arrays(self) -> tuple:
        """Return (features_array, labels_array) as numpy arrays."""
        if len(self.features) == 0:
            return np.array([]).reshape(0, 0), np.array([], dtype=int)
        X = np.vstack(self.features)
        y = np.array(self.labels, dtype=int)
        return X, y


class LocalTrainer:
    """Simulates on-device model fine-tuning."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.buffer = LocalTrainingBuffer()
        self.current_model_version = 0

    def ingest_call_data(self, features: np.ndarray, label: int):
        """Simulate user implicit labeling:
        - User blocks caller after alert -> label=1 (SCAM), high confidence
        - User reports as spam -> label=1, medium confidence
        - No alert, call lasted 5+ min -> label=0 (LEGIT), high confidence
        """
        self.buffer.add(features, label)

    def fine_tune(self, global_model_weights: np.ndarray) -> np.ndarray:
        """Fine-tune on local buffer, return weight delta.

        Uses mini-batch gradient descent on a logistic regression objective
        to simulate on-device training. The model is a simple linear classifier
        parameterised by weights (same shape as global_model_weights).

        Returns the gradient delta (local_weights - global_weights).
        """
        X, y = self.buffer.to_arrays()
        if X.shape[0] == 0:
            return np.zeros_like(global_model_weights)

        n_features = global_model_weights.shape[0]

        # If features have different dim, pad or truncate
        if X.shape[1] < n_features:
            pad = np.zeros((X.shape[0], n_features - X.shape[1]))
            X = np.hstack([X, pad])
        elif X.shape[1] > n_features:
            X = X[:, :n_features]

        # Start from the global weights
        local_weights = global_model_weights.copy()

        # Mini-batch SGD on logistic loss
        lr = 0.05
        n_epochs = 5
        batch_size = min(32, X.shape[0])
        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward pass: logistic regression
                logits = X_batch @ local_weights
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

                # Gradient of binary cross-entropy
                errors = probs - y_batch
                grad = (X_batch.T @ errors) / len(y_batch)

                # L2 regularisation to stay close to global weights
                reg_strength = 0.01
                grad += reg_strength * (local_weights - global_model_weights)

                local_weights -= lr * grad

        # Return the delta
        return self.compute_gradient_delta(local_weights, global_model_weights)

    def compute_gradient_delta(self, local_weights: np.ndarray, global_weights: np.ndarray) -> np.ndarray:
        """Compute weight difference."""
        return local_weights - global_weights
