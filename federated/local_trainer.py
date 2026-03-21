"""On-device local training with real MLP backpropagation."""
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
    """On-device model fine-tuning via real mini-batch SGD on a 3-layer MLP.

    The MLP architecture mirrors MLPClassifier:
        input -> hidden1 (128, ReLU) -> hidden2 (64, ReLU) -> output (1, Sigmoid)

    Fine-tuning performs real numpy backpropagation through all three layers
    with BCE loss.
    """

    # MLP architecture constants (must match MLPClassifier)
    HIDDEN1_DIM = 128
    HIDDEN2_DIM = 64
    OUTPUT_DIM = 1

    def __init__(self, device_id: str, input_dim: int = 402):
        self.device_id = device_id
        self.input_dim = input_dim
        self.buffer = LocalTrainingBuffer()
        self.current_model_version = 0

    def ingest_call_data(self, features: np.ndarray, label: int):
        """Store a labeled sample from user feedback."""
        self.buffer.add(features, label)

    def _unpack_weights(self, flat_weights: np.ndarray) -> dict:
        """Unpack a flat weight vector into the six MLP parameter arrays."""
        idx = 0
        params = {}

        size = self.input_dim * self.HIDDEN1_DIM
        params['W1'] = flat_weights[idx:idx + size].reshape(self.input_dim, self.HIDDEN1_DIM)
        idx += size

        size = self.HIDDEN1_DIM
        params['b1'] = flat_weights[idx:idx + size].copy()
        idx += size

        size = self.HIDDEN1_DIM * self.HIDDEN2_DIM
        params['W2'] = flat_weights[idx:idx + size].reshape(self.HIDDEN1_DIM, self.HIDDEN2_DIM)
        idx += size

        size = self.HIDDEN2_DIM
        params['b2'] = flat_weights[idx:idx + size].copy()
        idx += size

        size = self.HIDDEN2_DIM * self.OUTPUT_DIM
        params['W3'] = flat_weights[idx:idx + size].reshape(self.HIDDEN2_DIM, self.OUTPUT_DIM)
        idx += size

        size = self.OUTPUT_DIM
        params['b3'] = flat_weights[idx:idx + size].copy()
        idx += size

        return params

    def _pack_weights(self, params: dict) -> np.ndarray:
        """Pack the six MLP parameter arrays into a flat weight vector."""
        return np.concatenate([
            params['W1'].ravel(), params['b1'].ravel(),
            params['W2'].ravel(), params['b2'].ravel(),
            params['W3'].ravel(), params['b3'].ravel(),
        ])

    def fine_tune(self, global_model_weights: np.ndarray,
                  lr: float = 0.01, n_epochs: int = 3,
                  batch_size: int = 32) -> np.ndarray:
        """Fine-tune on local buffer using real mini-batch SGD with numpy backprop.

        Steps:
            1. Load global MLP weights via unpack
            2. Run forward pass on each mini-batch
            3. Compute BCE loss gradient (backprop through 3-layer MLP)
            4. Update weights with SGD
            5. Return gradient delta = local_weights - global_weights

        Args:
            global_model_weights: Flat 1D array of all MLP parameters.
            lr: Learning rate for SGD.
            n_epochs: Number of training epochs over local data.
            batch_size: Mini-batch size.

        Returns:
            Gradient delta (flat 1D array): local_weights - global_weights.
        """
        X, y = self.buffer.to_arrays()
        if X.shape[0] == 0:
            return np.zeros_like(global_model_weights)

        n_samples = X.shape[0]

        # Pad or truncate features to match input_dim
        if X.shape[1] < self.input_dim:
            pad = np.zeros((n_samples, self.input_dim - X.shape[1]))
            X = np.hstack([X, pad])
        elif X.shape[1] > self.input_dim:
            X = X[:, :self.input_dim]

        # Unpack global weights into local parameters
        params = self._unpack_weights(global_model_weights)

        # Mini-batch SGD with real backpropagation
        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]  # (B, input_dim)
                y_batch = y[batch_idx].reshape(-1, 1).astype(np.float64)  # (B, 1)
                B = X_batch.shape[0]

                # ---- Forward pass ----
                # Layer 1: input -> hidden1
                z1 = X_batch @ params['W1'] + params['b1']  # (B, 128)
                h1 = np.maximum(0, z1)  # ReLU

                # Layer 2: hidden1 -> hidden2
                z2 = h1 @ params['W2'] + params['b2']  # (B, 64)
                h2 = np.maximum(0, z2)  # ReLU

                # Layer 3: hidden2 -> output
                logit = h2 @ params['W3'] + params['b3']  # (B, 1)
                pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))  # sigmoid

                # ---- Backward pass (BCE loss) ----
                # d(BCE)/d(logit) = pred - y
                dlogit = (pred - y_batch) / B  # (B, 1), averaged over batch

                # Layer 3 gradients
                dW3 = h2.T @ dlogit  # (64, 1)
                db3 = np.sum(dlogit, axis=0)  # (1,)

                # Backprop into h2
                dh2 = dlogit @ params['W3'].T  # (B, 64)
                dh2 = dh2 * (z2 > 0)  # ReLU backward

                # Layer 2 gradients
                dW2 = h1.T @ dh2  # (128, 64)
                db2 = np.sum(dh2, axis=0)  # (64,)

                # Backprop into h1
                dh1 = dh2 @ params['W2'].T  # (B, 128)
                dh1 = dh1 * (z1 > 0)  # ReLU backward

                # Layer 1 gradients
                dW1 = X_batch.T @ dh1  # (input_dim, 128)
                db1 = np.sum(dh1, axis=0)  # (128,)

                # ---- SGD update ----
                params['W1'] -= lr * dW1
                params['b1'] -= lr * db1
                params['W2'] -= lr * dW2
                params['b2'] -= lr * db2
                params['W3'] -= lr * dW3
                params['b3'] -= lr * db3

        # Return the delta: local_weights - global_weights
        local_flat = self._pack_weights(params)
        delta = local_flat - global_model_weights
        return delta
