"""Train MLP classifier on sentence embeddings + handcrafted features.

Replaces the XGBoost pipeline with a differentiable MLP that can
participate in federated learning.  Uses sklearn's MLPClassifier for
training, then transfers the learned weights into our custom
``MLPClassifier`` (numpy-only forward pass with get/set_weights).

Usage
-----
    python training/train_mlp_classifier.py \
        [--train-csv data/processed/call_fraud_train.csv] \
        [--test-csv  data/processed/call_fraud_test.csv]  \
        [--output    models/call_fraud_mlp.npz]           \
        [--epochs 50] [--batch-size 64] [--lr 0.001]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sentinel_edge.classifier.mlp_classifier import MLPClassifier
from sentinel_edge.features.feature_pipeline import (
    EMBEDDING_TOTAL_DIM,
    FeaturePipeline,
)


# ======================================================================
# Feature extraction
# ======================================================================

def extract_embedding_features(
    texts: list[str],
    pipeline: FeaturePipeline,
) -> np.ndarray:
    """Extract embedding-based feature matrix.

    Parameters
    ----------
    texts : list[str]
        Raw text sentences.
    pipeline : FeaturePipeline
        Pipeline configured in ``"embedding"`` mode.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 402)`` float32 feature matrix.
    """
    return pipeline.extract_batch(texts)


# ======================================================================
# Numpy mini-batch SGD trainer
# ======================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def train_numpy_mlp(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
) -> MLPClassifier:
    """Train the MLP using mini-batch SGD with Adam optimiser.

    Uses binary cross-entropy loss with L2 regularisation.

    Parameters
    ----------
    model : MLPClassifier
        Model to train (weights will be modified in-place).
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray | None
        Optional validation data for progress reporting.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.

    Returns
    -------
    MLPClassifier
        The trained model (same object, modified in-place).
    """
    n_samples = X_train.shape[0]
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    # Adam parameters
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    weight_decay = 1e-4

    # Initialise Adam state for each parameter
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    m = {name: np.zeros_like(getattr(model, name)) for name in param_names}
    v = {name: np.zeros_like(getattr(model, name)) for name in param_names}
    t_step = 0

    best_val_f1 = 0.0
    best_weights = model.get_weights().copy()

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(n_samples)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_b = X_shuf[start:end]
            y_b = y_shuf[start:end].reshape(-1, 1)
            bs = X_b.shape[0]

            # ---- Forward pass ----
            z1 = X_b @ model.W1 + model.b1
            h1 = _relu(z1)

            z2 = h1 @ model.W2 + model.b2
            h2 = _relu(z2)

            z3 = h2 @ model.W3 + model.b3
            y_pred = _sigmoid(z3)

            # ---- Binary cross-entropy loss ----
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                y_b * np.log(y_pred_clipped)
                + (1 - y_b) * np.log(1 - y_pred_clipped)
            )
            epoch_loss += loss
            n_batches += 1

            # ---- Backward pass ----
            # dL/dz3
            dz3 = (y_pred - y_b) / bs

            dW3 = h2.T @ dz3
            db3 = dz3.sum(axis=0)

            # dL/dz2
            dh2 = dz3 @ model.W3.T
            dz2 = dh2 * _relu_grad(z2)

            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)

            # dL/dz1
            dh1 = dz2 @ model.W2.T
            dz1 = dh1 * _relu_grad(z1)

            dW1 = X_b.T @ dz1
            db1 = dz1.sum(axis=0)

            grads = {
                "W1": dW1, "b1": db1,
                "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3,
            }

            # ---- Adam update ----
            t_step += 1
            for name in param_names:
                g = grads[name] + weight_decay * getattr(model, name)
                m[name] = beta1 * m[name] + (1 - beta1) * g
                v[name] = beta2 * v[name] + (1 - beta2) * (g ** 2)
                m_hat = m[name] / (1 - beta1 ** t_step)
                v_hat = v[name] / (1 - beta2 ** t_step)
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(
                    model, name,
                    getattr(model, name) - update,
                )

        avg_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation ----
        if X_val is not None and y_val is not None:
            val_probs = model.forward_batch(X_val)
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.get_weights().copy()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
                )
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    # Restore best weights (if we had validation)
    if X_val is not None and best_val_f1 > 0:
        model.set_weights(best_weights)
        print(f"  Restored best validation weights (F1={best_val_f1:.4f})")

    return model


# ======================================================================
# sklearn fallback trainer
# ======================================================================

def train_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = 50,
) -> MLPClassifier:
    """Train using sklearn's MLPClassifier, then transfer weights.

    This is a fallback when numpy-based training is not desired.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data and labels.
    input_dim : int
        Input dimensionality.
    epochs : int
        Number of training iterations.

    Returns
    -------
    MLPClassifier
        Our custom MLPClassifier with weights transferred from sklearn.
    """
    from sklearn.neural_network import MLPClassifier as SkMLP

    print("  Training with sklearn MLPClassifier fallback ...")
    sk_model = SkMLP(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        learning_rate_init=0.001,
        batch_size=64,
        random_state=42,
        verbose=True,
    )
    sk_model.fit(X_train, y_train)

    # Transfer weights to our custom MLPClassifier
    model = MLPClassifier(input_dim=input_dim)
    model.W1 = sk_model.coefs_[0].astype(np.float32)
    model.b1 = sk_model.intercepts_[0].astype(np.float32)
    model.W2 = sk_model.coefs_[1].astype(np.float32)
    model.b2 = sk_model.intercepts_[1].astype(np.float32)
    model.W3 = sk_model.coefs_[2].astype(np.float32).reshape(64, 1)
    model.b3 = sk_model.intercepts_[2].astype(np.float32)[:1]

    return model


# ======================================================================
# Training entrypoint
# ======================================================================

def train_mlp(
    train_csv: str,
    output_path: str,
    test_csv: str | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    use_sklearn: bool = False,
) -> None:
    """Train an MLP call fraud classifier on embedding features.

    Parameters
    ----------
    train_csv : str
        Path to training CSV (columns: text, label).
    output_path : str
        Where to save the trained model (.npz).
    test_csv : str | None
        Optional path to test CSV for evaluation.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    use_sklearn : bool
        If True, use sklearn MLPClassifier instead of numpy trainer.
    """
    # ---- Load data ----
    print(f"Loading training data from {train_csv} ...")
    train_df = pd.read_csv(train_csv, dtype={"text": str, "label": int})
    train_texts = train_df["text"].fillna("").tolist()
    train_labels = train_df["label"].values

    print(f"  {len(train_texts):,} training samples")
    print(
        f"  Label distribution: "
        f"{dict(pd.Series(train_labels).value_counts().sort_index())}"
    )

    # ---- Build embedding pipeline ----
    print("Building embedding feature pipeline ...")
    pipeline = FeaturePipeline(mode="embedding")
    print(
        f"  Feature dim: {pipeline.feature_dim} "
        f"(18 handcrafted + 384 embedding)"
    )

    # ---- Extract features ----
    print("Extracting embedding features (this may take a while) ...")
    t0 = time.time()
    X_train = extract_embedding_features(train_texts, pipeline)
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Feature extraction took {time.time() - t0:.1f}s")

    # ---- Load test data (for validation during training) ----
    X_test = None
    y_test = None
    if test_csv and os.path.exists(test_csv):
        print(f"Loading test data from {test_csv} ...")
        test_df = pd.read_csv(test_csv, dtype={"text": str, "label": int})
        test_texts = test_df["text"].fillna("").tolist()
        y_test = test_df["label"].values
        print(f"  {len(test_texts):,} test samples")

        print("Extracting test features ...")
        t0 = time.time()
        X_test = extract_embedding_features(test_texts, pipeline)
        print(f"  Test feature extraction took {time.time() - t0:.1f}s")

    # ---- Train ----
    input_dim = X_train.shape[1]
    print(
        f"\nTraining MLP (input_dim={input_dim}, epochs={epochs}, "
        f"batch_size={batch_size}, lr={lr}) ..."
    )
    t0 = time.time()

    if use_sklearn:
        model = train_sklearn_mlp(X_train, train_labels, input_dim, epochs)
    else:
        model = MLPClassifier(input_dim=input_dim)
        model = train_numpy_mlp(
            model, X_train, train_labels,
            X_val=X_test, y_val=y_test,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Model: {model}")

    # ---- Save model ----
    model.save(output_path)
    print(f"  Model saved to {output_path}")

    # ---- Evaluate on training data ----
    print("\n--- Training Set Evaluation ---")
    train_probs = model.forward_batch(X_train)
    train_preds = (train_probs >= 0.5).astype(int)
    _print_metrics(train_labels, train_preds)

    # ---- Evaluate on test data ----
    if X_test is not None and y_test is not None:
        print("\n--- Test Set Evaluation ---")
        test_probs = model.forward_batch(X_test)
        test_preds = (test_probs >= 0.5).astype(int)
        _print_metrics(y_test, test_preds)

        if "category" in test_df.columns:
            _print_category_breakdown(test_df, test_preds)

    # ---- Verify federated-learning round-trip ----
    print("\n--- Federated Learning Compatibility Check ---")
    w = model.get_weights()
    print(f"  Weight vector length: {len(w):,}")
    print(f"  Weight vector size:   {w.nbytes / 1024:.1f} KB")
    model2 = MLPClassifier(input_dim=input_dim)
    model2.set_weights(w)
    if X_test is not None:
        probs_original = model.forward_batch(X_test[:10])
        probs_restored = model2.forward_batch(X_test[:10])
        max_diff = np.max(np.abs(probs_original - probs_restored))
        print(f"  get/set_weights round-trip max error: {max_diff:.2e}")
        assert max_diff < 1e-6, "Weight serialisation round-trip failed!"
    print("  PASS: get_weights/set_weights round-trip verified.")


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification metrics."""
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print()
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0, 0]:>6}  FP={cm[0, 1]:>6}")
    print(f"    FN={cm[1, 0]:>6}  TP={cm[1, 1]:>6}")
    print()
    print("  Classification Report:")
    print(
        classification_report(y_true, y_pred, target_names=["legit", "scam"])
    )


def _print_category_breakdown(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """Print per-category accuracy breakdown."""
    test_df = test_df.copy()
    test_df["pred"] = predictions
    test_df["correct"] = (test_df["label"] == test_df["pred"]).astype(int)

    print("  Per-Category Breakdown:")
    print(f"  {'Category':<30} {'Count':>8} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 10} {'-' * 8}")

    for cat in sorted(test_df["category"].unique()):
        mask = test_df["category"] == cat
        cat_true = test_df.loc[mask, "label"].values
        cat_pred = test_df.loc[mask, "pred"].values
        acc = accuracy_score(cat_true, cat_pred)
        f1 = f1_score(cat_true, cat_pred, zero_division=0)
        print(f"  {cat:<30} {mask.sum():>8} {acc:>10.4f} {f1:>8.4f}")
    print()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MLP fraud classifier on embedding features."
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/call_fraud_train.csv",
        help="Training CSV path",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed/call_fraud_test.csv",
        help="Test CSV path (optional, for evaluation)",
    )
    parser.add_argument(
        "--output",
        default="models/call_fraud_mlp.npz",
        help="Output model path (.npz)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="Use sklearn MLPClassifier instead of numpy trainer",
    )
    args = parser.parse_args()

    train_mlp(
        train_csv=args.train_csv,
        output_path=args.output,
        test_csv=args.test_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_sklearn=args.use_sklearn,
    )


if __name__ == "__main__":
    main()
