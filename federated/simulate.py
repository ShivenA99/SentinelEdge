"""Simulate federated learning with N devices over M rounds."""
import numpy as np
import json
import os
import sys
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.local_trainer import LocalTrainer, LocalTrainingBuffer
from federated.dp_injector import DPInjector


class FederatedSimulation:
    """Simulate the full federated learning process."""

    def __init__(self, n_devices: int = 5, n_rounds: int = 3, n_features: int = 518):
        self.n_devices = n_devices
        self.n_rounds = n_rounds
        self.n_features = n_features
        self.devices: list = []
        self.global_weights: np.ndarray = None
        self.round_results: list = []
        self.dp_injector = DPInjector(epsilon=0.3)

        # Hold-out test set for evaluation
        self.test_features: np.ndarray = None
        self.test_labels: np.ndarray = None

    # ------------------------------------------------------------------
    # Device & data initialisation
    # ------------------------------------------------------------------

    def initialize(self):
        """Create N simulated devices with different data distributions.

        Each device gets a different mix of scam types:
        - Device 0: Heavy on IRS scams
        - Device 1: Heavy on tech support scams
        - Device 2: Mixed scam types
        - Device 3: Mostly legitimate calls (few scams)
        - Device 4: Heavy on bank fraud

        This simulates real-world non-IID data distribution.
        Additional devices (index >= 5) get a random profile.
        """
        np.random.seed(42)

        self.devices = []
        for i in range(self.n_devices):
            device = LocalTrainer(device_id=f"device_{i}")
            X, y = self.generate_synthetic_device_data(i, n_samples=100)
            for j in range(X.shape[0]):
                device.ingest_call_data(X[j], int(y[j]))
            self.devices.append(device)

        # Build a balanced test set from a fresh seed
        rng = np.random.RandomState(999)
        self.test_features, self.test_labels = self._generate_test_set(rng, n_samples=300)

    def generate_synthetic_device_data(
        self, device_idx: int, n_samples: int = 100
    ) -> tuple:
        """Generate synthetic training data for a device.

        Creates feature vectors that simulate different scam patterns.
        Different devices see different scam distributions (non-IID).

        The feature vector layout (simplified for simulation):
          [0:128]    - Audio embedding features
          [128:256]  - Vocal pattern features
          [256:384]  - NLP / keyword features
          [384:450]  - Call metadata features
          [450:518]  - Behavioural / context features

        Scam calls are generated with distinctive patterns in certain
        feature blocks depending on scam type.  Legitimate calls have
        lower-magnitude, more uniform features.
        """
        rng = np.random.RandomState(42 + device_idx * 1000)

        # Define scam-type probability per device (IRS, tech-support, bank, generic)
        device_profiles = {
            0: {"scam_rate": 0.60, "irs": 0.70, "tech": 0.10, "bank": 0.10, "generic": 0.10},
            1: {"scam_rate": 0.55, "irs": 0.10, "tech": 0.65, "bank": 0.10, "generic": 0.15},
            2: {"scam_rate": 0.50, "irs": 0.25, "tech": 0.25, "bank": 0.25, "generic": 0.25},
            3: {"scam_rate": 0.15, "irs": 0.25, "tech": 0.25, "bank": 0.25, "generic": 0.25},
            4: {"scam_rate": 0.55, "irs": 0.05, "tech": 0.10, "bank": 0.70, "generic": 0.15},
        }
        profile = device_profiles.get(device_idx, {
            "scam_rate": rng.uniform(0.3, 0.6),
            "irs": 0.25, "tech": 0.25, "bank": 0.25, "generic": 0.25,
        })

        X = np.zeros((n_samples, self.n_features))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            is_scam = rng.random() < profile["scam_rate"]
            y[i] = 1 if is_scam else 0

            if is_scam:
                scam_type = rng.choice(
                    ["irs", "tech", "bank", "generic"],
                    p=[profile["irs"], profile["tech"], profile["bank"], profile["generic"]],
                )
                X[i] = self._make_scam_vector(rng, scam_type)
            else:
                X[i] = self._make_legit_vector(rng)

        return X, y

    # ------------------------------------------------------------------
    # Synthetic feature vector generators
    # ------------------------------------------------------------------

    def _make_scam_vector(self, rng: np.random.RandomState, scam_type: str) -> np.ndarray:
        """Create a feature vector for a scam call.

        Each scam type activates different feature blocks so the model
        can learn distinct patterns.  All indices are computed as
        fractions of n_features so this works for any dimensionality.
        """
        n = self.n_features
        v = rng.normal(0.0, 0.3, size=n)

        # Common scam signals in first quarter
        q1 = n // 4
        v[0:q1] += rng.normal(0.5, 0.2, size=q1)

        # Scam-type-specific signals in different quarters
        q2_start = q1
        q2_end = n // 2
        q3_start = q2_end
        q3_end = 3 * n // 4
        q4_start = q3_end

        if scam_type == "irs":
            sz = q2_end - q2_start
            v[q2_start:q2_end] += rng.normal(0.8, 0.2, size=sz)
        elif scam_type == "tech":
            sz = q3_end - q3_start
            v[q3_start:q3_end] += rng.normal(0.7, 0.2, size=sz)
        elif scam_type == "bank":
            sz = n - q4_start
            v[q4_start:n] += rng.normal(0.9, 0.15, size=sz)
        else:  # generic
            sz = q2_end - q2_start
            v[q2_start:q2_end] += rng.normal(0.5, 0.2, size=sz)

        # Universal scam marker: global positive bias
        v += rng.normal(0.3, 0.1, size=n)

        return v

    def _make_legit_vector(self, rng: np.random.RandomState) -> np.ndarray:
        """Create a feature vector for a legitimate call."""
        n = self.n_features
        v = rng.normal(0.0, 0.25, size=n)
        # Legitimate calls: slightly negative bias
        v += rng.normal(-0.15, 0.1, size=n)
        return v

    def _generate_test_set(self, rng: np.random.RandomState, n_samples: int = 300) -> tuple:
        """Generate a balanced test set (50/50 scam/legit)."""
        n_half = n_samples // 2
        X = np.zeros((n_samples, self.n_features))
        y = np.zeros(n_samples, dtype=int)

        scam_types = ["irs", "tech", "bank", "generic"]
        for i in range(n_half):
            stype = rng.choice(scam_types)
            X[i] = self._make_scam_vector(rng, stype)
            y[i] = 1

        for i in range(n_half, n_samples):
            X[i] = self._make_legit_vector(rng)
            y[i] = 0

        # Shuffle
        perm = rng.permutation(n_samples)
        return X[perm], y[perm]

    # ------------------------------------------------------------------
    # Global model
    # ------------------------------------------------------------------

    def initialize_global_model(self):
        """Initialize global model weights (random small values)."""
        self.global_weights = np.random.randn(self.n_features) * 0.01

    # ------------------------------------------------------------------
    # Federated round
    # ------------------------------------------------------------------

    def run_round(self, round_num: int) -> dict:
        """Execute one federated round:
        1. Each device fine-tunes on its local data
        2. Each device computes gradient delta  (done inside fine_tune)
        3. Each device adds DP noise
        4. Hub aggregates with FedAvg (weighted by n_samples)
        5. Hub validates new global model
        6. If valid, distribute to all devices
        """
        updates = []
        device_sigmas = []

        for device in self.devices:
            n_local = device.buffer.size()
            if n_local == 0:
                continue

            # Fine-tune locally -> gradient delta
            delta = device.fine_tune(self.global_weights)

            # DP noise injection
            noised_delta, sigma = self.dp_injector.add_noise(delta, n_local)
            device_sigmas.append(sigma)

            updates.append((noised_delta, n_local))
            device.current_model_version = round_num + 1

        if len(updates) == 0:
            metrics = self.evaluate_global_model(self.test_features, self.test_labels)
            metrics.update({
                "round": round_num,
                "n_devices": 0,
                "epsilon_spent": self.dp_injector.privacy_budget_spent(round_num + 1),
                "avg_sigma": 0.0,
            })
            return metrics

        # FedAvg aggregation
        aggregated_delta = self.fedavg_aggregate(updates)

        # Apply aggregated update to global weights
        # Use a server-side learning rate that increases slightly each round
        # to allow the model to converge more aggressively as noise averages out
        server_lr = 1.0 + 0.15 * round_num
        self.global_weights = self.global_weights + server_lr * aggregated_delta

        # Evaluate
        metrics = self.evaluate_global_model(self.test_features, self.test_labels)
        metrics.update({
            "round": round_num,
            "n_devices": len(updates),
            "epsilon_spent": self.dp_injector.privacy_budget_spent(round_num + 1),
            "avg_sigma": float(np.mean(device_sigmas)),
        })

        # Feed new data each round to simulate ongoing call activity
        self._add_round_data(round_num)

        return metrics

    def _add_round_data(self, round_num: int):
        """Inject additional training samples each round to simulate ongoing
        call activity.  Later rounds get slightly cleaner signal as users
        provide more feedback."""
        extra = 30 + round_num * 10  # more data each round
        profiles = {
            0: 0.60, 1: 0.55, 2: 0.50, 3: 0.15, 4: 0.55,
        }
        scam_types = ["irs", "tech", "bank", "generic"]

        for i, device in enumerate(self.devices):
            # Unique seed per device per round so new data varies each round
            rng = np.random.RandomState(42 + i * 1000 + (round_num + 1) * 500)
            scam_rate = profiles.get(i, 0.4)

            for j in range(extra):
                if rng.random() < scam_rate:
                    stype = rng.choice(scam_types)
                    vec = self._make_scam_vector(rng, stype)
                    device.ingest_call_data(vec, 1)
                else:
                    vec = self._make_legit_vector(rng)
                    device.ingest_call_data(vec, 0)

    # ------------------------------------------------------------------
    # FedAvg aggregation
    # ------------------------------------------------------------------

    def fedavg_aggregate(self, updates: list) -> np.ndarray:
        """FedAvg: weighted mean of gradient deltas.
        updates = [(noised_delta, n_samples), ...]
        G_global = sum(n_i * G_i) / sum(n_i)
        """
        total_samples = sum(n for _, n in updates)
        if total_samples == 0:
            return np.zeros_like(updates[0][0])

        weighted_sum = np.zeros_like(updates[0][0])
        for delta, n_i in updates:
            weighted_sum += n_i * delta
        return weighted_sum / total_samples

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_global_model(self, test_features: np.ndarray, test_labels: np.ndarray) -> dict:
        """Evaluate current global model on test set.
        Return accuracy, precision, recall, F1.
        Uses a simple linear classifier (logistic regression) with current
        global weights.
        """
        X = test_features
        y = test_labels

        # Truncate/pad features to match weight dimension
        if X.shape[1] < self.n_features:
            pad = np.zeros((X.shape[0], self.n_features - X.shape[1]))
            X = np.hstack([X, pad])
        elif X.shape[1] > self.n_features:
            X = X[:, :self.n_features]

        logits = X @ self.global_weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        preds = (probs >= 0.5).astype(int)

        tp = int(np.sum((preds == 1) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> list:
        """Run full simulation: initialize + all rounds."""
        self.initialize()
        self.initialize_global_model()

        print(f"\n{'='*60}")
        print(f"SentinelEdge Federated Learning Simulation")
        print(f"Devices: {self.n_devices} | Rounds: {self.n_rounds}")
        print(f"Privacy: epsilon={self.dp_injector.epsilon}")
        print(f"{'='*60}\n")

        for r in range(self.n_rounds):
            result = self.run_round(r)
            self.round_results.append(result)

            print(f"Round {r+1}/{self.n_rounds}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1 Score: {result['f1']:.4f}")
            print(f"  Contributing devices: {result['n_devices']}")
            print(f"  Privacy budget spent: {result['epsilon_spent']:.4f}")
            print()

        # Save results
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results.json")
        serializable = []
        for r in self.round_results:
            serializable.append({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()})
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        return self.round_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning simulation")
    parser.add_argument("--devices", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    sim = FederatedSimulation(n_devices=args.devices, n_rounds=args.rounds)
    results = sim.run()
