"""Simulate federated learning with N devices over M rounds using a real MLP."""
import numpy as np
import json
import os
import sys
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.local_trainer import LocalTrainer, LocalTrainingBuffer
from federated.dp_injector import DPInjector
from sentinel_edge.classifier.mlp_classifier import MLPClassifier


INPUT_DIM = 402


class FederatedSimulation:
    """Simulate federated learning with a real 3-layer MLP classifier.

    Each round:
        1. Each device copies global weights, fine-tunes with real numpy backprop
        2. Computes gradient delta (local_weights - global_weights)
        3. Applies DP noise (clip + Gaussian)
        4. Hub aggregates via FedAvg weighted by n_samples
        5. Global model updated, evaluated on hold-out test set
    """

    def __init__(self, n_devices: int = 5, n_rounds: int = 10,
                 epsilon: float = 0.3, use_dp: bool = True):
        self.n_devices = n_devices
        self.n_rounds = n_rounds
        self.use_dp = use_dp
        self.devices: list = []
        self.global_model: MLPClassifier = None
        self.round_results: list = []
        self.dp_injector = DPInjector(epsilon=epsilon)

        # Hold-out test set
        self.test_features: np.ndarray = None
        self.test_labels: np.ndarray = None

    # ------------------------------------------------------------------
    # Device & data initialisation
    # ------------------------------------------------------------------

    def initialize(self):
        """Create N simulated devices with non-IID data distributions.

        Device data profiles:
            Device 0: Heavy on IRS scams (60% scam rate)
            Device 1: Heavy on tech support scams (55% scam rate)
            Device 2: Mixed scam types (50% scam rate)
            Device 3: Mostly legitimate calls (15% scam rate)
            Device 4: Heavy on bank fraud (55% scam rate)
            Device 5+: Random profile
        """
        np.random.seed(42)

        self.devices = []
        for i in range(self.n_devices):
            device = LocalTrainer(device_id=f"device_{i}", input_dim=INPUT_DIM)
            X, y = self._generate_device_data(i, n_samples=100)
            for j in range(X.shape[0]):
                device.ingest_call_data(X[j], int(y[j]))
            self.devices.append(device)

        # Balanced test set from a separate seed
        rng = np.random.RandomState(999)
        self.test_features, self.test_labels = self._generate_test_set(
            rng, n_samples=300
        )

    def _generate_device_data(self, device_idx: int,
                              n_samples: int = 100) -> tuple:
        """Generate synthetic 402-dim feature vectors for a device.

        Scam vectors: positive bias in the first half of dimensions.
        Legit vectors: negative bias in the first half.
        Each device gets different class distributions (non-IID).
        """
        rng = np.random.RandomState(42 + device_idx * 1000)

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

        X = np.zeros((n_samples, INPUT_DIM))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            is_scam = rng.random() < profile["scam_rate"]
            y[i] = 1 if is_scam else 0

            if is_scam:
                scam_type = rng.choice(
                    ["irs", "tech", "bank", "generic"],
                    p=[profile["irs"], profile["tech"],
                       profile["bank"], profile["generic"]],
                )
                X[i] = self._make_scam_vector(rng, scam_type)
            else:
                X[i] = self._make_legit_vector(rng)

        return X, y

    # ------------------------------------------------------------------
    # Synthetic feature vector generators
    # ------------------------------------------------------------------

    def _make_scam_vector(self, rng: np.random.RandomState,
                          scam_type: str) -> np.ndarray:
        """Create a 402-dim feature vector for a scam call.

        The discriminative signal is sparse: only a small subset of
        features carry class information, embedded in high-dimensional
        noise.  This makes the classification problem realistically
        difficult for federated learning with DP.
        """
        n = INPUT_DIM
        v = rng.normal(0.0, 1.0, size=n)

        # Sparse discriminative signal in a small block of features
        # Only ~20 features carry the scam signal (out of 402)
        signal_start = 0
        signal_end = 20
        v[signal_start:signal_end] += rng.normal(0.8, 0.4, size=signal_end - signal_start)

        # Scam-type-specific sub-patterns in another small block
        type_start = 20
        type_block = 10
        if scam_type == "irs":
            v[type_start:type_start + type_block] += rng.normal(0.6, 0.3, size=type_block)
        elif scam_type == "tech":
            v[type_start + type_block:type_start + 2 * type_block] += rng.normal(0.5, 0.3, size=type_block)
        elif scam_type == "bank":
            v[type_start + 2 * type_block:type_start + 3 * type_block] += rng.normal(0.7, 0.3, size=type_block)
        else:  # generic
            v[type_start + 3 * type_block:type_start + 4 * type_block] += rng.normal(0.4, 0.3, size=type_block)

        return v

    def _make_legit_vector(self, rng: np.random.RandomState) -> np.ndarray:
        """Create a 402-dim feature vector for a legitimate call.

        Negative bias in the same sparse feature block that scam
        vectors use, so the MLP must learn to separate in that subspace.
        """
        n = INPUT_DIM
        v = rng.normal(0.0, 1.0, size=n)

        # Opposite signal in the discriminative block
        signal_start = 0
        signal_end = 20
        v[signal_start:signal_end] += rng.normal(-0.8, 0.4, size=signal_end - signal_start)

        return v

    def _generate_test_set(self, rng: np.random.RandomState,
                           n_samples: int = 300) -> tuple:
        """Generate a balanced test set (50/50 scam/legit)."""
        n_half = n_samples // 2
        X = np.zeros((n_samples, INPUT_DIM))
        y = np.zeros(n_samples, dtype=int)

        scam_types = ["irs", "tech", "bank", "generic"]
        for i in range(n_half):
            stype = rng.choice(scam_types)
            X[i] = self._make_scam_vector(rng, stype)
            y[i] = 1

        for i in range(n_half, n_samples):
            X[i] = self._make_legit_vector(rng)
            y[i] = 0

        perm = rng.permutation(n_samples)
        return X[perm], y[perm]

    # ------------------------------------------------------------------
    # Global model
    # ------------------------------------------------------------------

    def initialize_global_model(self):
        """Initialize global MLPClassifier with random weights."""
        self.global_model = MLPClassifier(input_dim=INPUT_DIM)

    # ------------------------------------------------------------------
    # Federated round
    # ------------------------------------------------------------------

    def run_round(self, round_num: int) -> dict:
        """Execute one federated round.

        1. Each device fine-tunes on its local data (real backprop)
        2. Compute gradient delta
        3. Add DP noise (if enabled)
        4. Hub: FedAvg weighted by n_samples
        5. Update global model
        6. Evaluate on test set using real MLP forward pass
        """
        global_weights = self.global_model.get_weights()
        updates = []
        device_sigmas = []

        for device in self.devices:
            n_local = device.buffer.size()
            if n_local == 0:
                continue

            # Fine-tune locally using real backprop -> gradient delta
            delta = device.fine_tune(global_weights)

            if self.use_dp:
                # DP noise injection
                noised_delta, sigma, eps_round = self.dp_injector.add_noise(
                    delta, n_local
                )
                device_sigmas.append(sigma)
                updates.append((noised_delta, n_local))
            else:
                updates.append((delta, n_local))

            device.current_model_version = round_num + 1

        if len(updates) == 0:
            metrics = self._evaluate()
            metrics.update({
                "round": round_num,
                "n_devices": 0,
                "epsilon_spent": self.dp_injector.privacy_budget_spent(
                    round_num + 1
                ) if self.use_dp else 0.0,
                "avg_sigma": 0.0,
            })
            return metrics

        # FedAvg aggregation
        aggregated_delta = self._fedavg_aggregate(updates)

        # Apply aggregated update to global model (server lr = 1.0, no inflation)
        new_weights = global_weights + aggregated_delta
        self.global_model.set_weights(new_weights)

        # Evaluate
        metrics = self._evaluate()
        metrics.update({
            "round": round_num,
            "n_devices": len(updates),
            "epsilon_spent": self.dp_injector.privacy_budget_spent(
                round_num + 1
            ) if self.use_dp else 0.0,
            "avg_sigma": float(np.mean(device_sigmas)) if device_sigmas else 0.0,
        })

        # Inject fresh data each round to simulate ongoing call activity
        self._add_round_data(round_num)

        return metrics

    def _add_round_data(self, round_num: int):
        """Add new training samples each round to simulate ongoing calls."""
        extra = 30 + round_num * 10
        profiles = {0: 0.60, 1: 0.55, 2: 0.50, 3: 0.15, 4: 0.55}
        scam_types = ["irs", "tech", "bank", "generic"]

        for i, device in enumerate(self.devices):
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

    def _fedavg_aggregate(self, updates: list) -> np.ndarray:
        """FedAvg: weighted mean of gradient deltas.

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

    def _evaluate(self) -> dict:
        """Evaluate global MLP on test set.

        Uses the real MLP forward pass (not a linear classifier).
        Returns accuracy, precision, recall, F1.
        """
        X = self.test_features
        y = self.test_labels

        # Forward pass through the real MLP
        probs = self.global_model.forward(X)
        if isinstance(probs, float):
            probs = np.array([probs])
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

        dp_label = f"epsilon={self.dp_injector.epsilon}" if self.use_dp else "OFF"
        print(f"\n{'='*60}")
        print(f"SentinelEdge Federated Learning Simulation (MLP)")
        print(f"Devices: {self.n_devices} | Rounds: {self.n_rounds}")
        print(f"Differential Privacy: {dp_label}")
        print(f"MLP: {INPUT_DIM} -> 128 -> 64 -> 1")
        print(f"{'='*60}\n")

        for r in range(self.n_rounds):
            result = self.run_round(r)
            self.round_results.append(result)

            print(f"Round {r+1}/{self.n_rounds}:")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1 Score:  {result['f1']:.4f}")
            print(f"  Devices:   {result['n_devices']}")
            if self.use_dp:
                print(f"  Epsilon:   {result['epsilon_spent']:.4f}")
                print(f"  Avg sigma: {result['avg_sigma']:.6f}")
            print()

        return self.round_results


def run_dp_comparison(n_devices: int = 5, n_rounds: int = 10) -> dict:
    """Run the simulation twice: with DP and without DP.

    Returns a dict with keys 'with_dp' and 'without_dp', each containing
    the list of round results.  Used by visualization.py for comparison plots.
    """
    print("=" * 60)
    print("  RUNNING COMPARISON: WITH DP vs WITHOUT DP")
    print("=" * 60)

    # Run WITH DP
    np.random.seed(42)
    sim_dp = FederatedSimulation(
        n_devices=n_devices, n_rounds=n_rounds,
        epsilon=0.3, use_dp=True,
    )
    results_dp = sim_dp.run()

    # Run WITHOUT DP
    np.random.seed(42)
    sim_no_dp = FederatedSimulation(
        n_devices=n_devices, n_rounds=n_rounds,
        epsilon=0.3, use_dp=False,
    )
    results_no_dp = sim_no_dp.run()

    return {"with_dp": results_dp, "without_dp": results_no_dp}


def main():
    parser = argparse.ArgumentParser(
        description="Run federated learning simulation with real MLP"
    )
    parser.add_argument("--devices", type=int, default=5,
                        help="Number of simulated devices")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated rounds")
    parser.add_argument("--compare", action="store_true",
                        help="Run DP vs no-DP comparison")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.compare:
        comparison = run_dp_comparison(
            n_devices=args.devices, n_rounds=args.rounds
        )
        output_path = os.path.join(output_dir, "simulation_results.json")
        serializable = {
            "with_dp": _make_serializable(comparison["with_dp"]),
            "without_dp": _make_serializable(comparison["without_dp"]),
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved comparison results to {output_path}")

        # Also generate plots
        try:
            from federated.visualization import (
                plot_accuracy_over_rounds, plot_dp_comparison,
            )
            plot_accuracy_over_rounds(
                comparison["with_dp"],
                output_path=os.path.join(output_dir, "federated_results.png"),
            )
            plot_dp_comparison(
                comparison,
                output_path=os.path.join(output_dir, "dp_comparison.png"),
            )
        except ImportError:
            print("(Skipping plots: matplotlib not available)")
    else:
        np.random.seed(42)
        sim = FederatedSimulation(
            n_devices=args.devices, n_rounds=args.rounds
        )
        results = sim.run()

        output_path = os.path.join(output_dir, "simulation_results.json")
        serializable = _make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved results to {output_path}")

        # Generate plot
        try:
            from federated.visualization import plot_accuracy_over_rounds
            plot_accuracy_over_rounds(
                results,
                output_path=os.path.join(output_dir, "federated_results.png"),
            )
        except ImportError:
            print("(Skipping plot: matplotlib not available)")


def _make_serializable(results: list) -> list:
    """Convert numpy types to JSON-serializable Python types."""
    out = []
    for r in results:
        out.append({
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in r.items()
        })
    return out


if __name__ == "__main__":
    main()
