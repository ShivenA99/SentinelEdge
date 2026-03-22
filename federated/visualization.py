"""Visualize federated learning results."""
import json
import os


def plot_accuracy_over_rounds(results: list, output_path: str = "federated_results.png"):
    """Plot accuracy, F1, precision, recall over federated rounds.

    Shows:
    - Line chart: accuracy, F1, precision, recall per round
    - Bar chart: privacy budget consumed
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rounds = [r["round"] + 1 for r in results]
    accuracies = [r["accuracy"] for r in results]
    f1_scores = [r["f1"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    # Plot 1: Metrics over rounds
    ax1.plot(rounds, accuracies, "o-", color="#0D9488", linewidth=2,
             markersize=6, label="Accuracy")
    ax1.plot(rounds, f1_scores, "s-", color="#F59E0B", linewidth=2,
             markersize=6, label="F1 Score")
    ax1.plot(rounds, precisions, "^-", color="#3B82F6", linewidth=2,
             markersize=5, label="Precision")
    ax1.plot(rounds, recalls, "v-", color="#8B5CF6", linewidth=2,
             markersize=5, label="Recall")

    # Annotate first and last accuracy
    ax1.annotate(
        f"{accuracies[0]:.2f}",
        xy=(rounds[0], accuracies[0]),
        xytext=(rounds[0] + 0.3, accuracies[0] - 0.04),
        fontsize=9, color="#0D9488",
    )
    ax1.annotate(
        f"{accuracies[-1]:.2f}",
        xy=(rounds[-1], accuracies[-1]),
        xytext=(rounds[-1] - 0.8, accuracies[-1] + 0.03),
        fontsize=9, color="#0D9488",
    )

    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Score")
    ax1.set_title("MLP Model Improvement Over Federated Rounds")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(rounds)

    # Plot 2: Privacy budget over rounds
    epsilons = [r.get("epsilon_spent", 0) for r in results]
    if any(e > 0 for e in epsilons):
        bars = ax2.bar(rounds, epsilons, color="#EF4444", alpha=0.7)
        ax2.set_ylabel("Cumulative Privacy Budget (\u03b5)")
        for bar, eps in zip(bars, epsilons):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{eps:.2f}",
                ha="center", va="bottom", fontsize=8,
            )
    else:
        ax2.text(0.5, 0.5, "No DP applied", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14, color="gray")

    ax2.set_xlabel("Federated Round")
    ax2.set_title("Privacy Budget Consumption")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_dp_comparison(comparison: dict, output_path: str = "dp_comparison.png"):
    """Plot accuracy with DP vs without DP side by side.

    Args:
        comparison: dict with keys 'with_dp' and 'without_dp',
                    each a list of round result dicts.
        output_path: Where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dp = comparison["with_dp"]
    results_no_dp = comparison["without_dp"]

    rounds_dp = [r["round"] + 1 for r in results_dp]
    rounds_no_dp = [r["round"] + 1 for r in results_no_dp]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy comparison
    ax = axes[0]
    ax.plot(rounds_dp, [r["accuracy"] for r in results_dp],
            "o-", color="#EF4444", linewidth=2, markersize=6,
            label="With DP (\u03b5=0.3)")
    ax.plot(rounds_no_dp, [r["accuracy"] for r in results_no_dp],
            "s-", color="#0D9488", linewidth=2, markersize=6,
            label="Without DP")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: DP vs No-DP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(rounds_dp)

    # Annotate final values
    acc_dp_final = results_dp[-1]["accuracy"]
    acc_no_dp_final = results_no_dp[-1]["accuracy"]
    ax.annotate(f"{acc_dp_final:.3f}", xy=(rounds_dp[-1], acc_dp_final),
                xytext=(rounds_dp[-1] - 1.5, acc_dp_final - 0.05),
                fontsize=9, color="#EF4444")
    ax.annotate(f"{acc_no_dp_final:.3f}", xy=(rounds_no_dp[-1], acc_no_dp_final),
                xytext=(rounds_no_dp[-1] - 1.5, acc_no_dp_final + 0.03),
                fontsize=9, color="#0D9488")

    # Plot 2: F1 comparison
    ax = axes[1]
    ax.plot(rounds_dp, [r["f1"] for r in results_dp],
            "o-", color="#EF4444", linewidth=2, markersize=6,
            label="With DP (\u03b5=0.3)")
    ax.plot(rounds_no_dp, [r["f1"] for r in results_no_dp],
            "s-", color="#0D9488", linewidth=2, markersize=6,
            label="Without DP")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score: DP vs No-DP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(rounds_dp)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved DP comparison plot to {output_path}")
    plt.close()


def print_summary(results: list):
    """Print a text summary of simulation results."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING SIMULATION SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"\nRound {r['round']+1}:")
        print(f"  Accuracy:   {r['accuracy']:.4f}")
        print(f"  F1 Score:   {r['f1']:.4f}")
        print(f"  Precision:  {r.get('precision', 0):.4f}")
        print(f"  Recall:     {r.get('recall', 0):.4f}")
        print(f"  Devices:    {r['n_devices']}")
        eps = r.get('epsilon_spent', 0)
        if eps > 0:
            print(f"  \u03b5 spent:    {eps:.4f}")

    first = results[0]
    last = results[-1]
    acc_delta = last["accuracy"] - first["accuracy"]
    f1_delta = last["f1"] - first["f1"]
    print(
        f"\nImprovement: accuracy {first['accuracy']:.4f} -> "
        f"{last['accuracy']:.4f} ({acc_delta:+.4f})"
    )
    print(
        f"             F1      {first['f1']:.4f} -> "
        f"{last['f1']:.4f} ({f1_delta:+.4f})"
    )


if __name__ == "__main__":
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)

        # Handle both formats: list (single run) or dict (comparison)
        if isinstance(data, list):
            results = data
            print_summary(results)
            plot_accuracy_over_rounds(
                results,
                output_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "federated_results.png",
                ),
            )
        elif isinstance(data, dict):
            # Comparison format
            if "with_dp" in data:
                print("\n--- WITH DP ---")
                print_summary(data["with_dp"])
                print("\n--- WITHOUT DP ---")
                print_summary(data["without_dp"])
                plot_dp_comparison(
                    data,
                    output_path=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "dp_comparison.png",
                    ),
                )
            else:
                print("Unknown results format.")
    else:
        print("No simulation results found. Run simulate.py first.")
