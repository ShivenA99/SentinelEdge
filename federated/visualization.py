"""Visualize federated learning results."""
import json
import os


def plot_accuracy_over_rounds(results: list, output_path: str = "federated_results.png"):
    """Plot accuracy and F1 improvement over federated rounds.
    Uses matplotlib. Shows:
    - Line chart: accuracy and F1 per round
    - Bar chart: privacy budget consumed
    - Annotation: privacy budget on the accuracy plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy/F1 over rounds
    rounds = [r["round"] + 1 for r in results]
    accuracies = [r["accuracy"] for r in results]
    f1_scores = [r["f1"] for r in results]

    ax1.plot(
        rounds, accuracies, "o-",
        color="#0D9488", linewidth=2, markersize=8, label="Accuracy",
    )
    ax1.plot(
        rounds, f1_scores, "s-",
        color="#F59E0B", linewidth=2, markersize=8, label="F1 Score",
    )

    # Annotate first and last accuracy
    ax1.annotate(
        f"{accuracies[0]:.2f}",
        xy=(rounds[0], accuracies[0]),
        xytext=(rounds[0] + 0.2, accuracies[0] - 0.04),
        fontsize=9, color="#0D9488",
    )
    ax1.annotate(
        f"{accuracies[-1]:.2f}",
        xy=(rounds[-1], accuracies[-1]),
        xytext=(rounds[-1] - 0.5, accuracies[-1] + 0.02),
        fontsize=9, color="#0D9488",
    )

    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Score")
    ax1.set_title("Model Improvement Over Federated Rounds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xticks(rounds)

    # Plot 2: Privacy budget over rounds
    epsilons = [r["epsilon_spent"] for r in results]
    bars = ax2.bar(rounds, epsilons, color="#EF4444", alpha=0.7)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Cumulative Privacy Budget (\u03b5)")
    ax2.set_title("Privacy Budget Consumption")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)

    # Label each bar
    for bar, eps in zip(bars, epsilons):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{eps:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
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
        print(f"  \u03b5 spent:    {r['epsilon_spent']:.4f}")

    first = results[0]
    last = results[-1]
    acc_delta = last["accuracy"] - first["accuracy"]
    f1_delta = last["f1"] - first["f1"]
    print(
        f"\nImprovement: accuracy {first['accuracy']:.4f} -> "
        f"{last['accuracy']:.4f} (+{acc_delta:.4f})"
    )
    print(
        f"             F1      {first['f1']:.4f} -> "
        f"{last['f1']:.4f} (+{f1_delta:.4f})"
    )


if __name__ == "__main__":
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulation_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print_summary(results)
        plot_accuracy_over_rounds(
            results,
            output_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "federated_results.png"
            ),
        )
    else:
        print("No simulation results found. Run simulate.py first.")
