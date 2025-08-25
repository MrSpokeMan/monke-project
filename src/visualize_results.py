import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import DEFAULT_RETRIEVAL_COMPARISON_FILE, load_json


def load_retrieval_results() -> dict[int, dict]:
    results = {}
    data_dir = Path("data")

    top_k_values = [3, 5, 10, 15]

    for top_k in top_k_values:
        file_path = data_dir / f"retrieval_comparison_top_{top_k}.json"
        if file_path.exists():
            try:
                data = load_json(str(file_path))
                results[top_k] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return results


def create_accuracy_plot(results: dict[int, dict]) -> None:
    """Create a single plot showing accuracy vs top_k for both with and without reranker."""
    if not results:
        print("No results found. Please run the retrieval comparison first.")
        return

    # Extract data
    top_k_values = sorted(results.keys())

    # Data for with_reranker
    accuracy_with = [results[k]["accuracy"]["with_reranker"] for k in top_k_values]

    # Data for without_reranker
    accuracy_without = [results[k]["accuracy"]["without_reranker"] for k in top_k_values]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot both lines
    ax.plot(
        top_k_values,
        accuracy_with,
        color="tab:blue",
        marker="o",
        linewidth=2,
        label="With Reranker",
    )
    ax.plot(
        top_k_values,
        accuracy_without,
        color="tab:red",
        marker="s",
        linewidth=2,
        label="Without Reranker",
    )

    ax.set_xlabel("Top K", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Retrieval Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set x-axis to show only the actual top_k values
    ax.set_xticks(top_k_values)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("retrieval_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n=== Retrieval Accuracy Summary ===")
    print(f"Top K values tested: {top_k_values}")
    print(f"\nWith Reranker:")
    print(f"  Accuracy range: {min(accuracy_with):.3f} - {max(accuracy_with):.3f}")
    print(f"\nWithout Reranker:")
    print(f"  Accuracy range: {min(accuracy_without):.3f} - {max(accuracy_without):.3f}")


def main():
    """Main function to create visualizations."""
    print("Loading retrieval comparison results...")
    results = load_retrieval_results()

    if not results:
        print("No retrieval comparison results found.")
        print("Please run the main script first to generate comparison data.")
        return

    print(f"Found results for top_k values: {sorted(results.keys())}")

    # Create the accuracy plot
    create_accuracy_plot(results)

    print("\nVisualization completed! Plot saved as:")
    print("- retrieval_accuracy_comparison.png")


if __name__ == "__main__":
    main()
