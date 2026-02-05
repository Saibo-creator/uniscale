"""
Analyze rank correlation between small and large model performance.

This script answers the core research question: Can small models predict
which tokenizers work best for large models?
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load all evaluation results from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        DataFrame with all results
    """
    results_path = Path(results_dir)
    all_results = []

    for result_file in results_path.glob("**/*.json"):
        with open(result_file, "r") as f:
            data = json.load(f)
            all_results.append(data)

    df = pd.DataFrame(all_results)
    return df


def compute_rank_correlation(
    df: pd.DataFrame,
    small_model_size: str,
    large_model_size: str,
    metric: str = "bits_per_byte",
) -> Tuple[float, float, float, float]:
    """
    Compute rank correlation between small and large model performance.

    Args:
        df: DataFrame with evaluation results
        small_model_size: Size of small model (e.g., "50M")
        large_model_size: Size of large model (e.g., "1B")
        metric: Metric to compare (default: "bits_per_byte")

    Returns:
        Tuple of (spearman_rho, spearman_p, kendall_tau, kendall_p)
    """
    # Filter data
    small_df = df[df["model_size"] == small_model_size].copy()
    large_df = df[df["model_size"] == large_model_size].copy()

    # Group by tokenizer and average across seeds
    small_df = small_df.groupby("tokenizer_path")[metric].mean().reset_index()
    large_df = large_df.groupby("tokenizer_path")[metric].mean().reset_index()

    # Merge on tokenizer
    merged = small_df.merge(
        large_df,
        on="tokenizer_path",
        suffixes=("_small", "_large"),
    )

    if len(merged) < 2:
        print(f"Warning: Not enough data points for correlation analysis")
        return None, None, None, None

    # Compute correlations
    spearman_rho, spearman_p = spearmanr(
        merged[f"{metric}_small"],
        merged[f"{metric}_large"],
    )

    kendall_tau, kendall_p = kendalltau(
        merged[f"{metric}_small"],
        merged[f"{metric}_large"],
    )

    return spearman_rho, spearman_p, kendall_tau, kendall_p


def plot_correlation(
    df: pd.DataFrame,
    small_model_size: str,
    large_model_size: str,
    metric: str = "bits_per_byte",
    output_file: str = None,
):
    """
    Plot correlation between small and large model performance.

    Args:
        df: DataFrame with evaluation results
        small_model_size: Size of small model
        large_model_size: Size of large model
        metric: Metric to plot
        output_file: Optional file to save plot
    """
    # Filter and aggregate data
    small_df = df[df["model_size"] == small_model_size].copy()
    large_df = df[df["model_size"] == large_model_size].copy()

    small_df = small_df.groupby("tokenizer_path")[metric].mean().reset_index()
    large_df = large_df.groupby("tokenizer_path")[metric].mean().reset_index()

    merged = small_df.merge(
        large_df,
        on="tokenizer_path",
        suffixes=("_small", "_large"),
    )

    # Compute correlation
    rho, p = spearmanr(merged[f"{metric}_small"], merged[f"{metric}_large"])

    # Create plot
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(
        merged[f"{metric}_small"],
        merged[f"{metric}_large"],
        s=100,
        alpha=0.6,
    )

    # Add labels for each point
    for _, row in merged.iterrows():
        tokenizer_name = Path(row["tokenizer_path"]).name
        plt.annotate(
            tokenizer_name,
            (row[f"{metric}_small"], row[f"{metric}_large"]),
            fontsize=8,
            alpha=0.7,
        )

    # Add diagonal line
    min_val = min(merged[f"{metric}_small"].min(), merged[f"{metric}_large"].min())
    max_val = max(merged[f"{metric}_small"].max(), merged[f"{metric}_large"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.3, label="y=x")

    # Labels and title
    plt.xlabel(f"{small_model_size} Model - {metric}", fontsize=12)
    plt.ylabel(f"{large_model_size} Model - {metric}", fontsize=12)
    plt.title(
        f"Rank Correlation: {small_model_size} vs {large_model_size}\n"
        f"Spearman ρ = {rho:.3f} (p = {p:.4f})",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rank correlation between model sizes"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation result JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="bits_per_byte",
        choices=["bits_per_byte", "perplexity_per_byte", "perplexity_per_token"],
        help="Metric to analyze",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Rank Correlation Analysis")
    print(f"{'='*60}\n")

    # Load results
    print(f"Loading results from {args.results_dir}")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} evaluation results")

    # Get unique model sizes
    model_sizes = sorted(df["model_size"].unique())
    print(f"Model sizes: {model_sizes}")

    # Compute correlations for all pairs
    results_summary = []

    for i, small_size in enumerate(model_sizes[:-1]):
        for large_size in model_sizes[i + 1 :]:
            print(f"\nAnalyzing: {small_size} vs {large_size}")

            spearman_rho, spearman_p, kendall_tau, kendall_p = compute_rank_correlation(
                df, small_size, large_size, args.metric
            )

            if spearman_rho is not None:
                print(f"  Spearman ρ: {spearman_rho:.4f} (p = {spearman_p:.4f})")
                print(f"  Kendall τ:  {kendall_tau:.4f} (p = {kendall_p:.4f})")

                # Create plot
                plot_file = output_path / f"correlation_{small_size}_vs_{large_size}.png"
                plot_correlation(df, small_size, large_size, args.metric, str(plot_file))

                results_summary.append(
                    {
                        "small_model": small_size,
                        "large_model": large_size,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                        "kendall_tau": kendall_tau,
                        "kendall_p": kendall_p,
                        "metric": args.metric,
                    }
                )

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = output_path / "correlation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")

    # Save as JSON too
    summary_json = output_path / "correlation_summary.json"
    with open(summary_json, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
