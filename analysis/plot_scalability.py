"""
Scalability Plot

Generates line plots of F1 (or other metric) vs number of nodes for Group E
scalability sweep experiments.

Layout: 1x2 subplots (No Attack, Gaussian), with lines per aggregator showing
how performance changes as node count increases (5, 10, 20, 50).

Usage:
    python -m analysis.plot_scalability --root-dir /path/to/results
    python -m analysis.plot_scalability --root-dir /path/to/results --metric Accuracy --format png
"""

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.scenario_metadata import (
    get_final_metric,
    load_scenario_results,
    scan_scenario_root,
)

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Color / marker / order conventions
# ============================================================================

AGG_COLORS = {
    "FedAvg": "#1f77b4",
    "Krum": "#ff7f0e",
    "TrimmedMean": "#2ca02c",
    "Median": "#d62728",
    "RADAR-Agg": "#9467bd",
}
AGG_MARKERS = {"FedAvg": "o", "Krum": "s", "TrimmedMean": "^", "Median": "D", "RADAR-Agg": "*"}
AGG_ORDER = ["FedAvg", "Krum", "TrimmedMean", "Median", "RADAR-Agg"]

ATTACK_ORDER = ["No Attack", "Gaussian", "Noise Injection", "Weight Swapping", "Krum Attack", "Trim-Mean Attack"]

# Default node count for core experiments (Group A) that don't have n_nodes in metadata
DEFAULT_N_NODES = 5


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Scalability Plot (Group E)")
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario folders")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument("--metric", default="F1", help="Metric to plot (default: F1)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format (default: pdf)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "comparison_figures")
    os.makedirs(output_dir, exist_ok=True)

    metric = args.metric
    fmt = args.format

    # ── Scan scenarios ──────────────────────────────────────────────────────
    all_scenarios = scan_scenario_root(args.root_dir)

    # Scalability subplots: No Attack and Gaussian
    scalability_attacks = ["No Attack", "Gaussian"]

    # Collect data: {attack_label: {agg_label: {n_nodes: (mean, std)}}}
    data: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}

    for sc in all_scenarios:
        attack_label = sc["attack_label"]
        agg_label = sc["aggregator_label"]

        if attack_label not in scalability_attacks:
            continue

        # Determine node count: explicit n_nodes key or default
        n_nodes = sc.get("n_nodes", DEFAULT_N_NODES)

        # Skip scenarios with parameter sweeps (gamma/kappa/alpha) — those are Group D
        if any(k in sc for k in ("gamma", "kappa", "alpha")):
            continue

        # Load results
        _, results_df = load_scenario_results(sc["scenario_dir"])
        mean, std = get_final_metric(results_df, metric)
        if mean is None:
            continue

        data.setdefault(attack_label, {}).setdefault(agg_label, {})[n_nodes] = (mean, std)

    # Check if we have any scalability variation (more than one node count for any aggregator)
    has_scalability_data = any(any(len(nodes) > 1 for nodes in aggs.values()) for aggs in data.values())

    if not data:
        logger.warning("No scalability data found. Generating placeholder plot.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
        for idx, attack in enumerate(scalability_attacks):
            ax = axes[0][idx]
            ax.set_title(attack)
            ax.set_xlabel("Number of Nodes")
            ax.set_ylabel(f"Final {metric}")
            ax.text(
                0.5,
                0.5,
                "No scalability data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                fontstyle="italic",
            )
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"scalability_{metric.lower()}.{fmt}")
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved placeholder to {out_path}")
        return

    if not has_scalability_data:
        logger.warning(
            "Only single node-count data found (no scalability sweep). "
            "Plot will show single data points per aggregator."
        )

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

    for idx, attack_label in enumerate(scalability_attacks):
        ax = axes[0][idx]
        attack_data = data.get(attack_label, {})

        if not attack_data:
            ax.set_title(attack_label)
            ax.set_xlabel("Number of Nodes")
            ax.set_ylabel(f"Final {metric}")
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                fontstyle="italic",
            )
            ax.grid(True, alpha=0.3)
            continue

        for agg_label in AGG_ORDER:
            if agg_label not in attack_data:
                continue

            node_data = attack_data[agg_label]
            # Sort by node count
            sorted_nodes = sorted(node_data.keys())
            x_vals = np.array(sorted_nodes)
            means = np.array([node_data[n][0] for n in sorted_nodes])
            stds = np.array([node_data[n][1] for n in sorted_nodes])

            ax.errorbar(
                x_vals,
                means,
                yerr=stds,
                color=AGG_COLORS[agg_label],
                marker=AGG_MARKERS[agg_label],
                markersize=6,
                capsize=3,
                linewidth=1.5,
                label=agg_label,
            )

        ax.set_title(attack_label)
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(f"Final {metric}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Set x-ticks to actual node counts present across all aggregators
        all_node_counts = sorted({n for agg_data in attack_data.values() for n in agg_data})
        ax.set_xticks(all_node_counts)
        ax.set_xticklabels([str(n) for n in all_node_counts])

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"scalability_{metric.lower()}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved scalability plot to {out_path}")


if __name__ == "__main__":
    main()
