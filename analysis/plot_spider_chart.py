"""
Spider (Radar) Chart: Aggregator comparison across multiple metrics

Generates a multi-panel spider/radar chart.  Each subplot corresponds to one
attack scenario, with 5 aggregator polygons overlaid on 5 metric axes
(F1, Accuracy, Precision, Recall, 1-Loss).

Restricted to the core experiment group (Erdos-Renyi topology, no parameter
or scalability sweeps).

Usage:
    python analysis/plot_spider_chart.py --root-dir /path/to/scenarios
    python analysis/plot_spider_chart.py --root-dir /path/to/scenarios --attack "Gaussian" --format png
"""

import argparse
import logging
import os
import sys

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

# Canonical display orders and styling
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

# Metrics for spider axes
SPIDER_METRICS = ["F1", "Accuracy", "Precision", "Recall", "Loss"]
SPIDER_LABELS = ["F1", "Accuracy", "Precision", "Recall", "1 - Loss"]


def _is_core_scenario(meta: dict) -> bool:
    """Return True if the scenario belongs to the core group (A)."""
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


def _normalize_value(metric: str, value: float) -> float:
    """Normalize a metric value to [0, 1].  Loss is inverted (1 - loss)."""
    if metric == "Loss":
        # Invert so higher = better; clamp to [0, 1]
        return max(0.0, min(1.0, 1.0 - value))
    # F1, Accuracy, Precision, Recall are already in [0, 1]
    return max(0.0, min(1.0, value))


def collect_spider_data(
    scenarios: list[dict],
) -> dict[str, dict[str, list[float]]]:
    """Collect normalized metric values per attack per aggregator.

    Returns:
        {attack_label: {agg_label: [v_f1, v_acc, v_prec, v_rec, v_1minloss]}}
    """
    data: dict[str, dict[str, list[float]]] = {}

    for meta in scenarios:
        agg_label = meta["aggregator_label"]
        atk_label = meta["attack_label"]

        if agg_label not in AGG_ORDER or atk_label not in ATTACK_ORDER:
            continue

        _, results_df = load_scenario_results(meta["scenario_dir"])
        if results_df is None:
            continue

        values = []
        for metric in SPIDER_METRICS:
            mean, _ = get_final_metric(results_df, metric)
            if mean is None:
                values.append(0.0)
            else:
                values.append(_normalize_value(metric, mean))

        if atk_label not in data:
            data[atk_label] = {}
        data[atk_label][agg_label] = values

    return data


def plot_spider(
    data: dict[str, dict[str, list[float]]],
    attack_filter: str | None,
    output_path: str,
):
    """Render and save the spider chart figure."""
    # Determine which attacks to plot
    if attack_filter:
        attacks_to_plot = [a for a in ATTACK_ORDER if a == attack_filter and a in data]
        if not attacks_to_plot:
            logger.error(f"Attack '{attack_filter}' not found in data. Available: {list(data.keys())}")
            sys.exit(1)
    else:
        attacks_to_plot = [a for a in ATTACK_ORDER if a in data]

    n_plots = len(attacks_to_plot)
    if n_plots == 0:
        logger.error("No attack scenarios with data to plot.")
        sys.exit(1)

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows + 0.8))

    # Compute angles for the polygon
    n_metrics = len(SPIDER_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    legend_handles = []

    for idx, attack in enumerate(attacks_to_plot):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, polar=True)
        ax.set_title(attack, pad=15, fontsize=12, fontweight="bold")

        # Draw each aggregator polygon
        agg_data = data[attack]
        for agg in AGG_ORDER:
            if agg not in agg_data:
                continue

            values = agg_data[agg]
            values_closed = values + values[:1]  # close the polygon
            color = AGG_COLORS[agg]

            (line,) = ax.plot(
                angles, values_closed, color=color, linewidth=1.5, marker=AGG_MARKERS[agg], markersize=4, label=agg
            )
            ax.fill(angles, values_closed, color=color, alpha=0.1)

            # Collect legend handle from first subplot only
            if idx == 0:
                legend_handles.append(line)

        # Configure axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(SPIDER_LABELS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    total_slots = n_rows * n_cols
    for idx in range(n_plots, total_slots):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_visible(False)

    # Shared legend at bottom
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved spider chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spider/radar chart comparing aggregators across metrics",
    )
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario subdirectories")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument(
        "--attack", default=None, help="Plot a single attack only (default: all attacks, one subplot each)"
    )
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure format (default: pdf)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "comparison_figures")
    os.makedirs(output_dir, exist_ok=True)

    # Scan and filter to core group
    all_scenarios = scan_scenario_root(args.root_dir)
    core = [s for s in all_scenarios if _is_core_scenario(s)]
    logger.info(f"Core scenarios (Erdos-Renyi, no sweeps): {len(core)}")

    if not core:
        logger.error("No core-group scenarios found. Check --root-dir and directory naming.")
        sys.exit(1)

    # Collect data
    data = collect_spider_data(core)
    logger.info(f"Attacks with data: {list(data.keys())}")

    # Plot
    fig_path = os.path.join(output_dir, f"spider_chart.{args.format}")
    plot_spider(data, attack_filter=args.attack, output_path=fig_path)

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
