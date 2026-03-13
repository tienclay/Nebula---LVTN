"""
Per-Node Metric Distribution Plot

Generates violin plots with embedded box plots showing per-node metric
distributions for each aggregator, faceted by attack scenario.

Uses per-participant TensorBoard event files to extract individual node
values (benign only), providing visibility into intra-scenario variance.

Usage:
    python analysis/plot_node_distribution.py --root-dir /path/to/scenarios
    python analysis/plot_node_distribution.py --root-dir /path/to/scenarios --metric Accuracy --format png
"""

import argparse
import glob
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
# Shared constants
# ============================================================================

AGG_COLORS = {
    "FedAvg": "#1f77b4",
    "Krum": "#ff7f0e",
    "TrimmedMean": "#2ca02c",
    "Median": "#d62728",
    "RADAR-Agg": "#9467bd",
}
AGG_ORDER = ["FedAvg", "Krum", "TrimmedMean", "Median", "RADAR-Agg"]
ATTACK_ORDER = ["No Attack", "Gaussian", "Noise Injection", "Weight Swapping", "Krum Attack", "Trim-Mean Attack"]

# Metrics bounded in [0, 1]
_BOUNDED_01 = {"F1", "Accuracy", "Precision", "Recall"}


# ============================================================================
# Per-node metric extraction
# ============================================================================


def _discover_n_participants(metrics_dir: str) -> int:
    """Count participant_* directories under metrics_dir."""
    dirs = glob.glob(os.path.join(metrics_dir, "participant_*"))
    return len(dirs)


def extract_per_node_final_metric(scenario_dir: str, metric: str = "F1") -> list[float]:
    """Extract per-node final metric values for benign nodes in a scenario.

    Returns a list of final-round metric values (one per benign node).
    """
    from analysis.radar_metrics_aggregator import ALL_METRIC_TAGS, find_malicious_participants, read_participant_scalars

    metrics_dir = os.path.join(scenario_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        return []

    n_total = _discover_n_participants(metrics_dir)
    if n_total == 0:
        return []

    malicious = find_malicious_participants(scenario_dir)
    tb_tag = ALL_METRIC_TAGS.get(metric, f"Decentralized/{metric}")

    per_node_values = []
    for idx in range(n_total):
        if idx in malicious:
            continue
        try:
            scalars = read_participant_scalars(metrics_dir, idx)
        except Exception as e:
            logger.debug(f"Could not read participant {idx}: {e}")
            continue
        if scalars.get(tb_tag):
            # Take the last step's value
            per_node_values.append(scalars[tb_tag][-1][1])

    return per_node_values


# ============================================================================
# Core group filter
# ============================================================================


def _is_core_scenario(meta: dict) -> bool:
    """Return True if scenario belongs to the core group (A).

    Core group: topology=erdosrenyi, no parameter/scalability sweep tokens.
    """
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


# ============================================================================
# Main plotting
# ============================================================================


def plot_node_distribution(root_dir: str, output_dir: str, metric: str = "F1", fmt: str = "pdf"):  # noqa: C901
    """Generate violin + box plot of per-node metric distribution."""
    from analysis.scenario_metadata import scan_scenario_root

    os.makedirs(output_dir, exist_ok=True)

    # Scan and filter to core group
    all_scenarios = scan_scenario_root(root_dir)
    core = [s for s in all_scenarios if _is_core_scenario(s)]
    logger.info(f"Found {len(core)} core scenarios (out of {len(all_scenarios)} total)")

    if not core:
        logger.error("No core scenarios found. Check --root-dir path and naming convention.")
        sys.exit(1)

    # Collect per-node values: {attack_label: {agg_label: [values]}}
    data: dict[str, dict[str, list[float]]] = {}
    for meta in core:
        attack_label = meta["attack_label"]
        agg_label = meta["aggregator_label"]

        if attack_label not in ATTACK_ORDER:
            logger.debug(f"Skipping unknown attack: {attack_label}")
            continue

        values = extract_per_node_final_metric(meta["scenario_dir"], metric)
        logger.info(f"  {meta['scenario_name']}: {len(values)} benign node values")

        if attack_label not in data:
            data[attack_label] = {}
        # Accumulate (in case multiple scenarios map to the same cell)
        if agg_label not in data[attack_label]:
            data[attack_label][agg_label] = []
        data[attack_label][agg_label].extend(values)

    # Layout: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)

    for subplot_idx, attack_label in enumerate(ATTACK_ORDER):
        row, col = divmod(subplot_idx, 3)
        ax = axes[row][col]

        attack_data = data.get(attack_label, {})

        # Prepare data for each aggregator in order
        violin_data = []
        positions = []
        colors = []
        tick_labels = []
        has_any_data = False

        for agg_idx, agg_label in enumerate(AGG_ORDER):
            values = attack_data.get(agg_label, [])
            if values:
                violin_data.append(values)
                has_any_data = True
            else:
                # Placeholder empty list (will skip violin for this position)
                violin_data.append([])
            positions.append(agg_idx)
            colors.append(AGG_COLORS.get(agg_label, "#888888"))
            tick_labels.append(agg_label)

        if not has_any_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                fontstyle="italic",
            )
            ax.set_title(attack_label)
            ax.set_xticks(range(len(AGG_ORDER)))
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
            if metric in _BOUNDED_01:
                ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel(metric)
            continue

        # Draw violins for positions that have data
        non_empty_positions = []
        non_empty_data = []
        non_empty_colors = []
        for i, vals in enumerate(violin_data):
            if len(vals) >= 2:
                non_empty_positions.append(positions[i])
                non_empty_data.append(vals)
                non_empty_colors.append(colors[i])

        if non_empty_data:
            parts = ax.violinplot(
                non_empty_data,
                positions=non_empty_positions,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )
            # Color each violin body
            for body_idx, body in enumerate(parts["bodies"]):
                body.set_facecolor(non_empty_colors[body_idx])
                body.set_edgecolor("black")
                body.set_linewidth(0.5)
                body.set_alpha(0.4)

        # Overlay box plots for all positions that have data
        bp_positions = []
        bp_data = []
        for i, vals in enumerate(violin_data):
            if vals:
                bp_positions.append(positions[i])
                bp_data.append(vals)

        if bp_data:
            bp = ax.boxplot(
                bp_data,
                positions=bp_positions,
                widths=0.3,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
            )
            # Color each box
            bp_color_idx = 0
            for i, vals in enumerate(violin_data):
                if vals:
                    bp["boxes"][bp_color_idx].set_facecolor(colors[i])
                    bp["boxes"][bp_color_idx].set_alpha(0.6)
                    bp_color_idx += 1

        # Overlay individual data points (jittered)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(violin_data):
            if vals:
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    positions[i] + jitter,
                    vals,
                    c=colors[i],
                    s=12,
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=5,
                )

        ax.set_title(attack_label)
        ax.set_xticks(range(len(AGG_ORDER)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        if metric in _BOUNDED_01:
            ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Per-Node {metric} Distribution", fontsize=15, y=1.01)
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"node_distribution_{metric.lower()}.{fmt}")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved node distribution plot to {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Per-Node Metric Distribution Plot")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing scenario subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures",
        help="Output directory for figures (default: analysis/figures)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="F1",
        help="Metric to plot (default: F1). Must match a key in ALL_METRIC_TAGS.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    args = parser.parse_args()

    plot_node_distribution(args.root_dir, args.output_dir, args.metric, args.format)


if __name__ == "__main__":
    main()
