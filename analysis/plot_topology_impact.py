"""
Topology Impact Comparison Plot

Generates grouped bar charts comparing F1 (or other metric) across topologies
(Erdos-Renyi, Dense, Ring) for Group C experiments.

Layout: 1x3 subplots (one per attack: No Attack, Gaussian, Trim-Mean Attack),
each showing aggregator labels on X-axis with bars grouped by topology.

Usage:
    python -m analysis.plot_topology_impact --root-dir /path/to/results
    python -m analysis.plot_topology_impact --root-dir /path/to/results --metric Accuracy --format png
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

TOPOLOGY_COLORS = {"Erdos-Renyi": "#1f77b4", "Dense": "#ff7f0e", "Ring": "#2ca02c"}
TOPOLOGY_ORDER = ["Erdos-Renyi", "Dense", "Ring"]


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Topology Impact Comparison Plot (Group C)")
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

    # Group C attacks (the three used in topology comparison)
    group_c_attacks = ["No Attack", "Gaussian", "Trim-Mean Attack"]

    # Collect data: {attack_label: {agg_label: {topo_label: (mean, std)}}}
    data: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}

    for sc in all_scenarios:
        attack_label = sc["attack_label"]
        agg_label = sc["aggregator_label"]
        topo_label = sc["topology_label"]

        if attack_label not in group_c_attacks:
            continue

        # Load results
        _, results_df = load_scenario_results(sc["scenario_dir"])
        mean, std = get_final_metric(results_df, metric)
        if mean is None:
            continue

        data.setdefault(attack_label, {}).setdefault(agg_label, {})[topo_label] = (mean, std)

    # Check if we have topology variation (at least one attack with multiple topologies)
    has_topology_variation = any(any(len(topos) > 1 for topos in aggs.values()) for aggs in data.values())

    if not has_topology_variation:
        logger.warning("No topology variation data found. Generating placeholder plot.")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        for idx, attack in enumerate(group_c_attacks):
            ax = axes[0][idx]
            ax.set_title(attack)
            ax.set_ylabel(f"Final {metric}")
            ax.text(
                0.5,
                0.5,
                "No topology variation data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                fontstyle="italic",
            )
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"topology_impact_{metric.lower()}.{fmt}")
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved placeholder to {out_path}")
        return

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    for idx, attack_label in enumerate(group_c_attacks):
        ax = axes[0][idx]
        attack_data = data.get(attack_label, {})

        # Determine which aggregators are present for this attack
        present_aggs = [a for a in AGG_ORDER if a in attack_data]

        if not present_aggs:
            ax.set_title(attack_label)
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

        # Determine which topologies are present across all aggregators for this attack
        present_topos = []
        for topo in TOPOLOGY_ORDER:
            if any(topo in attack_data.get(agg, {}) for agg in present_aggs):
                present_topos.append(topo)

        n_aggs = len(present_aggs)
        n_topos = len(present_topos)
        bar_width = 0.25
        x = np.arange(n_aggs)

        for t_idx, topo in enumerate(present_topos):
            means = []
            stds = []
            for agg in present_aggs:
                val = attack_data.get(agg, {}).get(topo)
                if val is not None:
                    means.append(val[0])
                    stds.append(val[1])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (t_idx - (n_topos - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=TOPOLOGY_COLORS[topo],
                edgecolor="black",
                linewidth=0.5,
                label=topo if idx == 0 else None,
            )

        ax.set_title(attack_label)
        ax.set_ylabel(f"Final {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(present_aggs, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(TOPOLOGY_ORDER), bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, f"topology_impact_{metric.lower()}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved topology impact plot to {out_path}")


if __name__ == "__main__":
    main()
