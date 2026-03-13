"""
Malicious Ratio Impact Plot

Generates a 1x2 figure showing how final F1 (or other metric) degrades as the
percentage of malicious nodes increases.  One subplot per attack type used in
the Group B malicious-ratio sweep (Gaussian, Trim-Mean Attack).

The 0% malicious baseline comes from `{agg}_noattack_erdosrenyi_0pct` scenarios
(shared across all attacks).

Usage:
    python -m analysis.plot_malicious_ratio --root-dir /path/to/experiments
    python -m analysis.plot_malicious_ratio --root-dir /path/to/experiments --metric Accuracy
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
# Consistent styling
# ============================================================================

AGG_COLORS = {
    "FedAvg": "#1f77b4",
    "Krum": "#ff7f0e",
    "TrimmedMean": "#2ca02c",
    "Median": "#d62728",
    "RADAR-Agg": "#9467bd",
}

AGG_MARKERS = {
    "FedAvg": "o",
    "Krum": "s",
    "TrimmedMean": "^",
    "Median": "D",
    "RADAR-Agg": "*",
}

AGG_ORDER = ["FedAvg", "Krum", "TrimmedMean", "Median", "RADAR-Agg"]

ATTACK_ORDER = [
    "No Attack",
    "Gaussian",
    "Noise Injection",
    "Weight Swapping",
    "Krum Attack",
    "Trim-Mean Attack",
]

# Attacks used in Group B malicious-ratio sweep
RATIO_SWEEP_ATTACKS = ["Gaussian", "Trim-Mean Attack"]

# Malicious percentages expected in the sweep
MALICIOUS_PCTS = [0, 20, 40, 60, 80]


# ============================================================================
# Core logic
# ============================================================================


def plot_malicious_ratio(root_dir: str, output_dir: str, metric: str, fmt: str):  # noqa: C901
    """Generate the 1x2 malicious-ratio impact figure."""
    scenarios = scan_scenario_root(root_dir)
    logger.info(f"Total scenarios found: {len(scenarios)}")

    if not scenarios:
        logger.error("No scenarios found. Check --root-dir path and scenario naming.")
        return

    # Filter to erdosrenyi topology, no parameter/scalability sweeps
    relevant = [
        s
        for s in scenarios
        if s["topology"] == "erdosrenyi"
        and "gamma" not in s
        and "kappa" not in s
        and "alpha" not in s
        and "n_nodes" not in s
    ]
    logger.info(f"Erdos-Renyi scenarios (no param sweeps): {len(relevant)}")

    # Build lookup: {(agg_label, attack_label, pct): (mean, std)}
    data: dict[tuple[str, str, int], tuple[float, float]] = {}

    for meta in relevant:
        _, results_df = load_scenario_results(meta["scenario_dir"])
        if results_df is None:
            logger.debug(f"No results_summary.csv for {meta['scenario_name']}, skipping")
            continue

        mean, std = get_final_metric(results_df, metric)
        if mean is None:
            logger.debug(f"Metric '{metric}' not found in {meta['scenario_name']}")
            continue

        agg_label = meta["aggregator_label"]
        atk_label = meta["attack_label"]
        pct = meta["n_malicious_pct"]

        data[(agg_label, atk_label, pct)] = (mean, std if std is not None else 0.0)

    if not data:
        logger.error("No metric data collected. Ensure scenarios have analysis/results_summary.csv.")
        return

    # ── Build 1x2 figure ─────────────────────────────────────────────────────
    n_attacks = len(RATIO_SWEEP_ATTACKS)
    fig, axes = plt.subplots(1, n_attacks, figsize=(6 * n_attacks, 5), squeeze=False)

    for ax_idx, attack_label in enumerate(RATIO_SWEEP_ATTACKS):
        ax = axes[0][ax_idx]

        for agg_label in AGG_ORDER:
            pcts_available = []
            means = []
            stds = []

            for pct in MALICIOUS_PCTS:
                if pct == 0:  # noqa: SIM108
                    # Baseline: use noattack scenario at 0%
                    key = (agg_label, "No Attack", 0)
                else:
                    key = (agg_label, attack_label, pct)

                if key in data:
                    pcts_available.append(pct)
                    m, s = data[key]
                    means.append(m)
                    stds.append(s)

            if not pcts_available:
                logger.debug(f"No data for {agg_label} under {attack_label}")
                continue

            pcts_arr = np.array(pcts_available)
            means_arr = np.array(means)
            stds_arr = np.array(stds)

            color = AGG_COLORS[agg_label]
            marker = AGG_MARKERS[agg_label]

            ax.errorbar(
                pcts_arr,
                means_arr,
                yerr=stds_arr,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.5,
                capsize=3,
                label=agg_label,
            )

        ax.set_title(f"{metric} vs. Malicious Ratio \u2014 {attack_label}")
        ax.set_xlabel("Malicious %")
        ax.set_ylabel(metric)
        ax.set_xticks(MALICIOUS_PCTS)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = [], []
    for agg_label in AGG_ORDER:
        h = plt.Line2D(
            [0],
            [0],
            color=AGG_COLORS[agg_label],
            marker=AGG_MARKERS[agg_label],
            markersize=6,
            linewidth=1.5,
        )
        handles.append(h)
        labels.append(agg_label)

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"malicious_ratio_{metric.lower()}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Malicious ratio impact plot — F1 vs adversary percentage",
    )
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario folders")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument("--metric", default="F1", help="Metric to plot (default: F1)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure format (default: pdf)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "comparison_figures")
    plot_malicious_ratio(args.root_dir, output_dir, args.metric, args.format)


if __name__ == "__main__":
    main()
