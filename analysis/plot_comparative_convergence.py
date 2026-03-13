"""
Comparative Convergence Plot

Generates a 2x3 grid of subplots comparing aggregator convergence curves
under each attack scenario. Each subplot shows F1 (or other metric) vs
training round for all aggregators, with mean line and +/- std shaded band.

Uses core experiment group only (Erdos-Renyi topology, no parameter/scalability sweeps).

Usage:
    python -m analysis.plot_comparative_convergence --root-dir /path/to/experiments
    python -m analysis.plot_comparative_convergence --root-dir /path/to/experiments --metric Accuracy
"""

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis.scenario_metadata import (
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

# Metrics bounded in [0, 1]
_BOUNDED_01_METRICS = {"F1", "Accuracy", "Precision", "Recall"}


# ============================================================================
# Core logic
# ============================================================================


def is_core_scenario(meta: dict) -> bool:
    """Return True if this scenario belongs to the core group.

    Core group: Erdos-Renyi topology, no parameter sweep tokens, no scalability sweep.
    """
    if meta["topology"] != "erdosrenyi":
        return False
    # Exclude parameter sweeps (gamma/kappa/alpha variants)
    if any(k in meta for k in ("gamma", "kappa", "alpha")):
        return False
    # Exclude scalability sweeps (n_nodes token)
    return "n_nodes" not in meta


def plot_comparative_convergence(root_dir: str, output_dir: str, metric: str, fmt: str):  # noqa: C901
    """Generate the 2x3 comparative convergence figure."""
    scenarios = scan_scenario_root(root_dir)
    core = [s for s in scenarios if is_core_scenario(s)]
    logger.info(f"Core scenarios after filtering: {len(core)}")

    if not core:
        logger.error("No core scenarios found. Check --root-dir path and scenario naming.")
        return

    # Organise: {attack_label: {agg_label: (convergence_dict, scenario_name)}}
    data_by_attack: dict[str, dict[str, tuple]] = {atk: {} for atk in ATTACK_ORDER}

    for meta in core:
        conv_dict, _ = load_scenario_results(meta["scenario_dir"])
        if conv_dict is None:
            logger.warning(f"No convergence data for {meta['scenario_name']}, skipping")
            continue

        atk_label = meta["attack_label"]
        agg_label = meta["aggregator_label"]

        if atk_label not in data_by_attack:
            logger.debug(f"Attack '{atk_label}' not in ATTACK_ORDER, skipping {meta['scenario_name']}")
            continue

        # convergence_dict is {scenario_name: {metric: [entries]}}
        # Pick the first (and usually only) key
        scenario_key = next(iter(conv_dict), None)
        if scenario_key is None:
            continue

        data_by_attack[atk_label][agg_label] = (conv_dict[scenario_key], meta["scenario_name"])

    # ── Build 2x3 figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)

    for idx, attack_label in enumerate(ATTACK_ORDER):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        agg_data = data_by_attack.get(attack_label, {})

        for agg_label in AGG_ORDER:
            if agg_label not in agg_data:
                continue

            metrics_dict, scenario_name = agg_data[agg_label]
            if metric not in metrics_dict:
                logger.debug(f"Metric '{metric}' not found for {scenario_name}")
                continue

            series = metrics_dict[metric]
            valid = [e for e in series if e.get("mean") is not None]
            if not valid:
                continue

            steps = np.array([e["step"] for e in valid])
            means = np.array([e["mean"] for e in valid])
            stds = np.array([e.get("std", 0.0) for e in valid])

            color = AGG_COLORS[agg_label]
            marker = AGG_MARKERS[agg_label]
            me = max(1, len(steps) // 10)

            ax.plot(
                steps,
                means,
                color=color,
                marker=marker,
                markevery=me,
                markersize=4,
                linewidth=1.5,
                label=agg_label,
            )

            # Shaded +/- std band
            band_lo = means - stds
            band_hi = means + stds
            if metric in _BOUNDED_01_METRICS:
                band_lo = np.clip(band_lo, 0.0, 1.0)
                band_hi = np.clip(band_hi, 0.0, 1.0)
            ax.fill_between(steps, band_lo, band_hi, color=color, alpha=0.15)

        ax.set_title(attack_label)
        ax.set_xlabel("Training Round")
        ax.set_ylabel(metric)
        if metric in _BOUNDED_01_METRICS:
            ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)

    # Shared legend at the bottom
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

    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Convergence Comparison: {metric}", fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"comparative_convergence_{metric.lower()}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comparative convergence plot — aggregators vs attacks (2x3 grid)",
    )
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario folders")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument("--metric", default="F1", help="Metric to plot (default: F1)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure format (default: pdf)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "comparison_figures")

    # Generate requested metric
    plot_comparative_convergence(args.root_dir, output_dir, args.metric, args.format)

    # Also generate Accuracy version automatically (if primary metric is not Accuracy)
    if args.metric != "Accuracy":
        plot_comparative_convergence(args.root_dir, output_dir, "Accuracy", args.format)


if __name__ == "__main__":
    main()
