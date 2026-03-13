"""
Convergence Speed Analysis

Compares convergence speed across aggregators using two complementary metrics:

  AUC (Area Under Convergence Curve):
    Trapezoidal integration of F1 mean values, normalized by total rounds.
    Higher = better (faster convergence + higher final value).

  Rounds to Threshold:
    First round where F1 mean >= threshold.
    Lower = better (faster convergence). "Never reached" is shown with hatching.

Generates:
  Figure A: AUC bar chart (2x3 grid, one subplot per attack)
  Figure B: Rounds-to-threshold bar chart (2x3 grid, one subplot per attack)
  LaTeX summary table

Usage:
    python analysis/plot_convergence_speed.py --root-dir experiments/results/
    python analysis/plot_convergence_speed.py --root-dir experiments/results/ --threshold 0.7 --format png
"""

import argparse
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

from analysis.scenario_metadata import (  # noqa: E402
    load_scenario_results,
    scan_scenario_root,
)

AGG_COLORS = {
    "FedAvg": "#1f77b4",
    "Krum": "#ff7f0e",
    "TrimmedMean": "#2ca02c",
    "Median": "#d62728",
    "RADAR-Agg": "#9467bd",
}
AGG_ORDER = ["FedAvg", "Krum", "TrimmedMean", "Median", "RADAR-Agg"]
ATTACK_ORDER = ["No Attack", "Gaussian", "Noise Injection", "Weight Swapping", "Krum Attack", "Trim-Mean Attack"]


def _is_core_scenario(meta: dict) -> bool:
    """Return True if this scenario belongs to the core group."""
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


def _compute_auc(steps, values):
    """Compute normalized AUC using trapezoidal integration.

    Returns AUC normalized by max_round (so result is in [0, 1] for bounded metrics).
    """
    if len(steps) < 2:
        return None
    auc = float(np.trapz(values, steps))
    max_round = steps[-1] - steps[0]
    if max_round <= 0:
        return None
    return auc / max_round


def _compute_rounds_to_threshold(steps, values, threshold):
    """Find the first round where value >= threshold.

    Returns the round number, or None if threshold is never reached.
    """
    for step, val in zip(steps, values, strict=False):
        if val >= threshold:
            return int(step)
    return None


def _collect_convergence_data(root_dir: str, metric: str, threshold: float):
    """Scan root dir and compute AUC and rounds-to-threshold for core scenarios.

    Returns:
        auc_data: dict[(agg_label, attack_label)] -> float
        rtt_data: dict[(agg_label, attack_label)] -> int | None
        max_round: int  (maximum round observed across all scenarios)
    """
    scenarios = scan_scenario_root(root_dir)
    core = [s for s in scenarios if _is_core_scenario(s)]
    logger.info(f"Found {len(core)} core scenarios out of {len(scenarios)} total")

    auc_data: dict[tuple[str, str], float] = {}
    rtt_data: dict[tuple[str, str], int | None] = {}
    global_max_round = 0

    for meta in core:
        convergence_dict, _ = load_scenario_results(meta["scenario_dir"])
        if convergence_dict is None:
            logger.warning(f"No convergence_curves.json for {meta['scenario_name']}, skipping")
            continue

        # convergence_dict structure: {scenario_name: {metric: [{step, mean, ...}]}}
        # The scenario name key inside the JSON may differ from folder name;
        # iterate over all keys and pick the first one that has the metric.
        curve_data = None
        for _key, metrics_dict in convergence_dict.items():
            if metric in metrics_dict:
                curve_data = metrics_dict[metric]
                break

        if curve_data is None:
            logger.warning(f"Metric '{metric}' not found in convergence data for {meta['scenario_name']}")
            continue

        valid = [e for e in curve_data if e.get("mean") is not None]
        if len(valid) < 2:
            logger.warning(f"Insufficient data points for {meta['scenario_name']}, skipping")
            continue

        steps = np.array([e["step"] for e in valid])
        means = np.array([e["mean"] for e in valid])

        agg_label = meta["aggregator_label"]
        atk_label = meta["attack_label"]

        # AUC
        auc = _compute_auc(steps, means)
        if auc is not None:
            auc_data[(agg_label, atk_label)] = auc

        # Rounds to threshold
        rtt = _compute_rounds_to_threshold(steps, means, threshold)
        rtt_data[(agg_label, atk_label)] = rtt

        # Track max round
        if len(steps) > 0:
            global_max_round = max(global_max_round, int(steps[-1]))

    return auc_data, rtt_data, global_max_round


def plot_auc_grid(auc_data, metric, output_path):
    """Figure A: 2x3 grid of AUC bar charts, one subplot per attack."""
    n_attacks = len(ATTACK_ORDER)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), squeeze=False)

    for idx, atk in enumerate(ATTACK_ORDER):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        agg_labels = []
        auc_values = []
        colors = []

        for agg in AGG_ORDER:
            val = auc_data.get((agg, atk))
            if val is not None:
                agg_labels.append(agg)
                auc_values.append(val)
                colors.append(AGG_COLORS.get(agg, "#888888"))

        if agg_labels:
            x = np.arange(len(agg_labels))
            ax.bar(x, auc_values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(agg_labels, rotation=35, ha="right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")

        ax.set_title(atk, fontsize=11)
        ax.set_ylabel("Normalized AUC")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(n_attacks, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Convergence Speed: Normalized AUC ({metric})", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved AUC grid to {output_path}")


def plot_rtt_grid(rtt_data, max_round, metric, threshold, output_path):
    """Figure B: 2x3 grid of rounds-to-threshold bar charts."""
    n_attacks = len(ATTACK_ORDER)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), squeeze=False)

    never_reached_height = max_round + 1 if max_round > 0 else 100

    for idx, atk in enumerate(ATTACK_ORDER):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        agg_labels = []
        rtt_values = []
        colors = []
        hatches = []

        for agg in AGG_ORDER:
            if (agg, atk) not in rtt_data:
                continue
            val = rtt_data[(agg, atk)]
            agg_labels.append(agg)
            if val is not None:
                rtt_values.append(val)
                hatches.append("")
            else:
                rtt_values.append(never_reached_height)
                hatches.append("///")
            colors.append(AGG_COLORS.get(agg, "#888888"))

        if agg_labels:
            x = np.arange(len(agg_labels))
            bars = ax.bar(x, rtt_values, color=colors, edgecolor="black", linewidth=0.5)
            for bar, hatch in zip(bars, hatches, strict=False):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(agg_labels, rotation=35, ha="right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")

        ax.set_title(atk, fontsize=11)
        ax.set_ylabel("Round Number")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(n_attacks, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Rounds to Reach {metric} $\\geq$ {threshold} (hatched = never reached)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved rounds-to-threshold grid to {output_path}")


def generate_latex_table(auc_data, rtt_data, max_round, metric, threshold, output_path):
    """Generate a summary LaTeX table with AUC and rounds-to-threshold."""
    aggs_with_data = [a for a in AGG_ORDER if any((a, atk) in auc_data or (a, atk) in rtt_data for atk in ATTACK_ORDER)]
    if not aggs_with_data:
        logger.warning("No data for LaTeX convergence speed table")
        return

    n_attacks = len(ATTACK_ORDER)
    # Two sub-columns per attack: AUC and RTT
    col_spec = "l" + "c" * (n_attacks * 2)

    # Header with multicolumn for each attack
    header_top_parts = ["Aggregator"]
    header_bot_parts = [""]
    for atk in ATTACK_ORDER:
        header_top_parts.append(f"\\multicolumn{{2}}{{c}}{{{atk}}}")
        header_bot_parts.append("AUC")
        header_bot_parts.append("RTT")

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Convergence speed: Normalized AUC and Rounds to {metric} $\\geq$ {threshold}}}",
        "\\label{tab:convergence_speed}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_top_parts) + " \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} "
        "\\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}",
        " & ".join(header_bot_parts) + " \\\\",
        "\\midrule",
    ]

    for agg in aggs_with_data:
        row = [agg]
        for atk in ATTACK_ORDER:
            # AUC
            auc_val = auc_data.get((agg, atk))
            if auc_val is not None:
                row.append(f"${auc_val:.3f}$")
            else:
                row.append("--")
            # RTT
            rtt_val = rtt_data.get((agg, atk))
            if (agg, atk) in rtt_data:
                if rtt_val is not None:
                    row.append(f"${rtt_val}$")
                else:
                    row.append("$>$" + str(max_round))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    logger.info(f"Saved LaTeX convergence speed table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence speed analysis (AUC + rounds-to-threshold)")
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
        help="Output directory for generated figures (default: analysis/figures)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="F1",
        help="Metric to analyze (default: F1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for rounds-to-threshold metric (default: 0.8)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect convergence data
    auc_data, rtt_data, max_round = _collect_convergence_data(args.root_dir, args.metric, args.threshold)

    if not auc_data and not rtt_data:
        logger.error("No convergence data found. Ensure scenario directories contain analysis/convergence_curves.json")
        sys.exit(1)

    logger.info(f"Computed AUC for {len(auc_data)} pairs, RTT for {len(rtt_data)} pairs (max_round={max_round})")

    metric_lower = args.metric.lower()
    fmt = args.format

    # 2. Figure A: AUC grid
    if auc_data:
        plot_auc_grid(
            auc_data,
            args.metric,
            os.path.join(args.output_dir, f"convergence_speed_auc_{metric_lower}.{fmt}"),
        )

    # 3. Figure B: Rounds-to-threshold grid
    if rtt_data:
        plot_rtt_grid(
            rtt_data,
            max_round,
            args.metric,
            args.threshold,
            os.path.join(args.output_dir, f"convergence_speed_rounds_{metric_lower}.{fmt}"),
        )

    # 4. LaTeX table
    generate_latex_table(
        auc_data,
        rtt_data,
        max_round,
        args.metric,
        args.threshold,
        os.path.join(args.output_dir, f"convergence_speed_{metric_lower}.tex"),
    )

    logger.info(f"All convergence speed outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
