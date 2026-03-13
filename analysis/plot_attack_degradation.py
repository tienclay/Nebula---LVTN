"""
Attack Degradation Analysis

Generates two figures showing how each attack degrades F1 performance relative
to a no-attack baseline, for every aggregator:

  Figure A: Grouped bar chart of absolute F1 drop (with propagated error bars)
  Figure B: Heatmap of relative degradation percentage

Also produces a LaTeX table combining both absolute and relative drops.

Usage:
    python analysis/plot_attack_degradation.py --root-dir experiments/results/
    python analysis/plot_attack_degradation.py --root-dir experiments/results/ --metric Accuracy --format png
"""

import argparse
import logging
import math
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
    get_final_metric,
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

# Attacks shown on the bar chart / heatmap (exclude "No Attack" baseline)
ATTACK_ORDER_NO_BASELINE = [a for a in ATTACK_ORDER if a != "No Attack"]


def _is_core_scenario(meta: dict) -> bool:
    """Return True if this scenario belongs to the core group."""
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


def _collect_data(root_dir: str, metric: str):
    """Scan root dir and collect final metric values for core scenarios.

    Returns:
        data: dict[(agg_label, attack_label)] -> (mean, std)
    """
    scenarios = scan_scenario_root(root_dir)
    core = [s for s in scenarios if _is_core_scenario(s)]
    logger.info(f"Found {len(core)} core scenarios out of {len(scenarios)} total")

    data: dict[tuple[str, str], tuple[float, float]] = {}
    for meta in core:
        _, results_df = load_scenario_results(meta["scenario_dir"])
        if results_df is None:
            logger.warning(f"No results_summary.csv for {meta['scenario_name']}, skipping")
            continue
        mean, std = get_final_metric(results_df, metric)
        if mean is None:
            logger.warning(f"Metric '{metric}' not found in {meta['scenario_name']}, skipping")
            continue
        agg_label = meta["aggregator_label"]
        atk_label = meta["attack_label"]
        data[(agg_label, atk_label)] = (mean, std if std is not None else 0.0)

    return data


def _compute_degradation(data):
    """Compute absolute and relative degradation from no-attack baselines.

    Returns:
        abs_drop: dict[(agg, attack)] -> (drop_mean, drop_std)
        rel_drop: dict[(agg, attack)] -> float (percentage)
    """
    abs_drop = {}
    rel_drop = {}

    for agg in AGG_ORDER:
        baseline = data.get((agg, "No Attack"))
        if baseline is None:
            logger.warning(f"No baseline (No Attack) found for {agg}, skipping")
            continue
        f1_base, std_base = baseline

        for atk in ATTACK_ORDER_NO_BASELINE:
            attack_val = data.get((agg, atk))
            if attack_val is None:
                continue
            f1_atk, std_atk = attack_val

            drop = f1_base - f1_atk
            # Propagated std: sqrt(std_base^2 + std_atk^2)
            drop_std = math.sqrt(std_base**2 + std_atk**2)
            abs_drop[(agg, atk)] = (drop, drop_std)

            if f1_base > 0:
                rel_drop[(agg, atk)] = (drop / f1_base) * 100.0
            else:
                rel_drop[(agg, atk)] = 0.0  # graceful handling of zero baseline

    return abs_drop, rel_drop


def plot_absolute_bar(abs_drop, metric, output_path):
    """Figure A: Grouped bar chart of absolute F1 drop."""
    n_attacks = len(ATTACK_ORDER_NO_BASELINE)
    aggs_with_data = [a for a in AGG_ORDER if any((a, atk) in abs_drop for atk in ATTACK_ORDER_NO_BASELINE)]
    if not aggs_with_data:
        logger.warning("No data for absolute degradation bar chart")
        return

    n_aggs = len(aggs_with_data)
    bar_width = 0.8 / n_aggs
    x = np.arange(n_attacks)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, agg in enumerate(aggs_with_data):
        drops = []
        errs = []
        for atk in ATTACK_ORDER_NO_BASELINE:
            val = abs_drop.get((agg, atk))
            if val is not None:
                drops.append(val[0])
                errs.append(val[1])
            else:
                drops.append(0.0)
                errs.append(0.0)

        offset = (i - n_aggs / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            drops,
            bar_width,
            yerr=errs,
            label=agg,
            color=AGG_COLORS.get(agg, "#888888"),
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

    ax.set_xlabel("Attack Type")
    ax.set_ylabel(f"Absolute {metric} Drop")
    ax.set_title(f"Attack-Induced {metric} Degradation (Absolute Drop)")
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_ORDER_NO_BASELINE, rotation=25, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved absolute degradation bar chart to {output_path}")


def plot_relative_heatmap(rel_drop, metric, output_path):
    """Figure B: Heatmap of relative degradation percentage."""
    aggs_with_data = [a for a in AGG_ORDER if any((a, atk) in rel_drop for atk in ATTACK_ORDER_NO_BASELINE)]
    if not aggs_with_data:
        logger.warning("No data for relative degradation heatmap")
        return

    n_aggs = len(aggs_with_data)
    n_attacks = len(ATTACK_ORDER_NO_BASELINE)

    matrix = np.full((n_aggs, n_attacks), np.nan)
    for i, agg in enumerate(aggs_with_data):
        for j, atk in enumerate(ATTACK_ORDER_NO_BASELINE):
            val = rel_drop.get((agg, atk))
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Mask NaN cells
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(n_aggs):
        for j in range(n_attacks):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(n_attacks))
    ax.set_xticklabels(ATTACK_ORDER_NO_BASELINE, rotation=25, ha="right")
    ax.set_yticks(range(n_aggs))
    ax.set_yticklabels(aggs_with_data)
    ax.set_title(f"Relative {metric} Degradation (%)")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Degradation (%)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved relative degradation heatmap to {output_path}")


def generate_latex_table(abs_drop, rel_drop, metric, output_path):
    """Generate a LaTeX table with both absolute and relative degradation."""
    aggs_with_data = [a for a in AGG_ORDER if any((a, atk) in abs_drop for atk in ATTACK_ORDER_NO_BASELINE)]
    if not aggs_with_data:
        logger.warning("No data for LaTeX degradation table")
        return

    n_attacks = len(ATTACK_ORDER_NO_BASELINE)
    col_spec = "l" + "c" * n_attacks

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Attack-induced {metric} degradation: absolute drop / relative drop (\\%)}}",
        "\\label{tab:attack_degradation}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Aggregator & " + " & ".join(ATTACK_ORDER_NO_BASELINE) + " \\\\",
        "\\midrule",
    ]

    for agg in aggs_with_data:
        row = [agg]
        for atk in ATTACK_ORDER_NO_BASELINE:
            abs_val = abs_drop.get((agg, atk))
            rel_val = rel_drop.get((agg, atk))
            if abs_val is not None and rel_val is not None:
                drop_mean, drop_std = abs_val
                cell = f"${drop_mean:.3f} \\pm {drop_std:.3f}$ ({rel_val:.1f}\\%)"
            else:
                cell = "--"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    logger.info(f"Saved LaTeX degradation table to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot attack-induced F1 degradation (absolute drop + relative heatmap)"
    )
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect data
    data = _collect_data(args.root_dir, args.metric)
    if not data:
        logger.error("No data found. Ensure scenario directories contain analysis/results_summary.csv")
        sys.exit(1)

    logger.info(f"Collected {args.metric} values for {len(data)} (aggregator, attack) pairs")

    # 2. Compute degradation
    abs_drop, rel_drop = _compute_degradation(data)
    if not abs_drop:
        logger.error("No degradation could be computed (missing baselines or attack results)")
        sys.exit(1)

    metric_lower = args.metric.lower()
    fmt = args.format

    # 3. Figure A: Absolute drop bar chart
    plot_absolute_bar(
        abs_drop,
        args.metric,
        os.path.join(args.output_dir, f"attack_degradation_absolute_{metric_lower}.{fmt}"),
    )

    # 4. Figure B: Relative drop heatmap
    plot_relative_heatmap(
        rel_drop,
        args.metric,
        os.path.join(args.output_dir, f"attack_degradation_relative_{metric_lower}.{fmt}"),
    )

    # 5. LaTeX table
    generate_latex_table(
        abs_drop,
        rel_drop,
        args.metric,
        os.path.join(args.output_dir, f"attack_degradation_{metric_lower}.tex"),
    )

    logger.info(f"All attack degradation outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
