"""
F1 Heatmap: Aggregator x Attack

Generates a heatmap matrix where rows are aggregation algorithms and columns are
attack types.  Each cell shows the final F1 score (or other metric) for that
combination, restricted to the core experiment group (Erdos-Renyi topology,
no parameter/scalability sweeps).

Also produces a companion LaTeX table (.tex) for direct inclusion in papers.

Usage:
    python analysis/plot_f1_heatmap.py --root-dir /path/to/scenarios
    python analysis/plot_f1_heatmap.py --root-dir /path/to/scenarios --metric Accuracy --format png
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

# Canonical display orders
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


def _is_core_scenario(meta: dict) -> bool:
    """Return True if the scenario belongs to the core group (A).

    Core group: Erdos-Renyi topology, no parameter sweeps, no scalability sweeps.
    """
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


def build_heatmap_matrix(
    scenarios: list[dict],
    metric: str = "F1",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build a 2D matrix of metric values: rows=aggregators, cols=attacks.

    Returns:
        values: 2D array of means (NaN where data is missing)
        stds:   2D array of stds  (NaN where data is missing)
        row_labels: aggregator labels in AGG_ORDER
        col_labels: attack labels in ATTACK_ORDER
    """
    n_rows = len(AGG_ORDER)
    n_cols = len(ATTACK_ORDER)
    values = np.full((n_rows, n_cols), np.nan)
    stds = np.full((n_rows, n_cols), np.nan)

    for meta in scenarios:
        agg_label = meta["aggregator_label"]
        atk_label = meta["attack_label"]

        if agg_label not in AGG_ORDER or atk_label not in ATTACK_ORDER:
            continue

        row_idx = AGG_ORDER.index(agg_label)
        col_idx = ATTACK_ORDER.index(atk_label)

        _, results_df = load_scenario_results(meta["scenario_dir"])
        mean, std = get_final_metric(results_df, metric)

        if mean is not None:
            values[row_idx, col_idx] = mean
        if std is not None:
            stds[row_idx, col_idx] = std

    return values, stds, AGG_ORDER, ATTACK_ORDER


def plot_heatmap(
    values: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    metric: str,
    output_path: str,
):
    """Render and save the heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Mask NaN cells for the colormap
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Axis ticks
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=10, color="gray")
                continue
            # Choose text color based on cell brightness
            text_color = "white" if val < 0.5 else "black"
            std_val = stds[i, j]
            label = f"{val:.3f}\n({std_val:.3f})" if not np.isnan(std_val) and std_val > 0 else f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color)

    ax.set_title(f"Final {metric} Score: Aggregator x Attack")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


def generate_latex_table(
    values: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    metric: str,
    output_path: str,
):
    """Generate a LaTeX table version of the heatmap."""
    col_spec = "l" + "c" * len(col_labels)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Final {metric} score: Aggregator $\\times$ Attack (mean $\\pm$ std)}}",
        f"\\label{{tab:{metric.lower()}_heatmap}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & " + " & ".join(c.replace("&", "\\&") for c in col_labels) + " \\\\",
        "\\midrule",
    ]

    for i, agg in enumerate(row_labels):
        cells = [agg]
        for j in range(len(col_labels)):
            val = values[i, j]
            std_val = stds[i, j]
            if np.isnan(val):
                cells.append("--")
            elif not np.isnan(std_val) and std_val > 0:
                cells.append(f"${val:.4f} \\pm {std_val:.4f}$")
            else:
                cells.append(f"${val:.4f}$")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate F1 heatmap (Aggregator x Attack) from scenario results",
    )
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario subdirectories")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument("--metric", default="F1", help="Metric to plot (default: F1)")
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

    # Build matrix
    values, stds, row_labels, col_labels = build_heatmap_matrix(core, metric=args.metric)

    # Plot heatmap
    fig_path = os.path.join(output_dir, f"f1_heatmap.{args.format}")
    plot_heatmap(values, stds, row_labels, col_labels, args.metric, fig_path)

    # Generate LaTeX table
    tex_path = os.path.join(output_dir, "f1_heatmap.tex")
    generate_latex_table(values, stds, row_labels, col_labels, args.metric, tex_path)

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
