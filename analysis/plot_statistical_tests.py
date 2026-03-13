"""
Pairwise Statistical Significance Tests

Performs pairwise Wilcoxon signed-rank tests (or Mann-Whitney U for small samples)
between aggregators for each attack scenario, and visualizes the results as
significance heatmaps.

Reads per-participant TensorBoard event files to extract individual benign-node
metric values, then tests whether aggregator pairs produce significantly different
distributions.

Output:
  - Heatmap figure: -log10(p-value) matrix per attack (2x3 grid)
  - LaTeX table: p-values for all pairwise comparisons

Usage:
    python analysis/plot_statistical_tests.py --root-dir /path/to/scenarios
    python analysis/plot_statistical_tests.py --root-dir /path/to/scenarios --metric Accuracy --format png
"""

import argparse
import logging
import os
import sys
import warnings

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


# ============================================================================
# Core group filter
# ============================================================================


def _is_core_scenario(meta: dict) -> bool:
    """Return True if scenario belongs to the core group (A)."""
    if meta.get("topology") != "erdosrenyi":
        return False
    return all(key not in meta for key in ("gamma", "kappa", "alpha", "n_nodes"))


# ============================================================================
# Per-node metric extraction (reuse from plot_node_distribution)
# ============================================================================


def _extract_per_node_final_metric(scenario_dir: str, metric: str = "F1") -> list[float]:
    """Extract per-node final metric values for benign nodes."""
    import glob

    from analysis.radar_metrics_aggregator import ALL_METRIC_TAGS, find_malicious_participants, read_participant_scalars

    metrics_dir = os.path.join(scenario_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        return []

    participant_dirs = glob.glob(os.path.join(metrics_dir, "participant_*"))
    n_total = len(participant_dirs)
    if n_total == 0:
        return []

    malicious = find_malicious_participants(scenario_dir)
    tb_tag = ALL_METRIC_TAGS.get(metric, f"Decentralized/{metric}")

    values = []
    for idx in range(n_total):
        if idx in malicious:
            continue
        try:
            scalars = read_participant_scalars(metrics_dir, idx)
        except Exception as e:
            logger.debug(f"Could not read participant {idx}: {e}")
            continue
        if scalars.get(tb_tag):
            values.append(scalars[tb_tag][-1][1])

    return values


# ============================================================================
# Statistical testing
# ============================================================================


def _pairwise_test(values_a: list[float], values_b: list[float]) -> float | None:
    """Run pairwise statistical test between two sets of per-node values.

    Uses Wilcoxon signed-rank test if both samples have >= 5 paired observations.
    Falls back to Mann-Whitney U for smaller samples.
    Returns p-value, or None if test cannot be performed.
    """
    from scipy.stats import mannwhitneyu, wilcoxon

    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)

    # Need at least 2 observations in each
    if len(a) < 2 or len(b) < 2:
        return None

    # Check if all values are identical (test would be meaningless)
    if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
        return 1.0

    try:
        # For paired test, need equal-length samples
        n_paired = min(len(a), len(b))
        if n_paired >= 5:
            # Wilcoxon signed-rank (paired)
            a_paired = a[:n_paired]
            b_paired = b[:n_paired]
            # Check if differences are all zero
            diff = a_paired - b_paired
            if np.allclose(diff, 0.0):
                return 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = wilcoxon(a_paired, b_paired, alternative="two-sided")
            return float(p)
        else:
            # Mann-Whitney U (unpaired) for small samples
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            return float(p)
    except (ValueError, ZeroDivisionError):
        return None


def _significance_marker(p: float | None) -> str:
    """Return significance annotation string."""
    if p is None:
        return "--"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ============================================================================
# Main plotting
# ============================================================================


def plot_statistical_tests(root_dir: str, output_dir: str, metric: str = "F1", fmt: str = "pdf"):  # noqa: C901
    """Generate significance heatmaps and LaTeX table."""
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
            continue

        values = _extract_per_node_final_metric(meta["scenario_dir"], metric)
        logger.info(f"  {meta['scenario_name']}: {len(values)} benign node values")

        if attack_label not in data:
            data[attack_label] = {}
        if agg_label not in data[attack_label]:
            data[attack_label][agg_label] = []
        data[attack_label][agg_label].extend(values)

    n_agg = len(AGG_ORDER)

    # Compute pairwise p-values per attack
    # {attack_label: n_agg x n_agg matrix of p-values (None where not computed)}
    pvalue_matrices: dict[str, np.ndarray] = {}

    for attack_label in ATTACK_ORDER:
        attack_data = data.get(attack_label, {})
        matrix = np.full((n_agg, n_agg), np.nan)

        for i in range(n_agg):
            for j in range(n_agg):
                if i == j:
                    continue
                vals_i = attack_data.get(AGG_ORDER[i], [])
                vals_j = attack_data.get(AGG_ORDER[j], [])
                p = _pairwise_test(vals_i, vals_j)
                if p is not None:
                    matrix[i, j] = p

        pvalue_matrices[attack_label] = matrix

    # ---- Figure: 2x3 heatmap grid ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), squeeze=False)

    for subplot_idx, attack_label in enumerate(ATTACK_ORDER):
        row, col = divmod(subplot_idx, 3)
        ax = axes[row][col]

        matrix = pvalue_matrices.get(attack_label)
        if matrix is None:
            matrix = np.full((n_agg, n_agg), np.nan)

        # Convert to -log10(p) for visualization (cap at 5 for display)
        with np.errstate(divide="ignore", invalid="ignore"):
            neg_log_p = -np.log10(matrix)
        neg_log_p = np.where(np.isfinite(neg_log_p), neg_log_p, np.nan)
        neg_log_p = np.clip(neg_log_p, 0, 5)

        # Create masked array for diagonal
        masked = np.ma.array(neg_log_p, mask=np.eye(n_agg, dtype=bool))

        # Plot heatmap
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="#d0d0d0")  # diagonal / missing = grey

        ax.imshow(masked, cmap=cmap, vmin=0, vmax=5, aspect="equal")

        # Annotate cells with significance markers and p-values
        for i in range(n_agg):
            for j in range(n_agg):
                if i == j:
                    ax.text(j, i, "---", ha="center", va="center", fontsize=9, color="#666666")
                    continue

                p_val = matrix[i, j]
                if np.isnan(p_val):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="#999999")
                else:
                    sig = _significance_marker(p_val)
                    # Choose text color based on background intensity
                    text_color = "white" if neg_log_p[i, j] > 3 else "black"
                    ax.text(
                        j,
                        i,
                        sig,
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                        color=text_color,
                    )
                    # Show p-value below the marker
                    p_str = f"{p_val:.3f}" if p_val >= 0.001 else f"{p_val:.1e}"
                    ax.text(
                        j,
                        i + 0.25,
                        p_str,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )

        ax.set_xticks(range(n_agg))
        ax.set_yticks(range(n_agg))
        ax.set_xticklabels(AGG_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(AGG_ORDER, fontsize=9)
        ax.set_title(attack_label, fontsize=12)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 5)),
        cax=cbar_ax,
    )
    cbar.set_label("$-\\log_{10}(p)$", fontsize=11)
    # Add significance threshold lines on colorbar
    for threshold, label in [(1.301, "p=0.05"), (2.0, "p=0.01"), (3.0, "p=0.001")]:
        cbar_ax.axhline(y=threshold, color="black", linewidth=0.5, linestyle="--")
        cbar_ax.text(1.8, threshold, label, fontsize=7, va="center")

    fig.suptitle(f"Pairwise Statistical Tests ({metric})", fontsize=15, y=0.98)
    fig.subplots_adjust(right=0.91, hspace=0.35, wspace=0.35)

    fig_path = os.path.join(output_dir, f"statistical_tests_{metric.lower()}.{fmt}")
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved statistical tests heatmap to {fig_path}")

    # ---- LaTeX table ----
    _generate_latex_table(pvalue_matrices, output_dir, metric)


def _generate_latex_table(
    pvalue_matrices: dict[str, np.ndarray],
    output_dir: str,
    metric: str,
):
    """Generate a LaTeX table summarizing pairwise p-values per attack."""
    n_agg = len(AGG_ORDER)

    lines = [
        "% Auto-generated pairwise statistical test results",
        f"% Metric: {metric}",
        "",
    ]

    for attack_label in ATTACK_ORDER:
        matrix = pvalue_matrices.get(attack_label)
        if matrix is None:
            continue

        # Check if there is any non-NaN data
        if np.all(np.isnan(matrix[~np.eye(n_agg, dtype=bool)])):
            continue

        safe_label = attack_label.replace(" ", "_").replace("-", "")
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Pairwise Wilcoxon p-values --- {attack_label} ({metric})}}",
            f"\\label{{tab:stat_{safe_label.lower()}_{metric.lower()}}}",
            f"\\begin{{tabular}}{{l{'c' * n_agg}}}",
            "\\toprule",
            " & " + " & ".join(AGG_ORDER) + " \\\\",
            "\\midrule",
        ])

        for i in range(n_agg):
            row_cells = [AGG_ORDER[i]]
            for j in range(n_agg):
                if i == j:
                    row_cells.append("---")
                elif np.isnan(matrix[i, j]):
                    row_cells.append("--")
                else:
                    p = matrix[i, j]
                    sig = _significance_marker(p)
                    if p < 0.001:
                        row_cells.append(f"${p:.1e}${sig}")
                    else:
                        row_cells.append(f"${p:.4f}${sig}")
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    # Add legend
    lines.extend([
        "% Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
        "% Test: Wilcoxon signed-rank (n>=5 paired) or Mann-Whitney U (n<5)",
    ])

    tex_path = os.path.join(output_dir, "statistical_tests.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved LaTeX table to {tex_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pairwise Statistical Significance Tests")
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
        help="Output directory for figures and tables (default: analysis/figures)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="F1",
        help="Metric to test (default: F1). Must match a key in ALL_METRIC_TAGS.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    args = parser.parse_args()

    plot_statistical_tests(args.root_dir, args.output_dir, args.metric, args.format)


if __name__ == "__main__":
    main()
