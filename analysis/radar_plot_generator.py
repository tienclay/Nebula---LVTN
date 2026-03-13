"""
RADAR Paper Plot Generator

Generates publication-quality figures and LaTeX tables from processed metrics.
Reads output from radar_metrics_aggregator.py (results_summary.csv, convergence_curves.json).

Figures generated:
  1. Convergence curves with std bands (F1 vs round, per attack scenario)
     - Uses median as central tendency (robust to attacker outliers)
     - Shows min/max lines to distinguish benign vs malicious nodes
     - Clips confidence bands to valid metric ranges
  2. Enhanced bar charts with error bars (Training Time, Network, CPU, RAM)
  3. Aggregation behavior plots (Threshold decay, Neighbor accept rate, Avg distance)
  4. Full metrics comparison table (F1 + Accuracy + Precision + Recall with ± std)

Usage:
    python analysis/radar_plot_generator.py --input analysis/results/ --output analysis/figures/
    python analysis/radar_plot_generator.py --convergence-file convergence_curves.json --output figures/
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Consistent color scheme matching paper conventions
METHOD_COLORS = {
    "FedAvg": "#1f77b4",  # blue
    "Krum": "#ff7f0e",  # orange
    "TrimmedMean": "#2ca02c",  # green
    "Trim-Mean": "#2ca02c",  # green (alias)
    "Median": "#d62728",  # red
    "RADAR-Agg": "#9467bd",  # purple
    "Balance": "#9467bd",  # purple (alias for RADAR-Agg)
}

METHOD_MARKERS = {
    "FedAvg": "o",
    "Krum": "s",
    "TrimmedMean": "^",
    "Trim-Mean": "^",
    "Median": "D",
    "RADAR-Agg": "*",
    "Balance": "*",
}

ATTACK_LABELS = {
    "none": "No Attack",
    "gaussian": "Gaussian",
    "noise_injection": "Noise Injection",
    "swapping": "Weight Swapping",
    "trimmedmean": "Trim-Mean",
    "no_attack": "No Attack",
}

# Metrics that are bounded in [0, 1]
_BOUNDED_01_METRICS = {"F1", "Accuracy", "Precision", "Recall", "NeighborAcceptRate", "CosineSimilarity"}
# Metrics that must be non-negative
_NONNEG_METRICS = {
    "Loss",
    "Threshold",
    "SimilarNeighbors",
    "TotalNeighbors",
    "AvgNeighborDist",
    "LocalModelNorm",
    "GlobalModelNorm",
    "ModelNormDelta",
}


def _clip_band(lower: np.ndarray, upper: np.ndarray, metric: str):
    """Clip confidence band to the valid range for a given metric."""
    if metric in _BOUNDED_01_METRICS:
        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
    if metric in _NONNEG_METRICS:
        return np.clip(lower, 0.0, None), upper
    return lower, upper


def load_convergence_data(filepath: str) -> dict:
    """Load convergence curves JSON from radar_metrics_aggregator."""
    with open(filepath) as f:
        return json.load(f)


def load_results_summary(filepath: str) -> pd.DataFrame:
    """Load results summary CSV from radar_metrics_aggregator."""
    return pd.read_csv(filepath)


def parse_scenario_name(name: str) -> dict:
    """
    Parse scenario name to extract method, attack, config info.
    Naming convention expected: e.g., "radar_gauss_baseline", "fedavg_none_encrypt"
    Users should adapt this to their naming convention.
    """
    parts = name.lower().replace("-", "_").split("_")
    return {"raw": name, "parts": parts}


# ============================================================================
# Figure 1: Convergence Curves with Std Bands
# ============================================================================


def plot_convergence_curves(
    convergence_data: dict,
    metric: str = "F1",
    output_path: str = "convergence_curves.pdf",
    scenarios_per_subplot: dict | None = None,
):
    """
    Plot F1 (or other metric) vs training round with median line, ± std shaded
    band (clipped to valid range), and min/max dashed lines to show the spread
    between benign and malicious nodes.

    Args:
        convergence_data: {scenario_name: {metric_name: [{step, mean, std, min, max, ...}]}}
        metric: which metric to plot (e.g., "F1", "Accuracy")
        output_path: where to save the figure
        scenarios_per_subplot: optional dict {subplot_title: [scenario_names]}
            If None, all scenarios are plotted on a single figure.
    """
    if scenarios_per_subplot is None:
        # Single plot with all scenarios
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _plot_convergence_on_axis(ax, convergence_data, metric)
        ax.set_xlabel("Training Round")
        ylabel = f"{metric} Score" if metric in _BOUNDED_01_METRICS else metric
        ax.set_ylabel(ylabel)
        if metric in _BOUNDED_01_METRICS:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    else:
        # Multiple subplots
        n_plots = len(scenarios_per_subplot)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for idx, (title, scenario_names) in enumerate(scenarios_per_subplot.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            subset = {k: v for k, v in convergence_data.items() if k in scenario_names}
            _plot_convergence_on_axis(ax, subset, metric)
            ax.set_title(title)
            ax.set_xlabel("Training Round")
            ylabel = f"{metric} Score" if metric in _BOUNDED_01_METRICS else metric
            ax.set_ylabel(ylabel)
            if metric in _BOUNDED_01_METRICS:
                ax.set_ylim(-0.02, 1.05)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved convergence curves to {output_path}")


def _plot_convergence_on_axis(ax, convergence_data: dict, metric: str):
    """Plot convergence curves for multiple scenarios on a single axis.

    Shows mean line with +/- std shaded band (clipped to valid range) and
    min/max as a lighter range band when spread is notable.
    Data is expected to be benign-only (malicious nodes excluded upstream).
    """
    for scenario_name, metrics in convergence_data.items():
        if metric not in metrics:
            continue

        series = metrics[metric]
        valid = [e for e in series if e.get("mean") is not None]
        if not valid:
            continue

        steps = np.array([e["step"] for e in valid])
        means = np.array([e["mean"] for e in valid])
        stds = np.array([e.get("std", 0.0) for e in valid])
        mins = np.array([e.get("min", e["mean"]) for e in valid])
        maxs = np.array([e.get("max", e["mean"]) for e in valid])

        color = _get_color(scenario_name)
        marker = _get_marker(scenario_name)
        label = _get_label(scenario_name)

        # Plot mean line
        ax.plot(
            steps,
            means,
            color=color,
            marker=marker,
            markevery=max(1, len(steps) // 10),
            markersize=4,
            linewidth=1.5,
            label=label,
        )

        # Plot +/- 1 std band (clipped to valid range)
        band_lo, band_hi = _clip_band(means - stds, means + stds, metric)
        ax.fill_between(steps, band_lo, band_hi, color=color, alpha=0.15)

        # Show min/max range as lighter band when there is notable spread
        if not np.allclose(mins, maxs):
            min_clipped, max_clipped = _clip_band(mins, maxs, metric)
            ax.fill_between(steps, min_clipped, max_clipped, color=color, alpha=0.06)
            ax.plot(steps, mins, color=color, linestyle=":", linewidth=0.6, alpha=0.4)
            ax.plot(steps, maxs, color=color, linestyle=":", linewidth=0.6, alpha=0.4)


def _get_color(scenario_name: str) -> str:
    """Get color for a scenario based on method name matching."""
    name_lower = scenario_name.lower()
    for method, color in METHOD_COLORS.items():
        if method.lower() in name_lower:
            return color
    # Default color cycle
    return plt.cm.tab10(hash(scenario_name) % 10)


def _get_marker(scenario_name: str) -> str:
    """Get marker for a scenario based on method name matching."""
    name_lower = scenario_name.lower()
    for method, marker in METHOD_MARKERS.items():
        if method.lower() in name_lower:
            return marker
    return "o"


def _get_label(scenario_name: str) -> str:
    """Get clean label for legend."""
    # Try to extract method name
    name_lower = scenario_name.lower()
    for method in METHOD_COLORS:
        if method.lower() in name_lower:
            return method
    return scenario_name


# ============================================================================
# Figure 2: Bar Charts with Error Bars
# ============================================================================


def plot_bar_charts_with_errors(
    results: pd.DataFrame,
    output_path: str = "bar_charts.pdf",
    metrics: list[str] | None = None,
):
    """
    Generate bar charts with error bars for multi-scenario comparison.

    Args:
        results: DataFrame from results_summary.csv
        output_path: where to save
        metrics: which metrics to plot (default: Time, CPU, RAM, Network)
    """
    if metrics is None:
        metrics = ["CPU (%)", "RAM (%)", "Network (bytes sent)", "Network (packets sent)"]

    available = [m for m in metrics if m in results["metric"].values]
    if not available:
        logger.warning("No system metrics found in results for bar charts")
        return

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)

    for idx, metric_name in enumerate(available):
        ax = axes[0][idx]
        metric_data = results[results["metric"] == metric_name].copy()

        if metric_data.empty:
            continue

        scenarios = metric_data["scenario"].values
        means = metric_data["mean"].values
        stds = metric_data["std"].values

        colors = [_get_color(s) for s in scenarios]
        labels = [_get_label(s) for s in scenarios]

        ax.bar(range(len(scenarios)), means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved bar charts to {output_path}")


def plot_resource_over_time(  # noqa: C901
    convergence_data: dict,
    output_path: str = "resource_usage.pdf",
    metrics: list[str] | None = None,
):
    """
    Plot resource usage metrics (CPU, RAM, Network) over training rounds.

    Args:
        convergence_data: {scenario_name: {metric_name: [{step, mean, std, ...}]}}
        output_path: where to save
        metrics: which resource metrics to plot
    """
    if metrics is None:
        metrics = ["CPU (%)", "RAM (%)", "Network (bytes sent)", "Network (bytes received)"]

    # Check which metrics have actual data points (not just empty lists)
    available = []
    for m in metrics:
        for scenario_metrics in convergence_data.values():
            if m in scenario_metrics and len(scenario_metrics[m]) > 0:
                available.append(m)
                break

    if not available:
        logger.warning("No resource metrics with data found in convergence data; generating placeholder")
        # Generate a placeholder figure indicating no data
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for idx, metric_name in enumerate(metrics):
            row, col = divmod(idx, 2)
            ax = axes[row][col]
            ax.set_xlabel("Training Round")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.text(
                0.5,
                0.5,
                "No data collected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                fontstyle="italic",
            )
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved resource usage placeholder to {output_path}")
        return

    n_metrics = len(available)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, metric_name in enumerate(available):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for scenario_name, scenario_metrics in convergence_data.items():
            if metric_name not in scenario_metrics:
                continue

            series = scenario_metrics[metric_name]
            valid = [e for e in series if e.get("mean") is not None]
            if not valid:
                continue

            steps = np.array([e["step"] for e in valid])
            means = np.array([e["mean"] for e in valid])
            stds = np.array([e.get("std", 0.0) for e in valid])

            color = _get_color(scenario_name)
            label = _get_label(scenario_name)

            ax.plot(steps, means, color=color, marker="o", markersize=3, linewidth=1.5, label=label)
            band_lo, band_hi = _clip_band(means - stds, means + stds, metric_name)
            ax.fill_between(steps, band_lo, band_hi, color=color, alpha=0.15)

        ax.set_xlabel("Training Round")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved resource usage plots to {output_path}")


# ============================================================================
# Figure 3: Aggregation Behavior (Balance/RADAR-Agg specific)
# ============================================================================


def plot_aggregation_behavior(
    convergence_data: dict,
    scenario_name: str,
    output_path: str = "aggregation_behavior.pdf",
):
    """
    Plot RADAR-Agg aggregation behavior over rounds:
      1. Threshold tau_t decay
      2. Number of accepted neighbors
      3. Average neighbor distance
      4. Neighbor accept rate

    Bands are clipped to valid ranges (non-negative for thresholds/distances,
    [0,1] for rates).

    Args:
        convergence_data: convergence data for a single scenario
        scenario_name: which scenario to plot
        output_path: where to save
    """
    if scenario_name not in convergence_data:
        logger.warning(f"Scenario {scenario_name} not found in convergence data")
        return

    metrics = convergence_data[scenario_name]

    plot_configs = [
        ("Threshold", "Threshold $\\tau_t$", "Threshold Value"),
        ("SimilarNeighbors", "Accepted Neighbors $|S_i^t|$", "Count"),
        ("AvgNeighborDist", "Avg. Neighbor Distance", "L2 Distance"),
        ("NeighborAcceptRate", "Neighbor Accept Rate", "Rate"),
    ]

    available = [(tag, title, ylabel) for tag, title, ylabel in plot_configs if tag in metrics]
    if not available:
        logger.warning("No aggregation metrics found for plotting")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.5), squeeze=False)

    for idx, (tag, title, ylabel) in enumerate(available):
        ax = axes[0][idx]
        series = metrics[tag]

        valid = [e for e in series if e.get("mean") is not None]
        if not valid:
            continue

        steps = np.array([e["step"] for e in valid])
        means = np.array([e["mean"] for e in valid])
        stds = np.array([e.get("std", 0.0) for e in valid])
        mins = np.array([e.get("min", e["mean"]) for e in valid])
        maxs = np.array([e.get("max", e["mean"]) for e in valid])

        ax.plot(steps, means, color="#9467bd", linewidth=1.5, label="Mean")

        # Clip band to valid range
        band_lo, band_hi = _clip_band(means - stds, means + stds, tag)
        ax.fill_between(steps, band_lo, band_hi, color="#9467bd", alpha=0.2)

        # Show min/max as thin dashed lines
        if not np.allclose(mins, maxs):
            ax.plot(steps, mins, color="#9467bd", linestyle=":", linewidth=0.7, alpha=0.5, label="Min")
            ax.plot(steps, maxs, color="#9467bd", linestyle="--", linewidth=0.7, alpha=0.5, label="Max")

        ax.set_xlabel("Training Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # Set y-axis lower bound for non-negative metrics
        if tag in _NONNEG_METRICS:
            ax.set_ylim(bottom=0)
        if tag in _BOUNDED_01_METRICS:
            ax.set_ylim(-0.02, 1.05)

    fig.suptitle(f"RADAR-Agg Aggregation Behavior \u2014 {scenario_name}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved aggregation behavior plot to {output_path}")


# ============================================================================
# Figure 4: Model Convergence Dashboard
# ============================================================================


def plot_convergence_metrics(  # noqa: C901
    convergence_data: dict,
    output_path: str = "convergence_model_metrics.pdf",
):
    """
    Plot model convergence diagnostics.

    Preferred panels (when data is available from round-aligned logging):
      1. ModelNormDelta: ||w_t - w_{t-1}|| / ||w_t|| -- should decrease toward 0
      2. GlobalModelNorm: ||w_t|| -- should stabilize
      3. CosineSimilarity: cos(local, aggregated) -- should approach 1.0

    Fallback panels (from training-step-level logs):
      - Training Loss, Accuracy

    Data is expected to be benign-only (malicious nodes excluded upstream).
    Clips bands to valid ranges.
    """
    # Define preferred convergence panels in priority order
    convergence_panels = [
        ("ModelNormDelta", "Relative Weight Change $\\|\\Delta\\mathbf{w}\\|/\\|\\mathbf{w}\\|$", "Relative Change"),
        ("GlobalModelNorm", "Global Model Norm $\\|\\mathbf{w}^t\\|$", "L2 Norm"),
        ("CosineSimilarity", "Local-Global Cosine Similarity", "Cosine Similarity"),
    ]
    fallback_panels = [
        ("Loss", "Training Loss", "Loss"),
        ("Accuracy", "Accuracy", "Accuracy"),
    ]

    def _has_data(metric_name):
        return any(metric_name in m and len(m[metric_name]) > 1 for m in convergence_data.values())

    # Use convergence panels if any are available, otherwise fall back
    panels = [(m, t, y) for m, t, y in convergence_panels if _has_data(m)]
    if not panels:
        panels = [(m, t, y) for m, t, y in fallback_panels if _has_data(m)]

    if not panels:
        logger.warning("No convergence metrics with enough data points to plot")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    for idx, (metric, title, ylabel) in enumerate(panels):
        ax = axes[0][idx]

        for scenario_name, scenario_metrics in convergence_data.items():
            if metric not in scenario_metrics:
                continue

            series = scenario_metrics[metric]
            valid = [e for e in series if e.get("mean") is not None]
            if not valid:
                continue

            steps = np.array([e["step"] for e in valid])
            means = np.array([e["mean"] for e in valid])
            stds = np.array([e.get("std", 0.0) for e in valid])
            mins = np.array([e.get("min", e["mean"]) for e in valid])
            maxs = np.array([e.get("max", e["mean"]) for e in valid])

            color = _get_color(scenario_name)
            label = _get_label(scenario_name)

            ax.plot(steps, means, color=color, marker="o", markersize=3, linewidth=1.5, label=label)
            if np.any(stds > 0):
                band_lo, band_hi = _clip_band(means - stds, means + stds, metric)
                ax.fill_between(steps, band_lo, band_hi, color=color, alpha=0.15)

            # Show min/max range when there is notable spread
            if not np.allclose(mins, maxs):
                min_clipped, max_clipped = _clip_band(mins, maxs, metric)
                ax.fill_between(steps, min_clipped, max_clipped, color=color, alpha=0.06)

        ax.set_xlabel("Training Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set axis limits based on metric type
        if metric in _NONNEG_METRICS:
            ax.set_ylim(bottom=0)
        if metric in _BOUNDED_01_METRICS:
            ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved convergence model metrics to {output_path}")


# ============================================================================
# Table Generation: Full Metrics Comparison
# ============================================================================


def generate_full_metrics_table(
    results: pd.DataFrame,
    output_path: str = "full_metrics_table.tex",
    metrics: list[str] | None = None,
):
    """
    Generate a comprehensive LaTeX table with F1, Accuracy, Precision, Recall (all with +/- std).
    Reports both mean and median when they differ significantly (attacker present).
    """
    if metrics is None:
        metrics = ["F1", "Accuracy", "Precision", "Recall"]

    available = [m for m in metrics if m in results["metric"].values]
    if not available:
        logger.warning("No model metrics found for table generation")
        return

    scenarios = results["scenario"].unique()
    col_spec = "l" + "c" * len(available)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comprehensive model performance (mean $\\pm$ std across nodes; median in parentheses when different)}",
        "\\label{tab:full_metrics}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Configuration & " + " & ".join(available) + " \\\\",
        "\\midrule",
    ]

    for scenario in scenarios:
        row = [scenario.replace("_", "\\_")]
        for metric in available:
            match = results[(results["scenario"] == scenario) & (results["metric"] == metric)]
            if not match.empty:
                mean = match.iloc[0]["mean"]
                std = match.iloc[0]["std"]
                median = match.iloc[0].get("median", mean)
                if std > 0 and not np.isnan(std):
                    cell = f"${mean:.4f} \\pm {std:.4f}$"
                    # Show median when it differs notably from mean
                    if median is not None and not np.isnan(median) and abs(mean - median) > 0.05:
                        cell += f" (med: ${median:.4f}$)"
                    row.append(cell)
                else:
                    row.append(f"${mean:.4f}$")
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
    logger.info(f"Saved LaTeX table to {output_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="RADAR Paper Plot Generator")
    parser.add_argument(
        "--input",
        type=str,
        default="analysis/results",
        help="Input directory (from radar_metrics_aggregator.py output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/figures",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--convergence-file",
        type=str,
        help="Direct path to convergence_curves.json (overrides --input)",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        help="Direct path to results_summary.csv (overrides --input)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Resolve file paths
    convergence_file = args.convergence_file or os.path.join(args.input, "convergence_curves.json")
    summary_file = args.summary_file or os.path.join(args.input, "results_summary.csv")

    fmt = args.format

    # Load data
    convergence_data = None
    results_df = None

    if os.path.exists(convergence_file):
        convergence_data = load_convergence_data(convergence_file)
        logger.info(f"Loaded convergence data: {len(convergence_data)} scenarios")
    else:
        logger.warning(f"Convergence file not found: {convergence_file}")

    if os.path.exists(summary_file):
        results_df = load_results_summary(summary_file)
        logger.info(f"Loaded results summary: {len(results_df)} rows")
    else:
        logger.warning(f"Summary file not found: {summary_file}")

    # Generate figures
    if convergence_data:
        # 1. Convergence curves (F1)
        plot_convergence_curves(
            convergence_data,
            metric="F1",
            output_path=os.path.join(args.output, f"convergence_f1.{fmt}"),
        )

        # 2. Convergence curves (Accuracy)
        plot_convergence_curves(
            convergence_data,
            metric="Accuracy",
            output_path=os.path.join(args.output, f"convergence_accuracy.{fmt}"),
        )

        # 3. Model convergence metrics (norm, delta)
        plot_convergence_metrics(
            convergence_data,
            output_path=os.path.join(args.output, f"convergence_model_metrics.{fmt}"),
        )

        # 4. Resource usage over time
        plot_resource_over_time(
            convergence_data,
            output_path=os.path.join(args.output, f"resource_usage.{fmt}"),
        )

        # 5. Aggregation behavior (for first Balance/RADAR scenario found)
        for scenario_name in convergence_data:
            if any(kw in scenario_name.lower() for kw in ["radar", "balance"]):
                plot_aggregation_behavior(
                    convergence_data,
                    scenario_name,
                    output_path=os.path.join(args.output, f"aggregation_behavior.{fmt}"),
                )
                break

    if results_df is not None:
        # 6. Bar charts with error bars (multi-scenario comparison)
        plot_bar_charts_with_errors(
            results_df,
            output_path=os.path.join(args.output, f"bar_charts.{fmt}"),
        )

        # 7. Full metrics LaTeX table
        generate_full_metrics_table(
            results_df,
            output_path=os.path.join(args.output, "full_metrics_table.tex"),
        )

    if convergence_data is None and results_df is None:
        logger.error("No input data found. Run radar_metrics_aggregator.py first.")
        sys.exit(1)

    logger.info(f"All figures saved to {args.output}")


if __name__ == "__main__":
    main()
