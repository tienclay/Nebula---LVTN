"""
Parameter Sensitivity Plot (Balance / RADAR-Agg)

Generates line plots showing how Balance aggregator performance varies with
parameter sweeps (gamma, kappa, alpha) from Group D experiments.

Layout: 1x3 subplots (gamma sweep, kappa sweep, alpha sweep), each showing
two lines (No Attack dashed, Gaussian solid) in RADAR-Agg color.

Default Balance parameters (from core Group A experiments without explicit
param keys): gamma=1.5, kappa=1.0, alpha=0.5.

Usage:
    python -m analysis.plot_parameter_sensitivity --root-dir /path/to/results
    python -m analysis.plot_parameter_sensitivity --root-dir /path/to/results --metric Accuracy --format png
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

# Default Balance parameters (used when no explicit param key in scenario name)
DEFAULT_GAMMA = 1.5
DEFAULT_KAPPA = 1.0
DEFAULT_ALPHA = 0.5

# Parameter display names and Greek letters
PARAM_CONFIGS = [
    ("gamma", r"$\gamma$", DEFAULT_GAMMA),
    ("kappa", r"$\kappa$", DEFAULT_KAPPA),
    ("alpha", r"$\alpha$", DEFAULT_ALPHA),
]

# RADAR-Agg color for both lines
RADAR_COLOR = "#9467bd"


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Plot (Group D)")
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

    # Filter to Balance/RADAR-Agg scenarios only
    balance_scenarios = [sc for sc in all_scenarios if sc["aggregator"] == "balance"]

    # Attacks used in parameter sensitivity analysis
    sensitivity_attacks = ["No Attack", "Gaussian"]

    # ── Collect parameter sweep data ────────────────────────────────────────
    # Structure: {param_name: {attack_label: {param_value: (mean, std)}}}
    sweep_data: dict[str, dict[str, dict[float, tuple[float, float]]]] = {
        "gamma": {},
        "kappa": {},
        "alpha": {},
    }

    for sc in balance_scenarios:
        attack_label = sc["attack_label"]
        if attack_label not in sensitivity_attacks:
            continue

        # Load results
        _, results_df = load_scenario_results(sc["scenario_dir"])
        mean, std = get_final_metric(results_df, metric)
        if mean is None:
            continue

        has_explicit_param = any(k in sc for k in ("gamma", "kappa", "alpha"))

        if has_explicit_param:
            # This scenario has an explicit parameter sweep value
            for param_name in ("gamma", "kappa", "alpha"):
                if param_name in sc:
                    val = sc[param_name]
                    sweep_data[param_name].setdefault(attack_label, {})[val] = (mean, std)
        else:
            # Default Balance scenario (no param keys) — contributes default
            # values to ALL parameter sweeps as the baseline point
            # Skip if it has n_nodes (scalability) or non-erdosrenyi topology
            if "n_nodes" in sc or sc["topology"] != "erdosrenyi":
                continue

            for param_name, _, default_val in PARAM_CONFIGS:
                sweep_data[param_name].setdefault(attack_label, {})[default_val] = (mean, std)

    # Check if we have any parameter sweep data
    has_sweep_data = any(any(len(vals) > 0 for vals in attacks.values()) for attacks in sweep_data.values())

    if not has_sweep_data:
        logger.warning("No parameter sweep data found. Generating placeholder plot.")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        for idx, (param_name, param_symbol, _) in enumerate(PARAM_CONFIGS):  # noqa: B007
            ax = axes[0][idx]
            ax.set_title(f"Sensitivity to {param_symbol}")
            ax.set_xlabel(param_symbol)
            ax.set_ylabel(f"Final {metric}")
            ax.text(
                0.5,
                0.5,
                "No parameter sweep data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                fontstyle="italic",
            )
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"parameter_sensitivity_{metric.lower()}.{fmt}")
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved placeholder to {out_path}")
        return

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    for idx, (param_name, param_symbol, default_val) in enumerate(PARAM_CONFIGS):
        ax = axes[0][idx]
        param_data = sweep_data[param_name]

        if not param_data:
            ax.set_title(f"Sensitivity to {param_symbol}")
            ax.set_xlabel(param_symbol)
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

        for attack_label in sensitivity_attacks:
            if attack_label not in param_data:
                continue

            vals_dict = param_data[attack_label]
            sorted_vals = sorted(vals_dict.keys())
            x_vals = np.array(sorted_vals)
            means = np.array([vals_dict[v][0] for v in sorted_vals])
            stds = np.array([vals_dict[v][1] for v in sorted_vals])

            if attack_label == "No Attack":
                linestyle = "--"
                marker = "o"
            else:
                linestyle = "-"
                marker = "s"

            ax.errorbar(
                x_vals,
                means,
                yerr=stds,
                color=RADAR_COLOR,
                marker=marker,
                markersize=6,
                linestyle=linestyle,
                linewidth=1.5,
                capsize=3,
                label=attack_label,
            )

        ax.set_title(f"Sensitivity to {param_symbol}")
        ax.set_xlabel(param_symbol)
        ax.set_ylabel(f"Final {metric}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Mark default value with a vertical line
        ax.axvline(x=default_val, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"parameter_sensitivity_{metric.lower()}.{fmt}")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved parameter sensitivity plot to {out_path}")


if __name__ == "__main__":
    main()
