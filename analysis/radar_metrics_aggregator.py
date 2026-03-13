"""
RADAR Paper Metrics Aggregator

Extracts and aggregates metrics from Nebula scenario runs for paper reporting.
Reads per-participant TensorBoard event files, identifies malicious nodes from
config, and computes benign-only statistics (mean, std, min, max, median).

Output:
  - results_summary.csv: Final-round metrics with mean +/- std +/- 95% CI (benign only)
  - convergence_curves.json: Per-round metrics for convergence plots (benign only)
  - latex_tables.tex: Auto-generated LaTeX tables with +/- std format

Usage:
    python analysis/radar_metrics_aggregator.py --scenario-dir /path/to/scenario_output
    python analysis/radar_metrics_aggregator.py --scenarios-root /path/to/all_scenarios --output-dir analysis/results/

The scenario directory should contain:
  - metrics/participant_N/ directories with TensorBoard event files
  - A config directory (auto-discovered) with participant_N.json files
"""

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Metrics tags in TensorBoard event files
MODEL_METRIC_TAGS = {
    "F1": "Test (Global)/F1Score",
    "Accuracy": "Test (Global)/Accuracy",
    "Precision": "Test (Global)/Precision",
    "Recall": "Test (Global)/Recall",
    "Loss": "Train/Loss",
}

RESOURCE_METRIC_TAGS = {
    "CPU (%)": "W-CPU/CPU global (%)",
    "RAM (%)": "Z-RAM/RAM global (%)",
    "RAM (MB)": "Z-RAM/RAM global (MB)",
    "Network (bytes sent)": "X-Network/Network (bytes sent)",
    "Network (bytes received)": "X-Network/Network (bytes received)",
    "Network (packets sent)": "X-Network/Network (packets sent)",
    "Network (packets received)": "X-Network/Network (packets received)",
}

AGGREGATION_METRIC_TAGS = {
    "Threshold": "Aggregation/Threshold",
    "SimilarNeighbors": "Aggregation/SimilarNeighborsCount",
    "TotalNeighbors": "Aggregation/TotalNeighbors",
    "AvgNeighborDist": "Aggregation/AvgNeighborDistance",
    "LocalModelNorm": "Aggregation/LocalModelNorm",
    "NeighborAcceptRate": "Aggregation/NeighborAcceptRate",
}

CONVERGENCE_METRIC_TAGS = {
    "GlobalModelNorm": "Convergence/GlobalModelNorm",
    "ModelNormDelta": "Convergence/ModelNormDelta",
    "CosineSimilarity": "Convergence/CosineSimilarity",
}

ALL_METRIC_TAGS = {
    **MODEL_METRIC_TAGS,
    **RESOURCE_METRIC_TAGS,
    **AGGREGATION_METRIC_TAGS,
    **CONVERGENCE_METRIC_TAGS,
}


# ============================================================================
# Malicious node detection
# ============================================================================


def find_malicious_participants(scenario_dir: str) -> set[int]:
    """Identify malicious participant indices from config JSONs.

    Searches for participant config files and checks adversarial_args.attacks.
    Returns set of participant indices that are malicious.
    """
    malicious = set()

    # Try multiple config directory locations
    scenario_name = os.path.basename(scenario_dir)
    config_candidates = [
        os.path.join(os.path.dirname(scenario_dir), "..", "config", scenario_name),
        os.path.join(scenario_dir, "config"),
        os.path.join(scenario_dir),
    ]

    config_dir = None
    for candidate in config_candidates:
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "participant_*.json")):
            config_dir = candidate
            break

    if config_dir is None:
        logger.warning("No participant config directory found; assuming all nodes are benign")
        return malicious

    for cfg_path in sorted(glob.glob(os.path.join(config_dir, "participant_*.json"))):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            idx = int(os.path.basename(cfg_path).split("_")[1].split(".")[0])
            attacks = cfg.get("adversarial_args", {}).get("attacks", "No Attack")
            if attacks and attacks.lower() not in ("no attack", "none", ""):
                malicious.add(idx)
                logger.info(f"  Participant {idx}: MALICIOUS ({attacks})")
            else:
                logger.info(f"  Participant {idx}: benign")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse config {cfg_path}: {e}")

    return malicious


# ============================================================================
# Per-participant TensorBoard reading
# ============================================================================


def read_participant_scalars(metrics_dir: str, participant_idx: int) -> dict[str, list[tuple[int, float]]]:
    """Read scalar metrics from a single participant's TensorBoard event files.

    Returns {tag: [(step, value), ...]} for all available scalar tags.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    participant_dir = os.path.join(metrics_dir, f"participant_{participant_idx}")
    if not os.path.isdir(participant_dir):
        return {}

    ea = EventAccumulator(participant_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def aggregate_benign_metrics(  # noqa: C901
    metrics_dir: str,
    malicious_indices: set[int],
) -> tuple[dict, dict, int]:
    """Read per-participant TF events, exclude malicious, compute benign-only stats.

    Returns:
        convergence: {metric_name: [{step, mean, std, min, max, median}]}
        final_metrics: {metric_name: {mean, std, min, max, median, ci_lo, ci_hi, n_nodes}}
        n_benign: number of benign nodes
    """
    # Discover participant directories
    participant_dirs = sorted(glob.glob(os.path.join(metrics_dir, "participant_*")))
    all_indices = []
    for d in participant_dirs:
        try:
            idx = int(os.path.basename(d).split("_")[1])
            all_indices.append(idx)
        except (ValueError, IndexError):
            pass

    benign_indices = [i for i in all_indices if i not in malicious_indices]
    n_benign = len(benign_indices)

    if n_benign == 0:
        logger.warning("No benign participants found!")
        return {}, {}, 0

    logger.info(f"Aggregating {n_benign} benign nodes: {benign_indices} (excluding malicious: {malicious_indices})")

    # Read all benign participants' data
    # Structure: {tb_tag: {participant_idx: [(step, value), ...]}}
    all_data: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for idx in benign_indices:
        scalars = read_participant_scalars(metrics_dir, idx)
        for tag, values in scalars.items():
            if tag not in all_data:
                all_data[tag] = {}
            all_data[tag][idx] = values

    # Compute per-round benign-only statistics
    convergence = {}
    final_metrics = {}
    z_95 = 1.96

    for name, tb_tag in ALL_METRIC_TAGS.items():
        if tb_tag not in all_data:
            continue

        tag_data = all_data[tb_tag]  # {idx: [(step, value), ...]}

        # Find max number of steps across benign nodes
        max_steps = max((len(v) for v in tag_data.values()), default=0)
        if max_steps == 0:
            continue

        series = []
        for step in range(max_steps):
            # Collect values at this step from all benign nodes that have it
            values = []
            for idx in benign_indices:
                if idx in tag_data and step < len(tag_data[idx]):
                    values.append(tag_data[idx][step][1])

            if not values:
                continue

            arr = np.array(values)
            entry = {
                "step": step,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }
            series.append(entry)

        if series:
            convergence[name] = series

            # Final-round metrics (last step)
            last = series[-1]
            ci_half = z_95 * last["std"] / math.sqrt(n_benign) if n_benign > 1 else 0.0
            final_metrics[name] = {
                "mean": last["mean"],
                "std": last["std"],
                "min": last["min"],
                "max": last["max"],
                "median": last["median"],
                "ci_lo": last["mean"] - ci_half,
                "ci_hi": last["mean"] + ci_half,
                "n_nodes": n_benign,
            }

    return convergence, final_metrics, n_benign


# ============================================================================
# Legacy CSV-based extraction (fallback)
# ============================================================================


def load_reduced_csv(csv_path: str) -> pd.DataFrame:
    """Load tensorboard_reducer CSV output."""
    if not os.path.exists(csv_path):
        msg = f"CSV not found: {csv_path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"Loaded {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def find_column(df: pd.DataFrame, tag: str, op: str) -> str | None:
    """Find a column matching tag/op pattern in the CSV."""
    _OP_SUFFIX = {"mean": "", "min": ".1", "max": ".2", "median": ".3", "std": ".4", "var": ".5"}
    suffix = _OP_SUFFIX.get(op, "")
    candidate = f"{tag}{suffix}" if suffix else tag
    if candidate in df.columns:
        return candidate
    for c in [f"{tag}/{op}", f"{tag}-{op}"]:
        if c in df.columns:
            return c
    for col in df.columns:
        if tag in col and op in col:
            return col
    return None


def extract_final_round_metrics(df: pd.DataFrame, n_honest_nodes: int = 20) -> dict:
    """Extract final-round metrics from reduced CSV (legacy, includes all nodes)."""
    if df.empty:
        return {}

    z_95 = 1.96
    results = {}
    all_tags = {**MODEL_METRIC_TAGS, **RESOURCE_METRIC_TAGS}

    for name, tag in all_tags.items():
        mean_col = find_column(df, tag, "mean")
        std_col = find_column(df, tag, "std")
        min_col = find_column(df, tag, "min")
        max_col = find_column(df, tag, "max")
        median_col = find_column(df, tag, "median")

        if mean_col is None:
            continue

        mean_series = df[mean_col].dropna()
        if mean_series.empty:
            continue
        last_idx = mean_series.index[-1]

        mean_val = float(df.loc[last_idx, mean_col])
        std_val = float(df.loc[last_idx, std_col]) if std_col and not pd.isna(df.loc[last_idx, std_col]) else 0.0
        min_val = float(df.loc[last_idx, min_col]) if min_col and not pd.isna(df.loc[last_idx, min_col]) else mean_val
        max_val = float(df.loc[last_idx, max_col]) if max_col and not pd.isna(df.loc[last_idx, max_col]) else mean_val
        median_val = (
            float(df.loc[last_idx, median_col])
            if median_col and not pd.isna(df.loc[last_idx, median_col])
            else mean_val
        )

        ci_half = z_95 * std_val / math.sqrt(n_honest_nodes) if n_honest_nodes > 0 else 0.0

        results[name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": median_val,
            "ci_lo": mean_val - ci_half,
            "ci_hi": mean_val + ci_half,
            "n_nodes": n_honest_nodes,
        }

    return results


def extract_convergence_series(df: pd.DataFrame) -> dict:
    """Extract per-round convergence data from reduced CSV (legacy, includes all nodes)."""
    convergence = {}
    all_tags = {**MODEL_METRIC_TAGS, **RESOURCE_METRIC_TAGS, **AGGREGATION_METRIC_TAGS, **CONVERGENCE_METRIC_TAGS}

    for name, tag in all_tags.items():
        mean_col = find_column(df, tag, "mean")
        std_col = find_column(df, tag, "std")
        min_col = find_column(df, tag, "min")
        max_col = find_column(df, tag, "max")

        if mean_col is None:
            continue

        median_col = find_column(df, tag, "median")

        series = []
        seq_idx = 0
        for _row_idx, row in df.iterrows():
            if pd.isna(row[mean_col]):
                continue
            entry = {
                "step": int(seq_idx),
                "mean": float(row[mean_col]),
            }
            seq_idx += 1
            if std_col:
                entry["std"] = float(row[std_col]) if not pd.isna(row[std_col]) else 0.0
            if min_col:
                entry["min"] = float(row[min_col]) if not pd.isna(row[min_col]) else None
            if max_col:
                entry["max"] = float(row[max_col]) if not pd.isna(row[max_col]) else None
            if median_col:
                entry["median"] = float(row[median_col]) if not pd.isna(row[median_col]) else None
            series.append(entry)

        convergence[name] = series

    return convergence


# ============================================================================
# LaTeX generation
# ============================================================================


def generate_latex_value(mean: float, std: float, precision: int = 4) -> str:
    """Format a value as mean +/- std for LaTeX."""
    if std == 0 or np.isnan(std):
        return f"{mean:.{precision}f}"
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def generate_latex_table(
    scenarios: dict[str, dict],
    metrics: list[str],
    caption: str = "Performance comparison",
    label: str = "tab:comparison",
) -> str:
    """Generate a LaTeX table from multiple scenario results."""
    col_spec = "l" + "c" * len(metrics)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Scenario & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]

    for scenario_name, results in scenarios.items():
        row_values = [scenario_name]
        for metric in metrics:
            if metric in results:
                val = generate_latex_value(results[metric]["mean"], results[metric]["std"])
                row_values.append(val)
            else:
                row_values.append("--")
        lines.append(" & ".join(row_values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ============================================================================
# Scenario processing
# ============================================================================


def process_scenario(scenario_dir: str, n_honest_nodes: int = 20) -> dict:
    """Process a single scenario directory.

    Prefers per-participant TF event reading (benign-only) over legacy CSV.
    Falls back to reduced CSV if TF events are unavailable.
    """
    metrics_dir = os.path.join(scenario_dir, "metrics")

    # Check if per-participant TF event directories exist
    participant_dirs = glob.glob(os.path.join(metrics_dir, "participant_*"))
    has_tf_events = any(glob.glob(os.path.join(d, "events.out.tfevents.*")) for d in participant_dirs)

    if has_tf_events:
        logger.info("Using per-participant TensorBoard events (benign-only aggregation)")
        malicious = find_malicious_participants(scenario_dir)
        convergence, final_metrics, n_benign = aggregate_benign_metrics(metrics_dir, malicious)

        if convergence:
            return {
                "final_metrics": final_metrics,
                "convergence": convergence,
                "n_rows": max(len(v) for v in convergence.values()),
                "n_benign": n_benign,
                "malicious_indices": malicious,
            }

    # Fallback to legacy CSV
    csv_path = os.path.join(metrics_dir, "reduced-data-as.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"No metrics data found in {scenario_dir}, skipping")
        return {}

    logger.info("Falling back to reduced CSV (includes all nodes)")
    df = load_reduced_csv(csv_path)
    final_metrics = extract_final_round_metrics(df, n_honest_nodes)
    convergence = extract_convergence_series(df)

    return {
        "final_metrics": final_metrics,
        "convergence": convergence,
        "csv_path": csv_path,
        "n_rows": len(df),
    }


def process_all_scenarios(scenarios_root: str, output_dir: str, n_honest_nodes: int = 20):
    """Process all scenario directories under a root path."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all scenario directories (those containing metrics/)
    scenario_dirs = {}
    root = Path(scenarios_root)
    for metrics_dir in root.rglob("metrics"):
        if metrics_dir.is_dir():
            scenario_dir = metrics_dir.parent
            scenario_name = scenario_dir.name
            # Ensure it has participant data or reduced CSV
            has_data = list(metrics_dir.glob("participant_*")) or (metrics_dir / "reduced-data-as.csv").exists()
            if has_data:
                scenario_dirs[scenario_name] = str(scenario_dir)

    if not scenario_dirs:
        logger.error(f"No scenarios found under {scenarios_root}")
        return

    logger.info(f"Found {len(scenario_dirs)} scenarios: {list(scenario_dirs.keys())}")

    all_results = {}
    all_convergence = {}
    for name, path in sorted(scenario_dirs.items()):
        logger.info(f"Processing scenario: {name}")
        result = process_scenario(path, n_honest_nodes)
        if result:
            all_results[name] = result["final_metrics"]
            all_convergence[name] = result["convergence"]

    # 1. Save results summary CSV
    summary_rows = []
    for scenario, metrics in all_results.items():
        for metric_name, values in metrics.items():
            summary_rows.append({
                "scenario": scenario,
                "metric": metric_name,
                **values,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved results summary to {summary_path}")

    # 2. Save convergence data as JSON
    convergence_path = os.path.join(output_dir, "convergence_curves.json")
    with open(convergence_path, "w") as f:
        json.dump(all_convergence, f, indent=2)
    logger.info(f"Saved convergence data to {convergence_path}")

    # 3. Generate LaTeX tables
    model_metrics = ["F1", "Accuracy", "Precision", "Recall"]
    available_model_metrics = [m for m in model_metrics if any(m in r for r in all_results.values())]

    if available_model_metrics and all_results:
        latex = generate_latex_table(
            all_results,
            available_model_metrics,
            caption="Model performance comparison (mean $\\pm$ std across benign nodes)",
            label="tab:model_performance",
        )
        latex_path = os.path.join(output_dir, "latex_tables.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        logger.info(f"Saved LaTeX tables to {latex_path}")

    logger.info("Processing complete.")
    return all_results, all_convergence


def main():
    parser = argparse.ArgumentParser(description="RADAR Paper Metrics Aggregator")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        help="Path to a single scenario output directory",
    )
    parser.add_argument(
        "--scenarios-root",
        type=str,
        help="Path to root directory containing multiple scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/results",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--n-honest-nodes",
        type=int,
        default=20,
        help="Number of honest nodes (for CI computation, legacy CSV fallback). Default: 20",
    )
    args = parser.parse_args()

    if args.scenario_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result = process_scenario(args.scenario_dir, args.n_honest_nodes)
        if result:
            summary_rows = []
            scenario_name = os.path.basename(args.scenario_dir)
            for metric_name, values in result["final_metrics"].items():
                summary_rows.append({"scenario": scenario_name, "metric": metric_name, **values})
            pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "results_summary.csv"), index=False)
            with open(os.path.join(args.output_dir, "convergence_curves.json"), "w") as f:
                json.dump({scenario_name: result["convergence"]}, f, indent=2)
            logger.info("Single scenario processing complete.")
    elif args.scenarios_root:
        process_all_scenarios(args.scenarios_root, args.output_dir, args.n_honest_nodes)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
