"""
RADAR Comparison Pipeline

Master orchestrator that runs all comparison/visualization scripts on a given
experiment root directory.  Detects which experiment groups are present and
runs only the applicable scripts.

Usage:
    python analysis/run_comparison_pipeline.py --root-dir app/logs/smoke_core_20260311
    python analysis/run_comparison_pipeline.py --root-dir app/logs/full_run --skip spider_chart topology_impact
"""

import argparse
import logging
import os
import subprocess
import sys
import time

from analysis.scenario_metadata import scan_scenario_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Script registry: (name, module_path, required_group, description)
# required_group: "core" = needs core data, "malicious_ratio" = needs Group B, etc.
# None means always applicable (uses whatever data is available)
SCRIPTS = [
    # Core comparison scripts
    ("f1_heatmap", "analysis/plot_f1_heatmap.py", "core", "F1 Heatmap (Aggregator x Attack)"),
    ("comparative_convergence", "analysis/plot_comparative_convergence.py", "core", "Comparative Convergence Curves"),
    ("spider_chart", "analysis/plot_spider_chart.py", "core", "Spider/Radar Charts"),
    # Reviewer-requested scripts
    ("node_distribution", "analysis/plot_node_distribution.py", "core", "Per-Node Distribution (Violin/Box)"),
    ("statistical_tests", "analysis/plot_statistical_tests.py", "core", "Statistical Significance Tests"),
    ("attack_degradation", "analysis/plot_attack_degradation.py", "core", "Attack Degradation Analysis"),
    (
        "convergence_speed",
        "analysis/plot_convergence_speed.py",
        "core",
        "Convergence Speed (AUC + Rounds-to-threshold)",
    ),
    # Group-specific scripts
    ("malicious_ratio", "analysis/plot_malicious_ratio.py", None, "Malicious Ratio Sweep"),
    ("topology_impact", "analysis/plot_topology_impact.py", None, "Topology Impact"),
    ("scalability", "analysis/plot_scalability.py", None, "Scalability vs Node Count"),
    ("parameter_sensitivity", "analysis/plot_parameter_sensitivity.py", None, "Parameter Sensitivity"),
]


def _detect_groups(scenarios: list[dict]) -> set[str]:
    """Detect which experiment groups are present in the data."""
    groups = set()

    for meta in scenarios:
        topo = meta.get("topology")
        has_gamma = "gamma" in meta
        has_kappa = "kappa" in meta
        has_alpha = "alpha" in meta
        has_nodes = "n_nodes" in meta

        if has_gamma or has_kappa or has_alpha:
            groups.add("parameter")
        elif has_nodes:
            groups.add("scalability")
        elif topo == "erdosrenyi":
            groups.add("core")
            # Could also be malicious_ratio — check for unusual percentages
            pct = meta.get("n_malicious_pct", 0)
            if pct not in (0, 20, 60):
                groups.add("malicious_ratio")
        elif topo in ("dense", "ring"):
            groups.add("topology")

    return groups


def run_script(script_path: str, root_dir: str, output_dir: str, fmt: str) -> bool:
    """Run a comparison script as a subprocess. Returns True on success."""
    cmd = [
        sys.executable,
        script_path,
        "--root-dir",
        root_dir,
        "--output-dir",
        output_dir,
        "--format",
        fmt,
    ]
    logger.info(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    # Ensure project root is on PYTHONPATH for `from analysis.* import ...`
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603
    elapsed = time.time() - t0

    if result.returncode == 0:
        logger.info(f"  Done ({elapsed:.1f}s)")
        return True
    else:
        logger.error(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.error(f"    {line}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all RADAR comparison/visualization scripts")
    parser.add_argument("--root-dir", required=True, help="Root directory containing scenario subdirectories")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {root-dir}/comparison_figures)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure format")
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Script names to skip (e.g., spider_chart topology_impact)"
    )
    parser.add_argument("--only", nargs="*", default=[], help="Only run these scripts (e.g., f1_heatmap)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.root_dir, "comparison_figures")
    os.makedirs(output_dir, exist_ok=True)

    # Scan scenarios
    all_scenarios = scan_scenario_root(args.root_dir)
    if not all_scenarios:
        logger.error("No valid scenarios found in --root-dir")
        sys.exit(1)

    groups = _detect_groups(all_scenarios)
    logger.info(f"Found {len(all_scenarios)} scenarios, groups: {groups}")

    # Run scripts
    skip_set = set(args.skip)
    only_set = set(args.only) if args.only else None
    succeeded = 0
    failed = 0
    skipped = 0

    for name, script_path, required_group, description in SCRIPTS:
        if name in skip_set:
            logger.info(f"[SKIP] {description} (--skip)")
            skipped += 1
            continue

        if only_set and name not in only_set:
            logger.info(f"[SKIP] {description} (not in --only)")
            skipped += 1
            continue

        if required_group and required_group not in groups:
            logger.info(f"[SKIP] {description} (needs '{required_group}' data)")
            skipped += 1
            continue

        logger.info(f"[RUN]  {description}")
        if run_script(script_path, args.root_dir, output_dir, args.format):
            succeeded += 1
        else:
            failed += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Pipeline complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
