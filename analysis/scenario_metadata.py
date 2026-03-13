"""
RADAR Scenario Metadata Parser

Foundation library module used by all comparison scripts.
Parses scenario directory names into structured metadata dicts based on the
naming convention:  {aggregator}_{attack}_{topology}_{n_malicious}pct

Special variants:
  balance_gamma05_gaussian_erdosrenyi_20pct   ← parameter sweep (gamma=0.5)
  balance_kappa10_gaussian_erdosrenyi_20pct   ← parameter sweep (kappa=1.0)
  balance_alpha05_gaussian_erdosrenyi_20pct   ← parameter sweep (alpha=0.5)
  balance_nodes50_gaussian_erdosrenyi_20pct   ← scalability sweep (50 nodes)

NOT a CLI script — pure library imported by plot_*.py and run_comparison_pipeline.py.
"""

import json
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Canonical alias tables
# ============================================================================

AGGREGATOR_ALIASES: dict[str, str] = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "trimmedmean": "TrimmedMean",
    "median": "Median",
    "balance": "RADAR-Agg",
}

ATTACK_ALIASES: dict[str, str] = {
    "noattack": "No Attack",
    "gaussian": "Gaussian",
    "noise": "Noise Injection",
    "weightswap": "Weight Swapping",
    "krum_attack": "Krum Attack",
    # Short form aliases (no underscore) for folder name compatibility
    "krumattack": "Krum Attack",
    "trimmean_attack": "Trim-Mean Attack",
    "trimmeanattack": "Trim-Mean Attack",
    "trimmean": "Trim-Mean Attack",
    "labelflip": "Label Flip",
    "backdoor": "Backdoor",
}

TOPOLOGY_ALIASES: dict[str, str] = {
    "erdosrenyi": "Erdős-Rényi",
    "ring": "Ring",
    "dense": "Dense",
    "star": "Star",
    "random": "Random",
    "fullymeshed": "Fully Meshed",
    "barabasi": "Barabási-Albert",
}

# Ordered by length descending so longer names are matched first
_KNOWN_AGGREGATORS: list[str] = sorted(AGGREGATOR_ALIASES.keys(), key=len, reverse=True)
_KNOWN_ATTACKS_COMPOUND: list[str] = sorted([k for k in ATTACK_ALIASES if "_" in k], key=len, reverse=True)
_KNOWN_ATTACKS_SIMPLE: list[str] = sorted([k for k in ATTACK_ALIASES if "_" not in k], key=len, reverse=True)
_KNOWN_TOPOLOGIES: list[str] = sorted(TOPOLOGY_ALIASES.keys(), key=len, reverse=True)

# Regex to detect special parameter/scalability tokens
_PARAM_RE = re.compile(r"^(gamma|kappa|alpha|nodes)(\d+)$")


def _parse_param_value(name: str, raw: str) -> float | int:
    """Convert raw digit string to numeric: 'nodes' → int, others → float (/10)."""
    if name == "nodes":
        return int(raw)
    return int(raw) / 10


# ============================================================================
# parse_scenario_name
# ============================================================================


def parse_scenario_name(name: str) -> dict | None:  # noqa: C901
    """Parse a scenario directory name into a metadata dict.

    Expected format: {aggregator}_{attack}_{topology}_{n_malicious}pct

    Returns None (with a warning) if the name does not match the convention.

    Returned dict keys:
        scenario_name       str   — original name
        aggregator          str   — lower-case key, e.g. "balance"
        aggregator_label    str   — display name, e.g. "RADAR-Agg"
        attack              str   — lower-case key, e.g. "gaussian"
        attack_label        str   — display name, e.g. "Gaussian"
        topology            str   — lower-case key, e.g. "erdosrenyi"
        topology_label      str   — display name, e.g. "Erdős-Rényi"
        n_malicious_pct     int   — e.g. 20
        gamma               float — only if present in name
        kappa               float — only if present in name
        alpha               float — only if present in name
        n_nodes             int   — only if present in name
    """
    s = name.lower()

    # ── Step 1: strip _Npct suffix ────────────────────────────────────────────
    pct_match = re.search(r"_(\d+)pct$", s)
    if not pct_match:
        return None
    n_malicious_pct = int(pct_match.group(1))
    s = s[: pct_match.start()]

    # ── Step 2: match topology at end ─────────────────────────────────────────
    topology = None
    for topo in _KNOWN_TOPOLOGIES:
        if s.endswith("_" + topo):
            topology = topo
            s = s[: -(len(topo) + 1)]
            break
    if topology is None:
        return None

    # ── Step 3: match aggregator at start ─────────────────────────────────────
    aggregator = None
    for agg in _KNOWN_AGGREGATORS:
        if s == agg or s.startswith(agg + "_"):
            aggregator = agg
            s = s[len(agg) :]
            s = s.lstrip("_")
            break
    if aggregator is None:
        return None

    # ── Step 4: extract special parameter/scalability tokens ──────────────────
    gamma = kappa = alpha = n_nodes = None
    # Tokens are separated by '_'; scan left-to-right
    tokens: list[str] = s.split("_") if s else []
    remaining_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        m = _PARAM_RE.match(tokens[i])
        if m:
            pname, praw = m.group(1), m.group(2)
            val = _parse_param_value(pname, praw)
            if pname == "gamma":
                gamma = val
            elif pname == "kappa":
                kappa = val
            elif pname == "alpha":
                alpha = val
            elif pname == "nodes":
                n_nodes = int(val)
        else:
            remaining_tokens.append(tokens[i])
        i += 1

    # ── Step 5: remaining tokens form the attack name ─────────────────────────
    attack_str = "_".join(remaining_tokens).strip("_")
    if not attack_str:
        return None

    # Normalise: try compound attacks first (e.g. krum_attack, trimmean_attack)
    attack = None
    for compound in _KNOWN_ATTACKS_COMPOUND:
        if attack_str == compound or attack_str.startswith(compound):
            attack = compound
            break
    if attack is None:
        # Try simple attacks
        for simple in _KNOWN_ATTACKS_SIMPLE:
            if attack_str == simple:
                attack = simple
                break
    if attack is None:
        # Unknown attack — store as-is but warn
        logger.debug(f"Unknown attack token '{attack_str}' in scenario '{name}'; using as-is")
        attack = attack_str

    # ── Build result ──────────────────────────────────────────────────────────
    result: dict = {
        "scenario_name": name,
        "aggregator": aggregator,
        "aggregator_label": AGGREGATOR_ALIASES.get(aggregator, aggregator),
        "attack": attack,
        "attack_label": ATTACK_ALIASES.get(attack, attack),
        "topology": topology,
        "topology_label": TOPOLOGY_ALIASES.get(topology, topology),
        "n_malicious_pct": n_malicious_pct,
    }
    if gamma is not None:
        result["gamma"] = float(gamma)
    if kappa is not None:
        result["kappa"] = float(kappa)
    if alpha is not None:
        result["alpha"] = float(alpha)
    if n_nodes is not None:
        result["n_nodes"] = int(n_nodes)

    return result


# ============================================================================
# scan_scenario_root
# ============================================================================


def scan_scenario_root(root_dir: str) -> list[dict]:
    """Scan all immediate subdirectories under root_dir and parse their names.

    Each returned dict includes all keys from parse_scenario_name() plus:
        scenario_dir    str  — absolute path to the scenario directory

    Directories that do not match the naming convention are skipped with a
    warning.  Non-directory entries are silently ignored.
    """
    if not os.path.isdir(root_dir):
        logger.warning(f"scan_scenario_root: '{root_dir}' is not a directory")
        return []

    results: list[dict] = []
    entries = sorted(os.listdir(root_dir))
    for entry in entries:
        full_path = os.path.join(root_dir, entry)
        if not os.path.isdir(full_path):
            continue

        meta = parse_scenario_name(entry)
        if meta is None:
            logger.debug(f"Skipping '{entry}': does not match naming convention")
            continue

        # Only include directories that actually have metrics data
        metrics_path = os.path.join(full_path, "metrics")
        analysis_path = os.path.join(full_path, "analysis")
        has_data = os.path.isdir(metrics_path) or os.path.isdir(analysis_path)
        if not has_data:
            logger.debug(f"Skipping '{entry}': no metrics/ or analysis/ subdirectory")
            continue

        meta["scenario_dir"] = full_path
        results.append(meta)
        logger.debug(f"Found scenario: {entry} → {meta['aggregator_label']} / {meta['attack_label']}")

    logger.info(
        f"scan_scenario_root('{root_dir}'): {len(results)} valid scenarios found "
        f"(skipped {len(entries) - len(results)} non-matching entries)"
    )
    return results


# ============================================================================
# load_scenario_results
# ============================================================================


def load_scenario_results(scenario_dir: str) -> tuple[dict | None, pd.DataFrame | None]:
    """Load pre-computed analysis artifacts for a scenario.

    Looks in {scenario_dir}/analysis/ for:
        convergence_curves.json  → dict
        results_summary.csv      → pd.DataFrame

    Either return value may be None if the file has not yet been generated.
    Does NOT raise — callers should handle None gracefully.

    Returns:
        (convergence_dict, results_df)
    """
    analysis_dir = os.path.join(scenario_dir, "analysis")

    convergence: dict | None = None
    results_df: pd.DataFrame | None = None

    # ── convergence_curves.json ───────────────────────────────────────────────
    # radar_metrics_aggregator always writes {scenario_name: {metric: [entries]}}
    # for both single-scenario and multi-scenario runs.
    conv_path = os.path.join(analysis_dir, "convergence_curves.json")
    if os.path.isfile(conv_path):
        try:
            with open(conv_path) as f:
                convergence = json.load(f)
            logger.debug(f"Loaded convergence_curves.json from {analysis_dir}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load {conv_path}: {e}")
    else:
        logger.debug(f"convergence_curves.json not found in {analysis_dir}")

    # ── results_summary.csv ───────────────────────────────────────────────────
    csv_path = os.path.join(analysis_dir, "results_summary.csv")
    if os.path.isfile(csv_path):
        try:
            results_df = pd.read_csv(csv_path)
            logger.debug(f"Loaded results_summary.csv from {analysis_dir}: {len(results_df)} rows")
        except (OSError, pd.errors.ParserError) as e:
            logger.warning(f"Could not load {csv_path}: {e}")
    else:
        logger.debug(f"results_summary.csv not found in {analysis_dir}")

    return convergence, results_df


# ============================================================================
# Helper: get final-round F1 mean/std for a scenario
# ============================================================================


def get_final_metric(results_df: pd.DataFrame, metric: str = "F1") -> tuple[float | None, float | None]:
    """Extract (mean, std) for a metric from a results_summary DataFrame.

    Returns (None, None) if metric not found.
    """
    if results_df is None or results_df.empty:
        return None, None
    row = results_df[results_df["metric"] == metric]
    if row.empty:
        return None, None
    mean = float(row.iloc[0]["mean"])
    std = float(row.iloc[0].get("std", 0.0))
    return mean, std


# ============================================================================
# Self-test (run directly)
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    test_cases = [
        ("fedavg_noattack_erdosrenyi_0pct", True),
        ("fedavg_gaussian_erdosrenyi_20pct", True),
        ("krum_trimmean_erdosrenyi_60pct", True),
        ("balance_gaussian_erdosrenyi_20pct", True),
        ("balance_noattack_ring_0pct", True),
        ("balance_gamma05_gaussian_erdosrenyi_20pct", True),
        ("balance_gamma15_noattack_erdosrenyi_0pct", True),
        ("balance_kappa10_gaussian_erdosrenyi_20pct", True),
        ("balance_alpha07_gaussian_erdosrenyi_20pct", True),
        ("balance_nodes50_gaussian_erdosrenyi_20pct", True),
        ("trimmedmean_krum_attack_erdosrenyi_20pct", True),
        ("median_weightswap_ring_40pct", True),
        ("not_a_valid_name", False),
        ("nebula_DFL_10_03_2026_03_56_59", False),  # raw Nebula output dir
    ]

    print("\n=== parse_scenario_name() tests ===")
    all_ok = True
    for name, should_parse in test_cases:
        result = parse_scenario_name(name)
        ok = (result is not None) == should_parse
        status = "✓" if ok else "✗ FAIL"
        if not ok:
            all_ok = False
        if result:
            print(f"  {status}  {name}")
            print(
                f"       → agg={result['aggregator_label']:12}  attack={result['attack_label']:20}"
                f"  topo={result['topology_label']:14}  mal={result['n_malicious_pct']}%",
                end="",
            )
            extras = {k: v for k, v in result.items() if k in ("gamma", "kappa", "alpha", "n_nodes")}
            if extras:
                print(f"  extras={extras}", end="")
            print()
        else:
            print(f"  {status}  {name}  → None")

    sys.exit(0 if all_ok else 1)
