"""
RADAR Experiment Matrix

Source-of-truth for all experiments. Pure data — no logic.
Each entry is a dict with keys matching build_scenario_config() arguments
plus 'target_name' (the directory name after renaming the Nebula output).

Groups:
  A — Core comparison       (5 aggregators × 6 attacks, erdosrenyi, 20 nodes, ~20% malicious)
  B — Malicious ratio sweep (5 agg × 2 attacks × {20,40,60,80}%)
  C — Topology comparison   (fedavg + balance × 3 topologies × 3 attacks)
  D — Parameter sensitivity (balance × γ/κ/α sweeps × 2 attacks)
  E — Scalability           (fedavg + balance × {10,20,50} nodes × 2 attacks)
"""  # noqa: RUF002

EXPERIMENT_MATRIX: list[dict] = []

# ── A: CORE COMPARISON ────────────────────────────────────────────────────────
_CORE_AGGREGATORS = ["fedavg", "krum", "trimmedmean", "median", "balance"]
_CORE_ATTACKS_20PCT = ["noattack", "gaussian", "noise", "weightswap", "krum_attack"]
_CORE_ATTACKS_60PCT = ["trimmean_attack"]

for _agg in _CORE_AGGREGATORS:
    for _atk in _CORE_ATTACKS_20PCT:
        _pct = 0 if _atk == "noattack" else 20
        EXPERIMENT_MATRIX.append({
            "group": "core",
            "aggregator": _agg,
            "attack": _atk,
            "topology": "erdosrenyi",
            "erdos_probability": 0.8,
            "n_nodes": 5,
            "n_malicious_pct": _pct,
            "rounds": 40,
            "target_name": f"{_agg}_{_atk}_erdosrenyi_{_pct}pct",
        })
    for _atk in _CORE_ATTACKS_60PCT:
        EXPERIMENT_MATRIX.append({
            "group": "core",
            "aggregator": _agg,
            "attack": _atk,
            "topology": "erdosrenyi",
            "erdos_probability": 0.8,
            "n_nodes": 5,
            "n_malicious_pct": 60,
            "rounds": 40,
            "target_name": f"{_agg}_{_atk}_erdosrenyi_60pct",
        })

# ── B: MALICIOUS RATIO SWEEP ──────────────────────────────────────────────────
# gaussian @ {40%, 80%} — 20% already in A
# trimmean_attack @ {20%, 40%, 80%} — 60% already in A
for _agg in _CORE_AGGREGATORS:
    for _atk, _pcts in [("gaussian", [40, 80]), ("trimmean_attack", [20, 40, 80])]:
        for _pct in _pcts:
            _name = f"{_agg}_{_atk}_erdosrenyi_{_pct}pct"
            if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
                EXPERIMENT_MATRIX.append({
                    "group": "malicious_ratio",
                    "aggregator": _agg,
                    "attack": _atk,
                    "topology": "erdosrenyi",
                    "n_nodes": 20,
                    "n_malicious_pct": _pct,
                    "rounds": 100,
                    "target_name": _name,
                })

# ── C: TOPOLOGY COMPARISON ────────────────────────────────────────────────────
# erdosrenyi runs already in A; only add dense + ring
for _agg in ["fedavg", "balance"]:
    for _top in ["dense", "ring"]:
        for _atk in ["noattack", "gaussian", "trimmean_attack"]:
            _pct = 60 if _atk == "trimmean_attack" else (0 if _atk == "noattack" else 20)
            _name = f"{_agg}_{_atk}_{_top}_{_pct}pct"
            if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
                EXPERIMENT_MATRIX.append({
                    "group": "topology",
                    "aggregator": _agg,
                    "attack": _atk,
                    "topology": _top,
                    "n_nodes": 20,
                    "n_malicious_pct": _pct,
                    "rounds": 100,
                    "target_name": _name,
                })


# ── D: PARAMETER SENSITIVITY ──────────────────────────────────────────────────
# default: gamma=1.5, kappa=1.0, alpha=0.5 (already in A as balance_*_erdosrenyi_*pct)
def _fmt_param(v: float) -> str:
    """Format float param value → 2-digit string, e.g. 0.5→'05', 1.5→'15', 2.0→'20'."""
    return str(round(v * 10)).zfill(2)


for _atk in ["noattack", "gaussian"]:
    _pct = 0 if _atk == "noattack" else 20
    # γ sweep (kappa=1.0, alpha=0.5 fixed)  # noqa: RUF003
    for _gamma in [0.5, 1.0, 2.0, 3.0]:  # 1.5 = default, already in A
        _name = f"balance_gamma{_fmt_param(_gamma)}_{_atk}_erdosrenyi_{_pct}pct"
        if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
            EXPERIMENT_MATRIX.append({
                "group": "parameter",
                "aggregator": "balance",
                "attack": _atk,
                "topology": "erdosrenyi",
                "n_nodes": 20,
                "n_malicious_pct": _pct,
                "rounds": 100,
                "balance_params": {"gamma": _gamma, "kappa": 1.0, "alpha": 0.5},
                "target_name": _name,
            })
    # κ sweep (gamma=1.5, alpha=0.5 fixed)
    for _kappa in [0.5, 1.5, 2.0]:  # 1.0 = default, already in A
        _name = f"balance_kappa{_fmt_param(_kappa)}_{_atk}_erdosrenyi_{_pct}pct"
        if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
            EXPERIMENT_MATRIX.append({
                "group": "parameter",
                "aggregator": "balance",
                "attack": _atk,
                "topology": "erdosrenyi",
                "n_nodes": 20,
                "n_malicious_pct": _pct,
                "rounds": 100,
                "balance_params": {"gamma": 1.5, "kappa": _kappa, "alpha": 0.5},
                "target_name": _name,
            })
    # α sweep (gamma=1.5, kappa=1.0 fixed)  # noqa: RUF003
    for _alpha in [0.3, 0.7, 0.9]:  # 0.5 = default, already in A
        _name = f"balance_alpha{_fmt_param(_alpha)}_{_atk}_erdosrenyi_{_pct}pct"
        if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
            EXPERIMENT_MATRIX.append({
                "group": "parameter",
                "aggregator": "balance",
                "attack": _atk,
                "topology": "erdosrenyi",
                "n_nodes": 20,
                "n_malicious_pct": _pct,
                "rounds": 100,
                "balance_params": {"gamma": 1.5, "kappa": 1.0, "alpha": _alpha},
                "target_name": _name,
            })

# ── E: SCALABILITY ────────────────────────────────────────────────────────────
# n_nodes=20 already in A; add 10 and 50
for _agg in ["fedavg", "balance"]:
    for _n in [10, 50]:
        for _atk in ["noattack", "gaussian"]:
            _pct = 0 if _atk == "noattack" else 20
            _name = f"{_agg}_nodes{_n}_{_atk}_erdosrenyi_{_pct}pct"
            if not any(e["target_name"] == _name for e in EXPERIMENT_MATRIX):
                EXPERIMENT_MATRIX.append({
                    "group": "scalability",
                    "aggregator": _agg,
                    "attack": _atk,
                    "topology": "erdosrenyi",
                    "n_nodes": _n,
                    "n_malicious_pct": _pct,
                    "rounds": 100,
                    "target_name": _name,
                })


# ── Summary ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter

    groups = Counter(e["group"] for e in EXPERIMENT_MATRIX)
    total = len(EXPERIMENT_MATRIX)

    print(f"Total experiments: {total}")
    for g, n in sorted(groups.items()):
        print(f"  {g:20s}: {n:3d}")

    # Verify no duplicate target_names
    names = [e["target_name"] for e in EXPERIMENT_MATRIX]
    dupes = [n for n in names if names.count(n) > 1]
    if dupes:
        print(f"\n⚠ Duplicate target names: {set(dupes)}")
    else:
        print(f"\n✓ All {total} target names are unique")

    print("\nFirst 5 entries:")
    for e in EXPERIMENT_MATRIX[:5]:
        print(f"  {e['target_name']}")
