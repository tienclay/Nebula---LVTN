"""
RADAR Experiment Scenario Builder

Builds complete scenario config dicts from high-level parameters,
compatible with ScenarioManagement(config, user).

Usage (as library):
    from experiments.scenario_builder import build_scenario_config
    config = build_scenario_config(aggregator="balance", attack="gaussian",
                                   topology="erdosrenyi", n_nodes=20, n_malicious_pct=20)
"""

import math
import random

# ── Nebula name maps ──────────────────────────────────────────────────────────

AGGREGATOR_MAP: dict[str, str] = {
    "fedavg": "FedAvg",
    "krum": "Krum",
    "trimmedmean": "TrimmedMean",
    "median": "Median",
    "balance": "Balance",
}

# (attack_display_name, attack_type_for_nodes)
ATTACK_MAP: dict[str, tuple[str, str]] = {
    "noattack": ("No Attack", "No Attack"),
    "gaussian": ("Model Poisoning", "ModelPoison"),  # ModelPoisonAttack adds Gaussian noise
    "noise": ("Noise Injection", "ModelPoison"),
    "weightswap": ("Swapping Weights", "ModelPoison"),
    "krum_attack": ("Krum", "ModelPoison"),
    "trimmean_attack": ("Trimmed Mean", "ModelPoison"),
    "labelflip": ("Label Flipping", "LabelFlip"),
}

TOPOLOGY_MAP: dict[str, str] = {
    "erdosrenyi": "Random",
    "dense": "Fully",
    "ring": "Ring",
}

DEFAULT_BALANCE_PARAMS: dict = {"gamma": 1.5, "kappa": 1.0, "alpha": 0.5}
DEFAULT_SECURITY: dict = {
    "encryption": False,
    "mtd": False,
    "acceptanceRandomProbability": 0.5,
}

# Network defaults — override via env or call parameters
BASE_SUBNET = "192.168.50.0/24"
BASE_GATEWAY = "192.168.50.1"
BASE_IP_PREFIX = "192.168.50"  # node i → 192.168.50.{2+i}
BASE_PORT = 45000  # all nodes use the same port (differentiated by IP)

# Localhost mode — single loopback IP, unique port per node
# TopologyManager derives DNS as participant-{port-45001}.nebula, so base=45001 → correct indices
LOCALHOST_IP = "127.0.0.1"
LOCALHOST_BASE_PORT = 45001


# ── Graph helpers ─────────────────────────────────────────────────────────────


def _is_connected(matrix: list[list[int]]) -> bool:
    """BFS connectivity check."""
    n = len(matrix)
    if n == 0:
        return True
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for nb in range(n):
            if matrix[node][nb] and not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    return all(visited)


def _ensure_connected(matrix: list[list[int]]) -> list[list[int]]:
    """Add random edges to make graph connected (modifies in-place)."""
    n = len(matrix)
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for nb in range(n):
            if matrix[node][nb] and not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    # Connect isolated components
    unvisited = [i for i in range(n) if not visited[i]]
    for iso in unvisited:
        rnd = random.choice([i for i in range(n) if visited[i]])  # noqa: S311
        matrix[iso][rnd] = 1
        matrix[rnd][iso] = 1
        visited[iso] = True
    return matrix


def _generate_adjacency(n_nodes: int, topology: str, p: float = 0.5) -> list[list[int]]:  # noqa: C901
    """Generate adjacency matrix for the given topology."""
    matrix = [[0] * n_nodes for _ in range(n_nodes)]

    if topology == "dense":
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    matrix[i][j] = 1

    elif topology == "ring":
        for i in range(n_nodes):
            matrix[i][(i + 1) % n_nodes] = 1
            matrix[(i + 1) % n_nodes][i] = 1

    elif topology == "erdosrenyi":
        max_retries = 100
        for _ in range(max_retries):
            m = [[0] * n_nodes for _ in range(n_nodes)]
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < p:  # noqa: S311
                        m[i][j] = 1
                        m[j][i] = 1
            if _is_connected(m):
                return m
        # Fallback: start with ring and add Erdős-Rényi edges
        for i in range(n_nodes):
            matrix[i][(i + 1) % n_nodes] = 1
            matrix[(i + 1) % n_nodes][i] = 1
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not matrix[i][j] and random.random() < p:  # noqa: S311
                    matrix[i][j] = 1
                    matrix[j][i] = 1
    else:
        raise ValueError(f"Unknown topology: {topology}")  # noqa: TRY003

    return matrix


# ── Main builder functions ────────────────────────────────────────────────────


def generate_nodes_and_graph(
    n_nodes: int,
    topology: str,
    n_malicious_pct: float,
    attack_key: str,
    attack_params: dict,
    erdos_probability: float = 0.5,
    ip_prefix: str = BASE_IP_PREFIX,
    port: int = BASE_PORT,
    localhost_mode: bool = False,
) -> tuple[dict, list, list[list[int]]]:
    """Generate nodes dict, nodes_graph list, and adjacency matrix.

    Returns:
        nodes_dict       {str(idx): node_info_dict}
        nodes_graph      [node_graph_dict, ...]
        adjacency_matrix n_nodes × n_nodes list-of-lists

    localhost_mode: all nodes share LOCALHOST_IP; each gets unique port LOCALHOST_BASE_PORT+i.
        TopologyManager derives DNS as participant-{port-45001}.nebula which aligns correctly.
    """  # noqa: RUF002
    num_malicious = math.ceil(n_malicious_pct / 100.0 * n_nodes)
    # Node 0 (start node) is always benign; malicious from index 1+
    malicious_indices = set(range(1, num_malicious + 1))
    # Cap at n_nodes - 1 (can't make all nodes malicious)
    malicious_indices = {i for i in malicious_indices if i < n_nodes}

    _attack_display, _ = ATTACK_MAP.get(attack_key, ("No Attack", "No Attack"))
    is_attack = attack_key != "noattack"

    # Adjacency matrix (for display and used as fallback by TopologyManager)
    matrix = _generate_adjacency(n_nodes, topology, erdos_probability)

    nodes_dict: dict[str, dict] = {}
    nodes_graph: list[dict] = []

    for i in range(n_nodes):
        if localhost_mode:
            ip = LOCALHOST_IP
            node_port = LOCALHOST_BASE_PORT + i
        else:
            ip = f"{ip_prefix}.{2 + i}"
            node_port = port
        dns = f"participant-{i}.nebula"
        malicious = (i in malicious_indices) and is_attack
        is_start = i == 0

        node_info = {
            "id": i,
            "ip": ip,
            "port": str(node_port),
            "role": "aggregator",
            "malicious": malicious,
            "proxy": False,
            "start": is_start,
            "dns": dns,
        }
        nodes_dict[str(i)] = node_info

        # neighbors from adjacency matrix
        neighbors = [j for j in range(n_nodes) if matrix[i][j] == 1]

        # Random 3-D coordinates for visualization
        x = random.uniform(-30, 30)  # noqa: S311
        y = random.uniform(-30, 30)  # noqa: S311
        z = random.uniform(-30, 30)  # noqa: S311

        nodes_graph.append({
            "id": i,
            "ip": ip,
            "port": str(node_port),
            "role": "aggregator",
            "malicious": malicious,
            "proxy": False,
            "start": is_start,
            "neighbors": neighbors,
            "links": [],
            "index": i,
            "x": x,
            "y": y,
            "z": z,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        })

    return nodes_dict, nodes_graph, matrix


def build_scenario_config(
    aggregator: str,
    attack: str,
    topology: str,
    n_nodes: int = 20,
    n_malicious_pct: float = 20,
    rounds: int = 100,
    epochs: int = 1,
    dataset: str = "MNIST",
    model: str = "MLP",
    balance_params: dict | None = None,
    security: dict | None = None,
    scenario_label: str = "",
    erdos_probability: float = 0.5,
    ip_prefix: str = BASE_IP_PREFIX,
    port: int = BASE_PORT,
    localhost_mode: bool = False,
) -> dict:
    """Build a complete scenario config dict for ScenarioManagement(config, user).

    All string/int types match what Scenario.from_dict() and attack_node_assign()
    expect based on nebula/scenarios.py and participant.json.example.
    """
    aggregator = aggregator.lower()
    attack = attack.lower()
    topology = topology.lower()

    if aggregator not in AGGREGATOR_MAP:
        raise ValueError(f"Unknown aggregator '{aggregator}'. Valid: {list(AGGREGATOR_MAP)}")  # noqa: TRY003
    if attack not in ATTACK_MAP:
        raise ValueError(f"Unknown attack '{attack}'. Valid: {list(ATTACK_MAP)}")  # noqa: TRY003
    if topology not in TOPOLOGY_MAP:
        raise ValueError(f"Unknown topology '{topology}'. Valid: {list(TOPOLOGY_MAP)}")  # noqa: TRY003

    nebula_aggregator = AGGREGATOR_MAP[aggregator]
    attack_display, _ = ATTACK_MAP[attack]
    nebula_topology = TOPOLOGY_MAP[topology]

    n_malicious_pct_int = int(n_malicious_pct)
    attack_params = {
        "poisoned_percent": "0",
        "poisoned_ratio": "0",
        "noise_type": "Salt",
        "targeted": False,
        "target_label": "4",
        "target_changed_label": "7",
        "strength": "10000",
        "layer_idx": "0",
        "delay": "10",
        "round_start_attack": "1",
        "round_stop_attack": str(rounds),
    }

    nodes_dict, nodes_graph, matrix = generate_nodes_and_graph(
        n_nodes=n_nodes,
        topology=topology,
        n_malicious_pct=n_malicious_pct,
        attack_key=attack,
        attack_params=attack_params,
        erdos_probability=erdos_probability,
        ip_prefix=ip_prefix,
        port=port,
        localhost_mode=localhost_mode,
    )

    if localhost_mode:
        net_subnet = "127.0.0.0/8"
        net_gateway = LOCALHOST_IP
    else:
        net_subnet = f"{ip_prefix}.0/24"
        net_gateway = f"{ip_prefix}.1"

    scenario_title = f"{aggregator}_{attack}_{topology}_{n_malicious_pct_int}pct"
    if scenario_label:
        scenario_title = f"{scenario_title}_{scenario_label}"

    description_parts = [
        f"{nebula_aggregator} aggregation",
        f"{attack_display} attack ({n_malicious_pct_int}% malicious)",
        f"{nebula_topology} topology",
        f"{n_nodes} nodes",
        f"{rounds} rounds",
    ]
    if aggregator == "balance" and balance_params:
        p = balance_params
        description_parts.append(f"γ={p.get('gamma', 1.5)} κ={p.get('kappa', 1.0)} α={p.get('alpha', 0.5)}")  # noqa: RUF001

    return {
        "scenario_title": scenario_title,
        "scenario_description": " | ".join(description_parts),
        "deployment": "process",
        "federation": "DFL",
        "topology": nebula_topology,
        "nodes": nodes_dict,
        "nodes_graph": nodes_graph,
        "n_nodes": n_nodes,
        "matrix": matrix,
        "random_topology_probability": erdos_probability,
        "dataset": dataset,
        "iid": False,
        "partition_selection": "dirichlet",
        "partition_parameter": "0.5",
        "model": model,
        "agg_algorithm": nebula_aggregator,
        "balance_params": balance_params if balance_params is not None else dict(DEFAULT_BALANCE_PARAMS),
        "rounds": str(rounds),
        "epochs": str(epochs),
        "logginglevel": True,
        "report_status_data_queue": False,  # no controller running in process mode
        "accelerator": "cpu",
        "gpu_id": [],
        "network_subnet": net_subnet,
        "network_gateway": net_gateway,
        "attacks": attack_display,
        "poisoned_node_percent": str(n_malicious_pct_int) if attack != "noattack" else "0",
        "poisoned_sample_percent": "0",
        "poisoned_noise_percent": "0",
        "attack_params": attack_params,
        "with_reputation": False,
        "is_dynamic_topology": False,
        "is_dynamic_aggregation": False,
        "target_aggregation": False,
        "random_geo": False,
        "latitude": 38.023522,
        "longitude": -1.174389,
        "mobility": False,
        "mobility_type": "topology",
        "radius_federation": "1000",
        "scheme_mobility": "random",
        "round_frequency": "1",
        "mobile_participants_percent": "0",
        "additional_participants": [],
        "schema_additional_participants": "random",
        "security": security if security is not None else dict(DEFAULT_SECURITY),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== scenario_builder self-test ===\n")

    cfg = build_scenario_config(
        aggregator="balance",
        attack="gaussian",
        topology="erdosrenyi",
        n_nodes=5,
        n_malicious_pct=20,
        rounds=10,
        balance_params={"gamma": 0.5, "kappa": 1.0, "alpha": 0.5},
    )

    print(f"scenario_title : {cfg['scenario_title']}")
    print(f"description    : {cfg['scenario_description']}")
    print(f"n_nodes        : {cfg['n_nodes']}")
    print(f"agg_algorithm  : {cfg['agg_algorithm']}")
    print(f"attacks        : {cfg['attacks']}")
    print(f"poisoned_pct   : {cfg['poisoned_node_percent']}")
    print(f"topology       : {cfg['topology']}")
    print(f"balance_params : {cfg['balance_params']}")

    print("\nNodes:")
    for k, v in cfg["nodes"].items():
        flag = "MALICIOUS" if v["malicious"] else "benign "
        start = " [start]" if v["start"] else ""
        print(f"  {k}: {v['ip']}:{v['port']}  {flag}{start}  dns={v['dns']}")

    print("\nMatrix (5×5):")  # noqa: RUF001
    for row in cfg["matrix"]:
        print("  ", row)

    # Verify node 0 is always benign and start
    assert cfg["nodes"]["0"]["start"] is True  # noqa: S101
    assert cfg["nodes"]["0"]["malicious"] is False, "Node 0 must be benign"  # noqa: S101

    n_malicious = sum(1 for v in cfg["nodes"].values() if v["malicious"])
    print(f"\nMalicious nodes: {n_malicious} (expected: 1 = ceil(20% of 5))")
    assert n_malicious == 1  # noqa: S101

    print("\n✓ All assertions passed")
