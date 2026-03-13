# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEBULA is a platform for **Decentralized Federated Learning (DFL)**. It trains federated ML models across Docker-containerized nodes in centralized (CFL) or decentralized (DFL) topologies. The platform consists of a web frontend, a controller that orchestrates scenarios, and core nodes that run inside Docker containers.

## Common Commands

### Setup
```bash
make install              # Install uv + Python 3.11 + core/controller deps + pre-commit
source .venv/bin/activate # Activate virtual environment
```

### Run the Platform
```bash
python app/main.py                    # Start controller (port 5000) + frontend (port 6060)
python app/main.py -t                 # Run interactive integration test menu
python app/main.py --stop all         # Stop all platform services
```

### Code Quality
```bash
make check                # Run pre-commit hooks (ruff lint + format, yaml/toml checks)
make check-plus           # Above + black, mypy, deptry
uv run ruff check .       # Lint only
uv run ruff format .      # Format only
```

### Build & Docs
```bash
make build                # Build Python wheel
make doc-serve            # Serve MkDocs locally
make update               # Rebuild nebula-frontend and nebula-core Docker images
```

### Tests
Tests are Docker-based integration tests, not unit tests. Run via `python app/main.py -t` which presents a menu (Aggregation, Topology, Dataset, Attacks, Custom). Test scenario configs live in `nebula/tests/*.json`.

## Architecture

```
Frontend (FastAPI+Jinja2, port 6060)
    ↓ HTTP API
Controller (FastAPI, port 5000) — orchestrates scenarios, spawns Docker containers
    ↓ Docker SDK
Core Nodes (Docker containers) — each runs Engine with Trainer + Aggregator + CommsMgr
    ↔ P2P protobuf over TCP (TLS 1.3 optional) with LZ4 compression
```

### Key Components

- **`app/main.py`** — CLI entry point, launches Controller
- **`nebula/controller.py`** — FastAPI orchestrator; spawns node containers, monitors filesystem for scenario scripts
- **`nebula/scenarios.py`** — `Scenario` data class + `ScenarioManagement` lifecycle (start/stop nodes, Docker networking, certs, TensorBoard)
- **`nebula/node.py`** — Node entry point inside each Docker container; loads dataset+model, starts Engine
- **`nebula/core/engine.py`** — Central node coordinator; roles: TrainerNode, AggregatorNode, ServerNode, IdleNode, MaliciousNode
- **`nebula/core/network/`** — P2P networking: `CommunicationsManager`, `Connection`, `Discoverer`, `Forwarder`, `Propagator`
- **`nebula/core/pb/nebula.proto`** — Protobuf schema; message types: Discovery, Control, Federation, Model, Connection, Response, Security
- **`nebula/core/aggregation/`** — Pluggable algorithms: FedAvg, Krum, Median, TrimmedMean, BlockchainReputation, DualHistAgg, Balance
- **`nebula/core/training/`** — Lightning (PyTorch Lightning), Siamese, Scikit trainers
- **`nebula/core/datasets/`** — MNIST, FashionMNIST, EMNIST, CIFAR-10/100, Sentiment140 with IID/non-IID partitioning
- **`nebula/core/models/`** — Per-dataset CNN, MLP, ResNet, MobileNet variants
- **`nebula/frontend/`** — FastAPI web app (`app.py`), async SQLite databases, Jinja2 templates, Nginx reverse proxy
- **`nebula/addons/`** — Attacks (data poison, label flip, model poison, trimmed mean, median, krum), blockchain (Ethereum/Fabric), mobility simulation, WAF (Nginx+ModSecurity), trustworthiness metrics, topology manager (networkx)

### Key Patterns

- **Async-first**: All networking, DB access, HTTP, and event handling use `asyncio`
- **Event-driven P2P**: Nodes use pub/sub `EventManager` with `@event_handler` decorator to route protobuf messages
- **Docker-in-Docker**: Controller spawns participant nodes as Docker containers via Docker SDK
- **Config-as-JSON**: Every node/scenario is fully described by JSON config files passed to containers
- **Pluggable aggregation/training**: Factory patterns select algorithm/trainer from scenario config
- **Security layer**: Optional TLS 1.3 encryption (ECC/SECP256R1 certs with DNS-based SANs) and MTD (Moving Target Defense) neighbor selection via coin-flipping protocol, controlled per-scenario through `SecurityConfig(encryption, mtd, acceptanceRandomProbability)`
- **Coin-flipping neighbor selection**: 4-phase protocol (Ready → Commit → Reveal → Verify) in `Engine` + `CommunicationsManager` using SHA-256 commitments to randomly select federation peers each round when MTD is enabled

## Code Style

- Python 3.11, line length 120
- Linter: `ruff` with rules: flake8-bandit, flake8-bugbear, flake8-builtins, flake8-comprehensions, isort, pycodestyle, pyflakes, pyupgrade, and more (see `pyproject.toml [tool.ruff]`)
- Formatter: `ruff format` (preview mode enabled)
- Pre-commit hooks enforce ruff lint/format, YAML/TOML validation, trailing whitespace, EOF newlines
- Package manager: `uv` (all commands via `uv run` or `uv sync`)

## Security Configuration (dev branch)

Security is configured per-scenario via `security_args` in participant JSON config:
- **`encryption`** (bool) — Enables TLS 1.3 for all P2P connections. Certs are ECC (SECP256R1) with DNS-based SANs (`participant-{idx}.nebula`), generated by `nebula/core/utils/certificate.py`
- **`mtd`** (bool) — Enables Moving Target Defense: randomized neighbor selection each round via coin-flipping protocol
- **`acceptanceRandomProbability`** (float, 0-1) — Threshold for coin-flip outcome; peers whose `(bit_A + bit_B) mod 1 < p` are selected

Key files: `nebula/core/engine.py` (protocol orchestration), `nebula/core/network/communications.py` (commit/reveal/verify crypto, `SecurityNeighborData` state), `nebula/core/pb/nebula.proto` (`SecurityMessage`), `nebula/core/utils/certificate.py` (ECC cert generation with DNS SANs)

## Balance Aggregation (dev branch)

`nebula/core/aggregation/balance.py` — Similarity-based aggregator that filters neighbors by distance threshold:
- Threshold = `gamma * exp(-kappa * round/total_rounds) * ||local_model||`
- Final model = `alpha * local + (1-alpha) * avg(similar_neighbors)`
- Configured via `aggregator_args.balance_params` (`gamma`, `kappa`, `alpha`)

## New Model Attacks (dev branch)

- **TrimmedMeanAttack** (`nebula/addons/attacks/model/trimmedmeanattack.py`) — Full-knowledge (uses benign weights + global history) and partial-knowledge modes
- **MedianAttack** (`nebula/addons/attacks/model/medianattack.py`) — Partial-knowledge attack estimating wmax/wmin from model history
- **KrumAttack** (`nebula/addons/attacks/model/krumattack.py`) — Single-malicious-client Krum poisoning

## Address Format (dev branch)

Node addresses now include DNS: `{ip}:{port}:{dns}` (e.g., `192.168.1.1:5000:participant-0.nebula`), changed from the previous `{ip}:{port}` format. This affects connection parsing throughout `engine.py` and `communications.py`.

## Dependency Groups

Defined in `pyproject.toml` under `[dependency-groups]`:
- **core** — PyTorch, Lightning, aiohttp, protobuf, web3, scikit-learn (for node containers)
- **controller** — FastAPI, Docker SDK, protobuf, gunicorn (for orchestrator)
- **frontend** — FastAPI, aiosqlite, argon2-cffi, torch, tensorboard (for web UI)
- **docs** — MkDocs + material theme + mkdocstrings
