"""
RADAR Experiment Runner

Automatically runs the full experiment matrix using Nebula process mode.
Handles progress tracking, resume, cleanup, and per-scenario analysis.

Usage:
    # Dry run — print what would run
    python experiments/experiment_runner.py --dry-run

    # Run core comparison only
    python experiments/experiment_runner.py --only core

    # Resume (skip already-done experiments)
    python experiments/experiment_runner.py --resume

    # Run a specific experiment
    python experiments/experiment_runner.py --only-target balance_gaussian_erdosrenyi_20pct

    # Run with custom timeout (seconds)
    python experiments/experiment_runner.py --timeout 10800

    # Skip analysis after each run (run analysis separately at the end)
    python experiments/experiment_runner.py --skip-analysis

Network setup note:
    Process-mode nodes bind to 192.168.50.x IPs.  On macOS, add loopback aliases:
        sudo python experiments/experiment_runner.py --setup-network --n-nodes 20
    Or set --use-localhost to use 127.0.0.1 with unique ports per node (45001+i, no sudo needed).
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import contextlib  # noqa: E402

from experiments.experiment_matrix import EXPERIMENT_MATRIX  # noqa: E402
from experiments.scenario_builder import build_scenario_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 7200  # 2 hours per experiment
CLEANUP_SLEEP = 30  # seconds to wait after each experiment for port release
PROGRESS_FILE = PROJECT_ROOT / "experiments" / "progress.json"
LOGS_DIR = PROJECT_ROOT / "app" / "logs"
CONFIG_DIR = PROJECT_ROOT / "app" / "config"
CERTS_DIR = PROJECT_ROOT / "app" / "certs"

ALL_GROUPS = {"core", "malicious_ratio", "topology", "parameter", "scalability"}


# ── Environment setup ─────────────────────────────────────────────────────────


def setup_nebula_env(root_path: Path) -> None:
    """Set environment variables expected by ScenarioManagement."""
    root = str(root_path)
    os.environ.setdefault("NEBULA_ROOT", root)
    os.environ.setdefault("NEBULA_ROOT_HOST", root)
    os.environ.setdefault("NEBULA_LOGS_DIR", str(root_path / "app" / "logs"))
    os.environ.setdefault("NEBULA_CONFIG_DIR", str(root_path / "app" / "config"))
    os.environ.setdefault("NEBULA_CERTS_DIR", str(root_path / "app" / "certs"))
    os.environ.setdefault("NEBULA_FRONTEND_PORT", "6060")
    os.environ.setdefault("NEBULA_CONTROLLER_NAME", os.environ.get("USER", "nebula"))
    os.environ.setdefault("NEBULA_HOST_PLATFORM", "linux")  # treated as non-windows
    os.environ.setdefault("NEBULA_ADVANCED_ANALYTICS", "False")

    for d in [LOGS_DIR, CONFIG_DIR, CERTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── LocalProcessScenarioManagement ───────────────────────────────────────────


class LocalProcessScenarioManagement:
    """
    Thin wrapper around ScenarioManagement that fixes process-mode path issues
    and replaces the Docker-based scenario_finished() with a PID-based version.

    Keeps the original __init__ / load_configurations_and_start_nodes() but
    overrides start_nodes_process() to use self.config_dir (not hardcoded /nebula/).
    """

    def __init__(self, scenario_config: dict, user: str = "nebula-experiment"):
        from nebula.scenarios import ScenarioManagement as _SM

        # Patch: monkey-patch start_nodes_process on the instance after creation
        self._sm = _SM(scenario_config, user)
        self._sm.start_nodes_process = self._start_nodes_process_fixed
        self.scenario_name = self._sm.scenario_name
        self.config_dir = self._sm.config_dir

    def load_configurations_and_start_nodes(self) -> None:
        self._sm.load_configurations_and_start_nodes()

    def _start_nodes_process_fixed(self) -> None:
        """Fixed start_nodes_process that writes script to config_dir (not /nebula/)."""
        import stat

        sm = self._sm
        config_dir = sm.config_dir
        root_path = sm.root_path
        scenario_name = sm.scenario_name

        # Include additional config to the participants (cert paths, log dirs, etc.)
        # This replicates what the original start_nodes_process() does before writing configs.
        for idx, node in enumerate(sm.config.participants):  # noqa: B007
            node["tracking_args"]["log_dir"] = os.path.join(root_path, "app", "logs")
            node["tracking_args"]["config_dir"] = os.path.join(root_path, "app", "config", scenario_name)
            node["scenario_args"]["controller"] = sm.controller
            node["scenario_args"]["deployment"] = sm.scenario.deployment
            node["security_args"]["certfile"] = os.path.join(
                root_path, "app", "certs", f"participant_{node['device_args']['idx']}_cert.pem"
            )
            node["security_args"]["keyfile"] = os.path.join(
                root_path, "app", "certs", f"participant_{node['device_args']['idx']}_key.pem"
            )
            node["security_args"]["cafile"] = os.path.join(root_path, "app", "certs", "ca_cert.pem")

            # Write the updated config file
            with open(os.path.join(config_dir, f"participant_{node['device_args']['idx']}.json"), "w") as f:
                json.dump(node, f, indent=4)

        # Use .bash extension (not .sh) to avoid triggering the Nebula controller's
        # watchdog (NebulaEventHandler watches *.sh patterns and would double-run the script)
        script_path = os.path.join(config_dir, "current_scenario_commands.bash")
        pid_file = os.path.join(config_dir, "current_scenario_pids.txt")
        done_file = os.path.join(config_dir, "current_scenario_pids.done")

        python_exe = sys.executable  # full path to venv Python — avoids shim/fork PID aliasing

        # Build shell script
        lines = [
            "#!/bin/bash",
            f'PID_FILE="{pid_file}"',
            '> "$PID_FILE"',
            "",
        ]

        # Start node 0 (start node) last (it waits for peers)
        sorted_participants = sorted(
            sm.config.participants,
            key=lambda n: n["device_args"]["idx"],
            reverse=True,  # start node (idx=0) last
        )

        for node in sorted_participants:
            idx = node["device_args"]["idx"]
            is_start = node["device_args"]["start"]
            sleep_s = 10 if is_start else 2
            out_file = os.path.join(root_path, "app", "logs", scenario_name, f"participant_{idx}.out")
            cfg_file = os.path.join(config_dir, f"participant_{idx}.json")
            node_script = os.path.join(root_path, "nebula", "node.py")

            lines += [
                f"sleep {sleep_s}",
                f'echo "Starting node {idx}..."',
                f'"{python_exe}" "{node_script}" "{cfg_file}" > "{out_file}" 2>&1 &',
                'echo $! >> "$PID_FILE"',
                "",
            ]

        lines += [
            'echo "All nodes started."',
            f'echo done > "{done_file}"',  # completion marker — Phase 1 waits for this
        ]
        script_content = "\n".join(lines) + "\n"

        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)  # noqa: S103

        logger.info(f"Executing process script: {script_path}")
        # Execute the script as a subprocess (detached — it backgrounds the nodes)
        subprocess.Popen(  # noqa: S603
            ["bash", script_path],  # noqa: S607
            cwd=config_dir,
            start_new_session=True,
        )

    def scenario_finished(self, timeout_seconds: int = DEFAULT_TIMEOUT) -> bool:  # noqa: C901
        """Wait for all node processes to complete, with early-exit on errors.

        Early-exit conditions (kills all nodes and returns False immediately):
          1. A node PID dies unexpectedly while peers are still running.
          2. Any node's log shows a repeated aggregation timeout error
             (error count increases after already being ≥ 1 → systematic).

        Returns True if all finished within timeout, False on timeout or error.
        """
        sm = self._sm
        pid_file = os.path.join(sm.config_dir, "current_scenario_pids.txt")
        done_file = os.path.join(sm.config_dir, "current_scenario_pids.done")
        n_nodes = sm.n_nodes
        log_dir = Path(sm.root_path) / "app" / "logs" / sm.scenario_name

        start_time = time.time()
        # Phase 1: wait for the bash startup script to finish (all nodes backgrounded)
        # Budget = sleeps in script: (n_nodes-1)*2s (non-start nodes) + 10s (start node) + 30s buffer
        pid_wait_budget = (n_nodes - 1) * 2 + 10 + 30

        # ── Phase 1: wait for completion marker written by bash script ────────
        logger.info(f"Waiting for {n_nodes} node processes to start (up to {pid_wait_budget}s)...")
        while time.time() - start_time < pid_wait_budget:
            if os.path.exists(done_file):
                break
            time.sleep(2)
        else:
            logger.warning("Timed out waiting for all node processes to start")
            return False

        with open(pid_file) as f:
            pids = [int(p.strip()) for p in f if p.strip().isdigit()]

        logger.info(f"All {len(pids)} node processes started. Waiting for completion...")

        # ── Phase 2: poll until all PIDs exit, with early-exit checks ────────
        prev_error_counts: dict[str, int] = {}
        last_error_check = time.time()
        error_check_interval = 15  # seconds between log scans
        training_live = False  # becomes True once FL training starts
        finishing = False  # becomes True once any node starts normal os._exit shutdown
        finishing_deadline = None  # kill lingering nodes after this time
        # Pre-training crash detection: after startup + grace period, any dead PID = abort.
        # Grace = (time for bash script to finish) + 60s for nodes to initialize before connecting.
        pre_training_crash_after = time.time() + pid_wait_budget + 60

        def _kill_all() -> None:
            for pid in pids:
                _kill_pid(pid, signal.SIGKILL)

        while time.time() - start_time < timeout_seconds:
            elapsed = time.time() - start_time
            running = [p for p in pids if _pid_alive(p)]

            if not running:
                logger.info(f"All nodes finished in {elapsed:.0f}s")
                return True

            # Detect normal FL completion: nodes log "calling os._exit(0)" then die.
            # Once any node starts finishing, disable crash detection and wait with grace period.
            if not finishing and training_live:
                finishing = _training_finishing(log_dir, n_nodes)
                if finishing:
                    finishing_deadline = time.time() + 60  # 60s grace for all nodes to exit
                    logger.info(f"  Normal completion detected at {elapsed:.0f}s — waiting up to 60s for nodes to exit")

            if finishing:
                if time.time() > finishing_deadline:
                    # Nodes didn't exit cleanly (asyncio event loop blocked) — kill them
                    still_running = [p for p in pids if _pid_alive(p)]
                    if still_running:
                        logger.info(f"  Grace period expired — killing {len(still_running)} lingering nodes")
                        _kill_all()
                    return True  # experiment completed successfully; cleanup was forced
                time.sleep(2)
                continue

            # Pre-training crash: if a node dies before training starts (e.g. import
            # error, port conflict), federation setup hangs forever.  Abort early.
            if not training_live and time.time() > pre_training_crash_after and len(running) < len(pids):
                dead_count = len(pids) - len(running)
                logger.error(
                    f"  Early exit: {dead_count} node(s) died before training started "
                    f"(check .out file for crash details). Killing remaining nodes."
                )
                _kill_all()
                return False

            # Detect when FL training has actually started — federation setup
            # (connection + FEDERATION_READY exchange) can take an arbitrary amount
            # of time, so crash detection is disabled until training is underway.
            if not training_live:
                training_live = _training_started(log_dir, n_nodes)
                if training_live:
                    logger.info(f"  Training started detected at {elapsed:.0f}s — enabling crash detection")

            if training_live:
                # ── Early exit #1: unexpected node crash ──────────────────────
                # Once training is running, all nodes should stay alive until FL
                # completes.  A dead PID means a node crashed → peers will timeout
                # every round waiting for it → abort immediately.
                if len(running) < len(pids):
                    dead_count = len(pids) - len(running)
                    logger.error(
                        f"  Early exit: {dead_count} node(s) crashed unexpectedly "
                        f"({len(running)} still running). Killing remaining nodes."
                    )
                    _kill_all()
                    return False

                # ── Early exit #2: repeated timeout errors in logs ────────────
                if time.time() - last_error_check >= error_check_interval:
                    last_error_check = time.time()
                    abort, reason = _check_log_errors(log_dir, n_nodes, prev_error_counts)
                    if abort:
                        logger.error(f"  Early exit: {reason}. Killing all nodes.")
                        _kill_all()
                        return False

            # Progress log every 60s
            if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                logger.info(f"  {len(running)}/{len(pids)} nodes still running ({elapsed:.0f}s elapsed)")
            time.sleep(5)

        # ── Timeout: kill remaining processes ─────────────────────────────────
        running = [p for p in pids if _pid_alive(p)]
        if running:
            logger.warning(f"Timeout reached. Killing {len(running)} remaining processes: {running}")
            for pid in running:
                _kill_pid(pid, signal.SIGKILL)
        return False


# ── PID helpers ───────────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True  # noqa: TRY300
    except (ProcessLookupError, PermissionError):
        return False


def _kill_pid(pid: int, sig: int = signal.SIGTERM) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)


# ── Log error detection ───────────────────────────────────────────────────────

# Patterns that indicate a stuck node — if any appears ≥2 times, it's systematic
_ERROR_PATTERNS = [
    "Aggregation timeout",
    "aggregation timeout",
    "AGGREGATION TIMEOUT",
    "Timeout waiting",
    "asyncio.exceptions.TimeoutError",
]


def _training_started(log_dir: Path, n_nodes: int) -> bool:
    """Return True once any node's log shows the FL process has started."""
    for idx in range(n_nodes):
        log_file = log_dir / f"participant_{idx}.log"
        if not log_file.exists():
            continue
        try:
            if "Starting Federated Learning process" in log_file.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _training_finishing(log_dir: Path, n_nodes: int) -> bool:
    """Return True once any node's log shows normal FL completion (os._exit path)."""
    for idx in range(n_nodes):
        log_file = log_dir / f"participant_{idx}.log"
        if not log_file.exists():
            continue
        try:
            content = log_file.read_text(errors="ignore")
            if "calling os._exit(0)" in content or "Calling os._exit(0)" in content:
                return True
        except OSError:
            continue
    return False


def _check_log_errors(
    log_dir: Path,
    n_nodes: int,
    prev_counts: dict[str, int],
) -> tuple[bool, str]:
    """Scan participant log files for repeated error patterns.

    Returns (should_abort, reason_string).
    Aborts if any node's error count increased after already being ≥ 1,
    meaning the same error fired in two consecutive polling intervals.
    """
    for idx in range(n_nodes):
        log_file = log_dir / f"participant_{idx}.out"
        if not log_file.exists():
            continue
        try:
            content = log_file.read_text(errors="ignore")
        except OSError:
            continue
        key = f"participant_{idx}"
        count = sum(content.count(pat) for pat in _ERROR_PATTERNS)
        prev = prev_counts.get(key, 0)
        if prev >= 1 and count > prev:
            return True, f"participant_{idx} has {count} timeout errors (was {prev} on last check)"
        prev_counts[key] = count
    return False, ""


def cleanup_after_experiment(config_dir: str, grace_sleep: int = CLEANUP_SLEEP) -> None:
    """Kill any lingering processes and wait for ports to be released."""
    pid_file = os.path.join(config_dir, "current_scenario_pids.txt")
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pids = [int(p.strip()) for p in f if p.strip().isdigit()]
        for pid in pids:
            _kill_pid(pid)

    # Remove command script (.bash extension avoids the Nebula controller watchdog)
    cmd_script = os.path.join(config_dir, "current_scenario_commands.bash")
    if os.path.exists(cmd_script):
        os.remove(cmd_script)

    if grace_sleep > 0:
        logger.info(f"Sleeping {grace_sleep}s for port/resource cleanup...")
        time.sleep(grace_sleep)


# ── Progress tracking ─────────────────────────────────────────────────────────


def load_progress() -> dict[str, str]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict[str, str]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ── Single experiment runner ──────────────────────────────────────────────────


def run_single_experiment(  # noqa: C901
    entry: dict,
    logs_dir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    skip_analysis: bool = False,
    use_localhost: bool = False,
    test_overrides: dict | None = None,
) -> bool:
    """
    Run one experiment:
    1. Build scenario config
    2. Instantiate LocalProcessScenarioManagement and start nodes
    3. Wait for completion
    4. Rename auto-generated log dir → target_name
    5. Run radar_metrics_aggregator for per-scenario analysis

    Returns True on success, False on timeout or error.
    """
    target_name = entry["target_name"]

    # Check if already exists (resume guard)
    target_dir = logs_dir / target_name
    if target_dir.exists():
        logger.info(f"  Target dir already exists: {target_dir} — skipping rename")

    # ── Build config ──────────────────────────────────────────────────────────
    build_kwargs = {k: v for k, v in entry.items() if k not in ("group", "target_name")}
    if use_localhost:
        build_kwargs["localhost_mode"] = True
        # Remove ip_prefix/port if accidentally present from matrix entries
        build_kwargs.pop("ip_prefix", None)
        build_kwargs.pop("port", None)
    if test_overrides:
        build_kwargs.update(test_overrides)

    try:
        config = build_scenario_config(**build_kwargs)
    except Exception as e:
        logger.exception(f"  Failed to build config: {e}")  # noqa: TRY401
        return False

    # ── Start scenario ────────────────────────────────────────────────────────
    try:
        mgmt = LocalProcessScenarioManagement(config, user="nebula-experiment")
    except Exception as e:
        logger.exception(f"  Failed to init ScenarioManagement: {e}")  # noqa: TRY401
        return False

    auto_name = mgmt.scenario_name
    logger.info(f"  ScenarioManagement started: {auto_name}")

    try:
        mgmt.load_configurations_and_start_nodes()
    except Exception as e:
        logger.exception(f"  Failed to start nodes: {e}")  # noqa: TRY401
        cleanup_after_experiment(mgmt.config_dir, grace_sleep=0)
        return False

    # ── Wait for completion ───────────────────────────────────────────────────
    logger.info(f"  Waiting for completion (timeout: {timeout}s)...")
    t0 = time.time()
    finished = mgmt.scenario_finished(timeout)
    elapsed = int(time.time() - t0)
    status_str = "✓ Finished" if finished else "✗ Timeout"
    logger.info(f"  {status_str} in {elapsed}s")

    # ── Cleanup processes ─────────────────────────────────────────────────────
    cleanup_after_experiment(mgmt.config_dir)

    # ── Move output dir ──────────────────────────────────────────────────────
    # ScenarioManagement always writes to NEBULA_LOGS_DIR (LOGS_DIR), regardless
    # of the runner's --logs-dir flag. Look for auto_dir there, then move it
    # to the target location under logs_dir.
    import shutil as _shutil

    auto_dir = LOGS_DIR / auto_name
    if auto_dir.exists():
        if target_dir.exists():
            # Avoid conflict: add timestamp suffix
            ts = datetime.now().strftime("%m%d_%H%M%S")
            target_dir = logs_dir / f"{target_name}_{ts}"
            logger.warning(f"  Target dir already exists; using {target_dir.name}")
        try:
            _shutil.move(str(auto_dir), str(target_dir))
            logger.info(f"  Moved → {target_dir}")
        except OSError as e:
            logger.exception(f"  Could not move {auto_dir} → {target_dir}: {e}")  # noqa: TRY401
    else:
        logger.warning(f"  Auto-generated log dir not found: {auto_dir}")

    # ── Copy participant configs into scenario dir (for malicious-node detection) ──
    # radar_metrics_aggregator.find_malicious_participants() searches config_dir/{scenario_name}
    # but after renaming, the config dir still has the original nebula_DFL_... name.
    # Copying JSONs directly into target_dir makes them discoverable via config_candidates[2].
    if target_dir.exists():
        config_src = Path(mgmt.config_dir)
        for cfg_path in config_src.glob("participant_*.json"):
            dest = target_dir / cfg_path.name
            if not dest.exists():
                try:
                    _shutil.copy2(cfg_path, dest)
                except OSError as e:
                    logger.warning(f"  Could not copy {cfg_path.name} to scenario dir: {e}")
        logger.info(f"  Copied participant configs → {target_dir.name}/")

    # ── Per-scenario analysis ─────────────────────────────────────────────────
    if not skip_analysis and target_dir.exists():
        _run_per_scenario_analysis(target_dir)

    return finished


def _run_per_scenario_analysis(scenario_dir: Path) -> None:
    """Run the full per-scenario analysis pipeline:
    1. radar_metrics_aggregator  → convergence_curves.json, results_summary.csv, latex_tables.tex
    2. radar_plot_generator      → convergence_f1/accuracy/model_metrics, resource_usage, aggregation_behavior
    3. plot_convergence_eq13     → convergence_bound_eq13.png  (Eq.13 three-term bound)
    4. plot_convergence_bound_simple → convergence_bound_eq14_simple.png  (Eq.14 verification)
    """
    analysis_dir = scenario_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    ANALYSIS = PROJECT_ROOT / "analysis"

    steps = [
        # (label, cmd_args)
        (
            "metrics aggregator",
            [
                sys.executable,
                str(ANALYSIS / "radar_metrics_aggregator.py"),
                "--scenario-dir",
                str(scenario_dir),
                "--output-dir",
                str(analysis_dir),
            ],
        ),
        (
            "plot generator",
            [
                sys.executable,
                str(ANALYSIS / "radar_plot_generator.py"),
                "--convergence-file",
                str(analysis_dir / "convergence_curves.json"),
                "--summary-file",
                str(analysis_dir / "results_summary.csv"),
                "--output",
                str(figures_dir),
                "--format",
                "png",
            ],
        ),
        (
            "convergence bound eq13",
            [
                sys.executable,
                str(ANALYSIS / "plot_convergence_eq13.py"),
                str(scenario_dir),
            ],
        ),
        (
            "convergence bound eq14",
            [
                sys.executable,
                str(ANALYSIS / "plot_convergence_bound_simple.py"),
                str(scenario_dir),
            ],
        ),
    ]

    logger.info("  Running per-scenario analysis...")
    for label, cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode == 0:
            logger.info(f"    {label} ✓")
        else:
            logger.warning(f"    {label} failed (exit {result.returncode})")
            if result.stderr.strip():
                logger.debug(f"    stderr: {result.stderr.strip()[:300]}")


# ── Matrix runner ─────────────────────────────────────────────────────────────


def run_matrix(  # noqa: C901
    matrix: list[dict],
    logs_dir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    resume: bool = True,
    dry_run: bool = False,
    only_groups: set[str] | None = None,
    skip_groups: set[str] | None = None,
    only_target: str | None = None,
    skip_analysis: bool = False,
    use_localhost: bool = False,
    retry_failed: bool = False,
    test_overrides: dict | None = None,
    stop_on_failure: bool = False,
) -> None:
    """Run the experiment matrix with progress tracking."""
    # Apply filters
    filtered = matrix
    if only_target:
        filtered = [e for e in filtered if e["target_name"] == only_target]
        if not filtered:
            logger.error(f"Target '{only_target}' not found in matrix")
            return
    else:
        if only_groups:
            filtered = [e for e in filtered if e["group"] in only_groups]
        if skip_groups:
            filtered = [e for e in filtered if e["group"] not in skip_groups]

    total = len(filtered)
    if total == 0:
        logger.warning("No experiments match the given filters")
        return

    logger.info(f"Experiment matrix: {total} experiments")

    progress = load_progress() if resume else {}
    done_count = 0
    failed_count = 0
    durations: list[float] = []

    for i, entry in enumerate(filtered, 1):
        target = entry["target_name"]
        group = entry.get("group", "?")

        # Resume / skip logic
        status = progress.get(target)
        if status == "done" and not retry_failed:
            logger.info(f"[{i}/{total}] SKIP (done): {target}")
            done_count += 1
            continue
        if status == "failed" and not retry_failed:
            logger.info(f"[{i}/{total}] SKIP (failed, use --retry-failed): {target}")
            failed_count += 1
            continue

        # ETA
        if durations:
            avg_s = sum(durations) / len(durations)
            remaining = total - i + 1
            eta = timedelta(seconds=int(avg_s * remaining))
            eta_str = f"  ETA: ~{eta}"
        else:
            eta_str = ""

        logger.info(f"\n[{i}/{total}] Running ({group}): {target}{eta_str}")

        if dry_run:
            logger.info(f"  DRY RUN — would run: {entry}")
            done_count += 1
            continue

        # Mark as running
        progress[target] = "running"
        save_progress(progress)

        t0 = time.time()
        try:
            success = run_single_experiment(
                entry,
                logs_dir=logs_dir,
                timeout=timeout,
                skip_analysis=skip_analysis,
                use_localhost=use_localhost,
                test_overrides=test_overrides,
            )
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Saving progress...")
            progress[target] = "interrupted"
            save_progress(progress)
            sys.exit(0)
        except Exception as e:
            logger.exception(f"  Unexpected error: {e}")  # noqa: TRY401
            success = False

        elapsed = time.time() - t0
        durations.append(elapsed)

        if success:
            progress[target] = "done"
            done_count += 1
            logger.info(f"  Progress: {done_count}/{total} done{eta_str}")
        else:
            progress[target] = "failed"
            failed_count += 1
            logger.warning(f"  Experiment {target} failed/timed out")

        save_progress(progress)

        if not success and stop_on_failure:
            logger.error(f"  --stop-on-failure: halting after {target}")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE: {done_count}/{total} succeeded, {failed_count} failed")
    failed_names = [k for k, v in progress.items() if v == "failed"]
    if failed_names:
        logger.info("Failed experiments:")
        for n in failed_names:
            logger.info(f"  - {n}")
    logger.info("=" * 60)


# ── Network setup helpers ─────────────────────────────────────────────────────


def setup_loopback_aliases(n_nodes: int, ip_prefix: str = "192.168.50") -> None:
    """Add macOS loopback aliases for 192.168.50.{2..2+n_nodes-1}.

    Requires sudo. Run once before experiments.
    """
    import platform

    if platform.system() != "Darwin":
        logger.warning("Loopback alias setup is only needed on macOS")
        return

    for i in range(n_nodes):
        ip = f"{ip_prefix}.{2 + i}"
        cmd = ["sudo", "ifconfig", "lo0", "alias", ip, "255.255.255.0"]
        logger.info(f"  Adding alias: {ip}")
        result = subprocess.run(cmd)  # noqa: S603
        if result.returncode != 0:
            logger.error(f"  Failed to add alias {ip}")


def remove_loopback_aliases(n_nodes: int, ip_prefix: str = "192.168.50") -> None:
    """Remove macOS loopback aliases. Requires sudo."""
    import platform

    if platform.system() != "Darwin":
        return
    for i in range(n_nodes):
        ip = f"{ip_prefix}.{2 + i}"
        subprocess.run(["sudo", "ifconfig", "lo0", "-alias", ip], capture_output=True)  # noqa: S603, S607
    logger.info(f"Removed {n_nodes} loopback aliases")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RADAR Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="GROUP",
        help=f"Only run these groups: {sorted(ALL_GROUPS)}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="GROUP",
        help="Skip these groups",
    )
    parser.add_argument(
        "--only-target",
        metavar="NAME",
        help="Run a single experiment by target_name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip experiments already marked 'done' in progress.json (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore progress.json and re-run everything",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Re-run experiments marked 'failed' in progress.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print experiment list without running",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-experiment timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip radar_metrics_aggregator after each run",
    )
    parser.add_argument(
        "--use-localhost",
        action="store_true",
        default=False,
        help="Use 127.0.0.1:45001-N instead of 192.168.50.x:45000 (no sudo needed)",
    )
    parser.add_argument(
        "--setup-network",
        action="store_true",
        default=False,
        help="Add macOS loopback aliases for 192.168.50.x IPs (requires sudo)",
    )
    parser.add_argument(
        "--teardown-network",
        action="store_true",
        default=False,
        help="Remove macOS loopback aliases",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=20,
        help="Number of nodes for network alias setup (default: 20)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help=f"Nebula logs directory (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all experiments in the matrix and exit",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Override n_nodes/rounds for quick smoke-test (implies --use-localhost)",
    )
    parser.add_argument(
        "--test-nodes",
        type=int,
        default=3,
        metavar="N",
        help="Number of nodes when --test-mode is active (default: 3)",
    )
    parser.add_argument(
        "--test-rounds",
        type=int,
        default=5,
        metavar="R",
        help="Number of rounds when --test-mode is active (default: 5)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=False,
        help="Halt the entire run on first experiment failure (for interactive debugging)",
    )
    parser.add_argument(
        "--encryption",
        action="store_true",
        default=False,
        help="Enable TLS 1.3 encryption for all P2P connections",
    )
    parser.add_argument(
        "--mtd",
        action="store_true",
        default=False,
        help="Enable Moving Target Defense (randomized neighbor selection each round)",
    )
    args = parser.parse_args()

    # ── Special operations ────────────────────────────────────────────────────
    if args.setup_network:
        setup_loopback_aliases(args.n_nodes)
        return
    if args.teardown_network:
        remove_loopback_aliases(args.n_nodes)
        return
    if args.list:
        progress = load_progress()
        for i, e in enumerate(EXPERIMENT_MATRIX, 1):
            status = progress.get(e["target_name"], "pending")
            flag = {"done": "✓", "failed": "✗", "running": "►", "pending": " "}.get(status, " ")
            print(f"  {flag} [{i:3d}] [{e['group']:15s}] {e['target_name']}")
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    setup_nebula_env(PROJECT_ROOT)

    only_groups = set(args.only) if args.only else None
    skip_groups = set(args.skip) if args.skip else None

    if only_groups and not only_groups.issubset(ALL_GROUPS):
        unknown = only_groups - ALL_GROUPS
        logger.error(f"Unknown groups: {unknown}. Valid: {sorted(ALL_GROUPS)}")
        sys.exit(1)

    test_overrides = None
    use_localhost = args.use_localhost
    if args.test_mode:
        test_overrides = {"n_nodes": args.test_nodes, "rounds": args.test_rounds}
        use_localhost = True
        logger.info(f"TEST MODE: n_nodes={args.test_nodes}, rounds={args.test_rounds}, use_localhost=True")

    if args.encryption or args.mtd:
        security = {"encryption": args.encryption, "mtd": args.mtd, "acceptanceRandomProbability": 0.5}
        test_overrides = test_overrides or {}
        test_overrides["security"] = security
        logger.info(f"Security: encryption={args.encryption}, mtd={args.mtd}")

    run_matrix(
        matrix=EXPERIMENT_MATRIX,
        logs_dir=args.logs_dir,
        timeout=args.timeout,
        resume=args.resume,
        dry_run=args.dry_run,
        only_groups=only_groups,
        skip_groups=skip_groups,
        only_target=args.only_target,
        skip_analysis=args.skip_analysis,
        use_localhost=use_localhost,
        retry_failed=args.retry_failed,
        test_overrides=test_overrides,
        stop_on_failure=args.stop_on_failure,
    )


if __name__ == "__main__":
    main()
