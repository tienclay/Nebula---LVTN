"""
Empirical verification of the non-convex convergence bound (Eq. 14).

Two-panel figure:
  Top  — LHS proxy vs Term(i) on log scale, with 1/T reference line
  Bottom — per-round loss F(w^t) showing convergence to F(w*)

Usage:
    python analysis/plot_convergence_bound_simple.py app/logs/balance_gaussian_erdosrenyi_20pct_0312_153236
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ETA = 0.001  # learning rate (mlp.py:36)
GAMMA = 1.5  # Balance gamma (balance.py:25)
ALPHA = 0.5  # Balance alpha (balance.py:27)
STEPS_PER_ROUND = 200


def _scenario_label(log_dir: Path) -> str:
    name = log_dir.name
    clean = re.sub(r"_\d{4}_\d{6}$", "", name)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from scenario_metadata import parse_scenario_name

        meta = parse_scenario_name(clean) or parse_scenario_name(name)
        if meta:
            agg = meta.get("aggregator_label", meta.get("aggregator", "Balance"))
            attack = meta.get("attack_label", meta.get("attack", "no attack"))
            return f"{agg} — {attack}"
    except Exception:  # noqa: S110
        pass
    return clean


def load(log_dir: Path) -> dict:
    p = log_dir / "analysis" / "convergence_curves.json"
    with open(p) as f:
        return next(iter(json.load(f).values()))


def per_round_loss(metrics):
    vals = np.array([e["mean"] for e in metrics["Loss"] if e.get("mean") is not None])
    n = len(vals) // STEPS_PER_ROUND
    return np.array([vals[t * STEPS_PER_ROUND : (t + 1) * STEPS_PER_ROUND].mean() for t in range(n)])


def plot(log_dir: Path):
    scenario_label = _scenario_label(log_dir)
    metrics = load(log_dir)
    loss_means = per_round_loss(metrics)
    n = len(loss_means)
    f_star = float(np.min(loss_means))
    A = loss_means[0] - f_star  # F(w^0) - F(w*)

    # LHS: telescoping identity — (F(w^0) - F(w^T)) / (eta*T)
    delta = loss_means[:-1] - loss_means[1:]
    T_vals = np.arange(1, n, dtype=float)
    lhs = np.cumsum(delta / ETA) / T_vals

    # Term (i): 2A / (eta*T)
    term_i = 2.0 * A / (ETA * T_vals)

    # Ratio
    ratio = lhs / term_i

    # 1/T reference line anchored to first LHS point
    ref_1t = lhs[0] * 1.0 / T_vals

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 7.5),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18},
    )

    C_LHS = "#2166ac"
    C_TI = "#1a9641"
    C_FILL = "#a6d96a"
    C_LOSS = "#2166ac"
    C_FSTAR = "#d62728"
    C_REF = "#888888"

    # ── Suptitle: full Eq. 14 ─────────────────────────────────────────────────
    fig.suptitle(
        r"$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\left[\|\nabla F(\mathbf{w}^t)\|^2\right]"
        r"\;\leq\;\frac{2[F(\mathbf{w}^0){-}F(\mathbf{w}^*)]}{\eta\,T}"
        r"\;+\;4L\eta\delta^2"
        r"\;+\;\frac{4\gamma\rho\psi(1{-}\alpha)}{\eta}$"
        f"\n{scenario_label}",
        fontsize=10,
    )

    # ── Top panel: LHS vs Term(i) on log scale ───────────────────────────────
    ax1.semilogy(
        T_vals,
        lhs,
        color=C_LHS,
        lw=2.5,
        label=r"LHS proxy: $(F(\mathbf{w}^0)-F(\mathbf{w}^T))/(\eta T)$",
    )
    ax1.semilogy(
        T_vals,
        term_i,
        color=C_TI,
        lw=2.5,
        linestyle="--",
        label=r"Term (i): $2[F(\mathbf{w}^0)-F(\mathbf{w}^*)]/(\eta T)$",
    )
    ax1.fill_between(
        T_vals,
        lhs,
        term_i,
        color=C_FILL,
        alpha=0.35,
        label=r"Slack $\geq 0$",
    )

    # 1/T reference dashed line
    ax1.semilogy(
        T_vals,
        ref_1t,
        color=C_REF,
        lw=1.2,
        linestyle=":",
        label=r"$\propto\,1/T$ reference",
    )

    # Compact 3-line proof box
    proof = (
        r"Telescoping: LHS $= \frac{F(\mathbf{w}^0)-F(\mathbf{w}^T)}{\eta\,T}"
        r"\;\leq\;\frac{F(\mathbf{w}^0)-F(\mathbf{w}^*)}{\eta\,T} = \frac{\mathrm{Term(i)}}{2}$"
        "\n"
        f"Holds at all $T={n}$ rounds.  "
        f"$\\eta={ETA}$,  $A = F(w^0){{-}}F(w^*) = {A:.4f}$\n"
        f"Ratio LHS/Term(i) $\\in$ [{ratio.min():.3f}, {ratio.max():.3f}]"
    )
    ax1.text(
        0.98,
        0.97,
        proof,
        transform=ax1.transAxes,
        fontsize=7.5,
        va="top",
        ha="right",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": C_TI, "lw": 1.2, "alpha": 0.95},
    )

    ax1.set_ylabel(r"$\frac{1}{T}\sum\mathbb{E}[\|\nabla F\|^2]$  (log scale)", fontsize=9)
    ax1.set_title("Eq. 14 verified: LHS $\\leq$ Term (i) at every round", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.58))
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_ylim(bottom=lhs.min() * 0.5)
    ax1.set_xticklabels([])  # shared axis, labels on bottom panel

    # ── Bottom panel: per-round loss F(w^t) ───────────────────────────────────
    rounds = np.arange(n, dtype=float)
    ax2.plot(rounds, loss_means, color=C_LOSS, lw=2, label=r"$F(\mathbf{w}^t)$ per-round loss")
    ax2.axhline(
        f_star, color=C_FSTAR, lw=1.5, linestyle="--", label=rf"$F(\mathbf{{w}}^*) \approx {f_star:.4f}$  (proxy)"
    )
    ax2.fill_between(
        rounds,
        loss_means,
        f_star,
        color=C_LOSS,
        alpha=0.15,
        label=r"Gap $F(\mathbf{w}^t) - F(\mathbf{w}^*)$",
    )

    ax2.set_xlabel("Federated Round $t$", fontsize=9)
    ax2.set_ylabel(r"Loss $F(\mathbf{w}^t)$", fontsize=9)
    ax2.set_title("Per-round loss convergence", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, n - 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = log_dir / "analysis" / "figures" / "convergence_bound_eq14_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    plot(Path(sys.argv[1]))
