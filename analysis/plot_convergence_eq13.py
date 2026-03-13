"""
Regenerate convergence_model_metrics.png for a specific scenario to align
with the three-term convergence bound (Fang et al. Eq. 13):

  E[F(w^T) - F(w*)] <= (1-μη)^T [F(w^0) - F(w*)]   [Term i: optimization error]
                      + 2Lηδ²/μ                       [Term ii: stochastic noise floor]
                      + 2γρψ(1-α)/(μη)               [Term iii: adversarial error]

Usage:
    python analysis/plot_convergence_eq13.py <scenario_log_dir>

Example:
    python analysis/plot_convergence_eq13.py app/logs/nebula_DFL_10_03_2026_03_56_59
"""  # noqa: RUF002

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_curves(log_dir: Path) -> dict:
    json_path = log_dir / "analysis" / "convergence_curves.json"
    if not json_path.exists():
        raise FileNotFoundError(f"convergence_curves.json not found at {json_path}")  # noqa: TRY003
    with open(json_path) as f:
        data = json.load(f)
    # Return the first (and typically only) scenario's metrics
    return next(iter(data.values()))


def extract_series(metrics: dict, key: str):
    """Return (steps, means, stds) arrays for a metric key."""
    entries = metrics.get(key, [])
    valid = [e for e in entries if e.get("mean") is not None]
    if not valid:
        return None, None, None
    steps = np.array([e["step"] for e in valid], dtype=float)
    means = np.array([e["mean"] for e in valid], dtype=float)
    stds = np.array([e.get("std", 0.0) for e in valid], dtype=float)
    return steps, means, stds


def exp_decay(t, a, b, c):
    """a * exp(-b * t) + c  (asymptote at c = noise floor)."""
    return a * np.exp(-b * t) + c


def fit_exp_decay(steps, means):
    """Fit exponential decay; return (params, fitted_curve) or (None, None)."""
    try:
        p0 = [means[0] - means[-1], 0.05, means[-1]]
        bounds = ([0, 1e-6, 0], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(exp_decay, steps, means, p0=p0, bounds=bounds, maxfev=5000)
        return popt, exp_decay(steps, *popt)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def plot(log_dir: Path):
    metrics = load_curves(log_dir)
    scenario_name = log_dir.name

    steps_nd, means_nd, stds_nd = extract_series(metrics, "ModelNormDelta")
    steps_gmn, means_gmn, stds_gmn = extract_series(metrics, "GlobalModelNorm")
    steps_cs, means_cs, stds_cs = extract_series(metrics, "CosineSimilarity")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        r"Convergence Bound Decomposition  "
        r"$\mathbb{E}[F(\mathbf{w}^T){-}F(\mathbf{w}^*)]\leq$"
        " Term(i) + Term(ii) + Term(iii)",
        fontsize=11,
    )

    COLOR = "#7B2D8B"

    # ------------------------------------------------------------------
    # Panel 1 — Term (i): optimization error  ~ (1-μη)^T
    # ------------------------------------------------------------------
    ax1 = axes[0]
    if steps_nd is not None:
        ax1.plot(steps_nd, means_nd, color=COLOR, marker="o", markersize=3, linewidth=1.5, label=scenario_name)
        if np.any(stds_nd > 0):
            ax1.fill_between(
                steps_nd, np.clip(means_nd - stds_nd, 0, None), means_nd + stds_nd, color=COLOR, alpha=0.15
            )

        # Fit exponential decay
        popt, fitted = fit_exp_decay(steps_nd, means_nd)
        if fitted is not None:
            _a, _b, c = popt
            ax1.plot(steps_nd, fitted, "k--", linewidth=1.2, label=r"Fit: $a\,e^{-bt}+c$")
            # Noise floor line (Term ii asymptote)
            ax1.axhline(c, color="red", linewidth=1.0, linestyle=":", label=rf"Noise floor $c={c:.3f}$ (Term ii)")
            ax1.annotate(
                "Term (i)\n" + r"$(1-\mu\eta)^T$ decay",
                xy=(steps_nd[len(steps_nd) // 4], fitted[len(steps_nd) // 4]),
                xytext=(steps_nd[len(steps_nd) // 4] + 5, fitted[len(steps_nd) // 4] + 0.04),
                fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": 0.8},
            )
            ax1.annotate(
                "Term (ii)\nnoise floor",
                xy=(steps_nd[-1], c),
                xytext=(steps_nd[-1] - 20, c + 0.03),
                fontsize=7,
                color="red",
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8},
            )

    ax1.set_xlabel("Training Round")
    ax1.set_ylabel(r"$\|\Delta\mathbf{w}\|/\|\mathbf{w}\|$")
    ax1.set_title(
        r"$\|\Delta\mathbf{w}\|/\|\mathbf{w}\|$ — proxy for Term (i) + (ii)",
        fontsize=9,
    )
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2 — ψ = ||w^t||: model norm (Term iii factor)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    if steps_gmn is not None:
        ax2.plot(steps_gmn, means_gmn, color=COLOR, marker="o", markersize=3, linewidth=1.5, label=scenario_name)
        if np.any(stds_gmn > 0):
            ax2.fill_between(
                steps_gmn, np.clip(means_gmn - stds_gmn, 0, None), means_gmn + stds_gmn, color=COLOR, alpha=0.15
            )

        # Annotate growth
        growth = means_gmn[-1] / means_gmn[0]
        ax2.annotate(
            rf"$\times{growth:.1f}$ growth" + "\n(ψ grows → Term(iii) bound grows)",
            xy=(steps_gmn[-1], means_gmn[-1]),
            xytext=(steps_gmn[-1] * 0.5, means_gmn[-1] * 0.6),
            fontsize=7,
            color="darkred",
            arrowprops={"arrowstyle": "->", "color": "darkred", "lw": 0.8},
        )
        ax2.annotate(
            "No weight decay\n(Adam, wd=0)",
            xy=(steps_gmn[len(steps_gmn) // 3], means_gmn[len(steps_gmn) // 3]),
            xytext=(steps_gmn[len(steps_gmn) // 3] + 5, means_gmn[0] * 0.9),
            fontsize=7,
            color="gray",
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
        )

    ax2.set_xlabel("Training Round")
    ax2.set_ylabel(r"L2 Norm $\|\mathbf{w}^t\|$")
    ax2.set_title(
        r"$\psi = \|\mathbf{w}^t\|$ — Term (iii) factor" + "\n" + r"(adversarial error $\propto \psi$)",
        fontsize=9,
    )
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3 — Term (iii): adversarial error proxy = 1 - CosineSimilarity
    # ------------------------------------------------------------------
    ax3 = axes[2]
    if steps_cs is not None:
        divergence = 1.0 - means_cs
        stds_div = stds_cs  # same magnitude

        ax3.plot(steps_cs, divergence, color=COLOR, marker="o", markersize=3, linewidth=1.5, label=scenario_name)
        if np.any(stds_div > 0):
            ax3.fill_between(
                steps_cs,
                np.clip(divergence - stds_div, 0, None),
                np.clip(divergence + stds_div, 0, 1),
                color=COLOR,
                alpha=0.15,
            )

        final_div = divergence[-1]
        ax3.axhline(final_div, color="green", linewidth=1.0, linestyle=":", label=rf"Final: {final_div:.4f}")
        ax3.annotate(
            f"Term (iii) ≈ {final_div:.4f}\n(no attack → negligible)",
            xy=(steps_cs[-1], final_div),
            xytext=(steps_cs[-1] * 0.4, final_div + 0.005),
            fontsize=7,
            color="green",
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 0.8},
        )

    ax3.set_xlabel("Training Round")
    ax3.set_ylabel(r"$1 - \cos(\mathbf{w}_\mathrm{local},\, \mathbf{w}_\mathrm{global})$")
    ax3.set_title(
        r"$1-\mathrm{cos}(\mathbf{w}_\mathrm{local}, \mathbf{w}_\mathrm{global})$"
        + "\n"
        + r"proxy for Term (iii): adversarial error",
        fontsize=9,
    )
    ax3.set_ylim(-0.001, max(0.05, (1.0 - means_cs).max() * 2) if steps_cs is not None else 0.05)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Save — only to this specific scenario directory
    # ------------------------------------------------------------------
    figures_dir = log_dir / "analysis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "convergence_model_metrics.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    plot(Path(sys.argv[1]))
