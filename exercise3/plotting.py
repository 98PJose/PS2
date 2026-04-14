"""Exercise 3 — plotting utilities for NM copula q-quantile curves."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .nm_copula import nm_quantile_curve


def plot_quantile_curves(u2_grid, q_values, pi, rho1, rho2, figures_dir, dpi=150):
    """
    Produce the q-quantile curves of the NM copula and save the figure.

    The figure has two panels:
      * Left: curves on the unit square (u_2 on x-axis, u_1 on y-axis).
      * Right: same curves mapped to N(0,1) returns r_i = Phi^{-1}(u_i).

    Returns
    -------
    dict
        Maps q -> ndarray of u_1 values on ``u2_grid``.
    """
    os.makedirs(figures_dir, exist_ok=True)

    curves = {q: nm_quantile_curve(u2_grid, q, pi, rho1, rho2) for q in q_values}
    r2_grid = norm.ppf(u2_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_values)))

    for color, q in zip(colors, q_values):
        u1 = curves[q]
        axes[0].plot(u2_grid, u1, color=color, lw=1.8, label=f"q = {q:g}")
        axes[1].plot(r2_grid, norm.ppf(u1), color=color, lw=1.8, label=f"q = {q:g}")

    title = rf"NM copula quantile curves  ($\pi={pi}$, $\rho_1={rho1}$, $\rho_2={rho2}$)"
    fig.suptitle(title)

    axes[0].set_xlabel(r"$u_2$")
    axes[0].set_ylabel(r"$u_1$")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Unit-square scale")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_xlabel(r"$r_2 = \Phi^{-1}(u_2)$")
    axes[1].set_ylabel(r"$r_1 = \Phi^{-1}(u_1)$")
    axes[1].set_title("N(0,1) return scale")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(figures_dir, "fig_3_nm_quantile_curves.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[exercise3] saved {out}")

    return curves
