"""Exercise 6 — figures supporting the verification of Lo (2007) eq. (26)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_table1(results, freq, figures_dir, dpi=150):
    """
    For each rho, plot the closed-form active and passive components vs.
    zeta with the simulated counterparts overlaid. All values are
    annualized by ``freq``.
    """
    os.makedirs(figures_dir, exist_ok=True)
    rhos = sorted({c["rho"] for c in results})
    fig, axes = plt.subplots(1, len(rhos), figsize=(4.0 * len(rhos), 4.2),
                             sharey=True, squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        cells = [c for c in results if c["rho"] == rho]
        cells.sort(key=lambda c: c["zeta"])
        zetas = np.array([c["zeta"] for c in cells]) * 100.0  # %

        th_active = np.array([c["theory"]["active"] for c in cells]) * freq * 100.0
        th_pass   = np.array([c["theory"]["passive"] for c in cells]) * freq * 100.0
        th_total  = np.array([c["theory"]["E_Rp"]    for c in cells]) * freq * 100.0

        sim_active = np.array([c["sim"]["active"] for c in cells]) * freq * 100.0
        sim_pass   = np.array([c["sim"]["passive"] for c in cells]) * freq * 100.0
        sim_total  = np.array([c["sim"]["E_Rp"]    for c in cells]) * freq * 100.0
        sim_se     = np.array([c["sim"]["se_Rp"]   for c in cells]) * freq * 100.0

        ax.plot(zetas, th_total,  "-",  color="tab:blue",
                label=r"$E[R_{pt}]$ (eq. 26)")
        ax.plot(zetas, th_active, "--", color="tab:red",   label="active (theory)")
        ax.plot(zetas, th_pass,   "--", color="tab:green", label="passive (theory)")

        ax.errorbar(zetas, sim_total, yerr=1.96 * sim_se, fmt="o",
                    color="tab:blue",  ms=5, capsize=3, label="$E[R_{pt}]$ (sim.)")
        ax.plot(zetas, sim_active, "s", color="tab:red",   ms=5,
                label="active (sim.)")
        ax.plot(zetas, sim_pass,   "^", color="tab:green", ms=5,
                label="passive (sim.)")

        ax.axhline(0, color="k", lw=0.7, alpha=0.5)
        ax.set_title(rf"$\rho={rho:+.2f}$")
        ax.set_xlabel(r"threshold $\zeta$  (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("annualized return (%)")
    axes[0, -1].legend(fontsize=8, loc="best", ncol=1)
    fig.suptitle("Lo (2007) Table 1 — closed-form (lines) vs. simulation (markers)")
    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_6_table1.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[exercise6] saved {out}")
    return out


def plot_sample_path(R, zeta, R_f, omega, R_p, figures_dir, dpi=150,
                     rho=None):
    """
    Plot one illustrative AR(1) path together with the realised stop-loss
    switch (omega = 0 regions shaded) and the resulting portfolio return.
    """
    os.makedirs(figures_dir, exist_ok=True)
    n = R.size
    t = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

    ax1.plot(t, R * 100, color="tab:blue", lw=1.0, label=r"$R_t$ (risky)")
    ax1.axhline(zeta * 100, color="tab:red", ls="--",
                label=rf"$\zeta = {zeta * 100:.1f}\%$")
    ax1.axhline(R_f * 100, color="tab:green", ls=":",
                label=rf"$R_f = {R_f * 100:.3f}\%$")

    # shade the dates where omega_t = 0 (we are in the risk-free asset)
    in_risk_free = False
    start = 0
    for i, w in enumerate(omega):
        # date at which this omega applies is i + 1
        if w == 0.0 and not in_risk_free:
            in_risk_free = True
            start = i + 1
        elif w == 1.0 and in_risk_free:
            in_risk_free = False
            ax1.axvspan(start - 0.5, i + 0.5, alpha=0.15, color="tab:orange")
    if in_risk_free:
        ax1.axvspan(start - 0.5, n - 0.5, alpha=0.15, color="tab:orange")

    title = "AR(1) sample path with stop-loss switch (orange = invested in $R_f$)"
    if rho is not None:
        title += rf"  --  $\rho={rho:+.2f}$"
    ax1.set_title(title)
    ax1.set_ylabel("monthly return (%)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    ax2.plot(np.arange(1, n), R_p * 100, color="tab:purple", lw=1.0,
             label=r"$R_{pt}$ (stop-loss portfolio)")
    ax2.axhline(R_f * 100, color="tab:green", ls=":")
    ax2.set_xlabel("month  $t$")
    ax2.set_ylabel("monthly return (%)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_6_sample_path.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[exercise6] saved {out}")
    return out
