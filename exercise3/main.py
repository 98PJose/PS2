"""
Exercise 3 orchestrator.

Computes the q-quantile curves u_1(u_2 ; q) of the bivariate Normal Mixture
(NM) copula for q in {0.05, 0.25, 0.5, 0.75, 0.95} and (pi, rho_1, rho_2) =
(0.3, -0.7, 0.4), with N(0,1) marginals for both stock returns. Produces a
two-panel figure on the unit-square and the return scale, and reports a small
table of representative values for sanity checks.

Run with:
    cd PS2
    python -m exercise3.main
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import norm

from .nm_copula import nm_conditional_cdf, nm_quantile_curve
from .plotting import plot_quantile_curves


def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_3(cfg):
    ex3 = cfg["exercise3"]
    pi = ex3["pi"]
    rho1 = ex3["rho1"]
    rho2 = ex3["rho2"]
    q_values = ex3["q_values"]
    n_grid = ex3["n_grid"]
    eps = ex3.get("eps", 1e-4)

    # Grid on u_2 avoiding the exact boundaries (where Phi^{-1} diverges).
    u2_grid = np.linspace(eps, 1.0 - eps, n_grid)

    figures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", cfg["output"]["figures_dir"])
    curves = plot_quantile_curves(
        u2_grid, q_values, pi, rho1, rho2, figures_dir, dpi=cfg["output"]["dpi"])

    # Sanity checks: C^NM(u_1(u_2;q) | u_2) should equal q.
    print("\n[exercise3] verification — C^NM(u_1_solved | u_2) vs. q")
    print(f"  parameters: pi = {pi}, rho_1 = {rho1}, rho_2 = {rho2}")
    header = "    u_2   " + "  ".join(f"q={q:<5g}" for q in q_values)
    print(header)
    check_u2 = [0.1, 0.25, 0.5, 0.75, 0.9]
    for u2 in check_u2:
        row = f"    {u2:<6.2f}"
        for q in q_values:
            u1_idx = int(np.round((u2 - eps) / (1.0 - 2 * eps) * (n_grid - 1)))
            u1 = curves[q][u1_idx]
            c_val = float(nm_conditional_cdf(u1, u2, pi, rho1, rho2))
            row += f"  {c_val:.4f}"
        print(row)

    # Tabulate u_1 at a few u_2 values for the report.
    print("\n[exercise3] u_1 = u_1(u_2 ; q) at selected u_2")
    print(header)
    for u2 in check_u2:
        row = f"    {u2:<6.2f}"
        for q in q_values:
            u1_idx = int(np.round((u2 - eps) / (1.0 - 2 * eps) * (n_grid - 1)))
            row += f"  {curves[q][u1_idx]:.4f}"
        print(row)

    # Return-scale endpoints (N(0,1) marginals): r_1 at representative r_2.
    print("\n[exercise3] r_1 = Phi^{-1}(u_1) at r_2 = Phi^{-1}(u_2)")
    header_r = "    r_2    " + "  ".join(f"q={q:<5g}" for q in q_values)
    print(header_r)
    for u2 in check_u2:
        r2 = float(norm.ppf(u2))
        row = f"    {r2:<7.3f}"
        for q in q_values:
            u1_idx = int(np.round((u2 - eps) / (1.0 - 2 * eps) * (n_grid - 1)))
            r1 = float(norm.ppf(curves[q][u1_idx]))
            row += f"  {r1: .4f}"
        print(row)


def main():
    cfg = load_config()
    print("=" * 70)
    print("Exercise 3 — NM copula q-quantile curves")
    print("=" * 70)
    run_3(cfg)


if __name__ == "__main__":
    main()
