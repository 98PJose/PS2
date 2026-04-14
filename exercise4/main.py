"""
Exercise 4 orchestrator.

Runs the four scenarios from the problem set:

    4.1) Gaussian copula with rho from the MATLAB reference (rho = 0.9).
    4.2) Gaussian copula with rho = 0.3.
    4.3) Student t copula with rho = 0.3, nu = 5.
    4.4) Clayton copula with theta = 10.

For each scenario it simulates N = 100_000 paths of the two stock prices
(N(0,1) shocks coupled by the given copula; GBM marginals with mu = 0.08,
sigma = 0.2, T = 1, S0 = 100), computes cc / cc2 / pp / pp2 returns over the
strike grid K = 85..115 step 1, and saves a 4-panel figure of mean, std,
skewness, and (non-excess) kurtosis.

Run with:
    cd PS2
    python -m exercise4.main
"""

from __future__ import annotations

import json
import os

import numpy as np

from .copulas import sample_shocks
from .option_portfolio import portfolio_returns_over_strikes, simulate_terminal_prices
from .plotting import plot_strategy_statistics
from .stats import strike_stats


def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _strike_grid(grid_cfg):
    start, stop, step = grid_cfg["start"], grid_cfg["stop"], grid_cfg["step"]
    # Inclusive of both endpoints (MATLAB linspace(85,115,31) behaviour).
    n = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, n)


def run_scenario(name, title_suffix, copula_kind, copula_params, cfg, rng):
    gbm = cfg["exercise4"]["gbm"]
    grid = _strike_grid(cfg["exercise4"]["strike_grid"])
    N = cfg["exercise4"]["N"]

    shocks = sample_shocks(copula_kind, N, rng, **copula_params)
    ST = simulate_terminal_prices(shocks, gbm["mu"], gbm["sigma"], gbm["T"], gbm["S0"])
    S1T, S2T = ST[:, 0], ST[:, 1]

    R = portfolio_returns_over_strikes(
        S1T, S2T, grid,
        S0=gbm["S0"], r=gbm["r"], sigma=gbm["sigma"], T=gbm["T"])

    stats_by_strategy = {s: strike_stats(R[s]) for s in ("cc", "cc2", "pp", "pp2")}

    # Short console summary at the ATM point (K = 100).
    atm_idx = int(np.argmin(np.abs(grid - gbm["S0"])))
    print(f"\n[{name}] ATM (K = {grid[atm_idx]:g}) summary")
    print("   strategy   mean       std        skew       kurt")
    for s in ("cc", "cc2", "pp", "pp2"):
        st = stats_by_strategy[s]
        print(f"   {s:<8s}  {st['mean'][atm_idx]: .4f}  {st['std'][atm_idx]: .4f}"
              f"  {st['skew'][atm_idx]: .4f}  {st['kurt'][atm_idx]: .4f}")

    figures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", cfg["output"]["figures_dir"])
    plot_strategy_statistics(
        grid, stats_by_strategy, title_suffix,
        filename=f"fig_4_{name}.png",
        figures_dir=figures_dir,
        dpi=cfg["output"]["dpi"])

    return stats_by_strategy


def main():
    cfg = load_config()
    ex4 = cfg["exercise4"]
    seeds = ex4["seeds"]

    print("=" * 70)
    print("Exercise 4 — Option-based portfolios under alternative copulas")
    print("=" * 70)

    # 4.1 Gaussian, rho from reference MATLAB (0.9)
    run_scenario(
        name="4_1_gaussian_high",
        title_suffix=f"Gaussian copula, rho = {ex4['part_4_1']['rho']}",
        copula_kind="gaussian",
        copula_params={"rho": ex4["part_4_1"]["rho"]},
        cfg=cfg,
        rng=np.random.default_rng(seeds["part_4_1"]))

    # 4.2 Gaussian, rho = 0.3
    run_scenario(
        name="4_2_gaussian_low",
        title_suffix=f"Gaussian copula, rho = {ex4['part_4_2']['rho']}",
        copula_kind="gaussian",
        copula_params={"rho": ex4["part_4_2"]["rho"]},
        cfg=cfg,
        rng=np.random.default_rng(seeds["part_4_2"]))

    # 4.3 Student t, rho = 0.3, nu = 5
    run_scenario(
        name="4_3_student_t",
        title_suffix=(f"Student t copula, rho = {ex4['part_4_3']['rho']}, "
                      f"nu = {ex4['part_4_3']['nu']}"),
        copula_kind="t",
        copula_params={"rho": ex4["part_4_3"]["rho"], "nu": ex4["part_4_3"]["nu"]},
        cfg=cfg,
        rng=np.random.default_rng(seeds["part_4_3"]))

    # 4.4 Clayton, theta = 10
    run_scenario(
        name="4_4_clayton",
        title_suffix=f"Clayton copula, theta = {ex4['part_4_4']['theta']}",
        copula_kind="clayton",
        copula_params={"theta": ex4["part_4_4"]["theta"]},
        cfg=cfg,
        rng=np.random.default_rng(seeds["part_4_4"]))


if __name__ == "__main__":
    main()
