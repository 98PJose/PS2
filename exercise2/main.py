"""
Exercise 2 -- Main orchestrator.

Loads config.json and crypto.xlsx, runs parts 2.1 through 2.4.

Usage:
    python -m exercise2.main          (from PS2 directory)
"""

import json
import os
import sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise2.data import load_returns
from exercise2.descriptive import compute_descriptive_stats
from exercise2.garch import estimate_garch, estimate_gjr
from exercise2.dcc import edf, estimate_dcc_gaussian, estimate_dcc_t
from exercise2.plotting import plot_dynamic_correlations, plot_correlation_comparison


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------
# 2.1  Descriptive statistics
# -----------------------------------------------------------------

def run_2_1(returns, asset_names):
    print("=" * 78)
    print("2.1) Descriptive statistics")
    print("=" * 78)

    stats, corr = compute_descriptive_stats(returns)

    print(f"{'Asset':<8}{'Mean':>12}{'Std':>12}{'Skewness':>12}{'Kurtosis':>12}")
    print("-" * 56)
    for name in asset_names:
        s = stats[name]
        print(f"{name:<8}{s['mean']:>12.6f}{s['std']:>12.6f}"
              f"{s['skewness']:>12.4f}{s['kurtosis']:>12.4f}")

    print(f"\n  Sample correlation matrix:")
    print(f"  {'':>6}" + "".join(f"{n:>10}" for n in asset_names))
    for i, n in enumerate(asset_names):
        print(f"  {n:>6}" + "".join(f"{corr[i,j]:>10.4f}" for j in range(len(asset_names))))

    return stats, corr


# -----------------------------------------------------------------
# 2.2a  DCC-Gaussian with EDF marginals
# -----------------------------------------------------------------

def run_2_2a(returns, asset_names, cfg, fig_dir, dpi):
    print(f"\n{'=' * 78}")
    print("2.2a) DCC-Gaussian copula with EDF marginals")
    print("=" * 78)

    r = returns.values
    T, n = r.shape

    # Step I: EDF -> Phi^{-1}
    u = np.column_stack([edf(r[:, i]) for i in range(n)])
    z = norm.ppf(u)

    # Step II: DCC
    x0 = tuple(cfg["exercise2"]["dcc"]["x0_ab"])
    maxiter = cfg["exercise2"]["dcc"]["maxiter"]
    result = estimate_dcc_gaussian(z, x0=x0, maxiter=maxiter)

    print(f"  a = {result['a']:.6f}")
    print(f"  b = {result['b']:.6f}")
    print(f"  a + b = {result['a'] + result['b']:.6f}")
    print(f"  LL = {result['log_likelihood']:.2f}")
    print(f"  Converged = {result['success']}")

    print(f"\n  Unconditional correlation (Omega):")
    Omega = result["Omega"]
    print(f"  {'':>6}" + "".join(f"{n:>10}" for n in asset_names))
    for i, nm in enumerate(asset_names):
        print(f"  {nm:>6}" + "".join(f"{Omega[i,j]:>10.4f}" for j in range(n)))

    path = plot_dynamic_correlations(
        result["rho_series"], result["pair_labels"], asset_names,
        returns.index, "2.2a) Dynamic correlations (Gaussian copula, EDF marginals)",
        fig_dir, "fig_2_2a_dcc_edf.png", dpi)
    print(f"\n  Figure saved: {path}")

    return result, u


# -----------------------------------------------------------------
# 2.2b  DCC-Gaussian with GARCH-N marginals
# -----------------------------------------------------------------

def run_2_2b(returns, asset_names, cfg, fig_dir, dpi):
    print(f"\n{'=' * 78}")
    print("2.2b) DCC-Gaussian copula with GARCH(1,1)-N marginals (zero mean)")
    print("=" * 78)

    r = returns.values
    T, n = r.shape
    maxiter_garch = cfg["exercise2"]["garch"]["maxiter"]

    z_all = np.zeros((T, n))
    for i in range(n):
        res = estimate_garch(r[:, i], maxiter=maxiter_garch)
        z_all[:, i] = res["z"]
        print(f"  {asset_names[i]}: omega={res['omega']:.2e}, alpha={res['alpha']:.4f}, "
              f"beta={res['beta']:.4f}, persist={res['persistence']:.4f}, "
              f"LL={res['log_likelihood']:.2f}")

    # Step II: DCC
    x0 = tuple(cfg["exercise2"]["dcc"]["x0_ab"])
    maxiter = cfg["exercise2"]["dcc"]["maxiter"]
    result = estimate_dcc_gaussian(z_all, x0=x0, maxiter=maxiter)

    print(f"\n  DCC: a={result['a']:.6f}, b={result['b']:.6f}, "
          f"a+b={result['a']+result['b']:.6f}, LL={result['log_likelihood']:.2f}")

    path = plot_dynamic_correlations(
        result["rho_series"], result["pair_labels"], asset_names,
        returns.index, "2.2b) Dynamic correlations (Gaussian copula, GARCH-N marginals)",
        fig_dir, "fig_2_2b_dcc_garch.png", dpi)
    print(f"  Figure saved: {path}")

    return result


# -----------------------------------------------------------------
# 2.2c  DCC-Gaussian with GJR-N marginals
# -----------------------------------------------------------------

def run_2_2c(returns, asset_names, cfg, fig_dir, dpi):
    print(f"\n{'=' * 78}")
    print("2.2c) DCC-Gaussian copula with GJR(1,1)-N marginals (zero mean)")
    print("=" * 78)

    r = returns.values
    T, n = r.shape
    maxiter_garch = cfg["exercise2"]["garch"]["maxiter"]

    z_all = np.zeros((T, n))
    for i in range(n):
        res = estimate_gjr(r[:, i], maxiter=maxiter_garch)
        z_all[:, i] = res["z"]
        print(f"  {asset_names[i]}: omega={res['omega']:.2e}, alpha={res['alpha']:.4f}, "
              f"beta={res['beta']:.4f}, gamma={res['gamma']:.4f}, "
              f"persist={res['persistence']:.4f}, LL={res['log_likelihood']:.2f}")

    # Step II: DCC
    x0 = tuple(cfg["exercise2"]["dcc"]["x0_ab"])
    maxiter = cfg["exercise2"]["dcc"]["maxiter"]
    result = estimate_dcc_gaussian(z_all, x0=x0, maxiter=maxiter)

    print(f"\n  DCC: a={result['a']:.6f}, b={result['b']:.6f}, "
          f"a+b={result['a']+result['b']:.6f}, LL={result['log_likelihood']:.2f}")

    path = plot_dynamic_correlations(
        result["rho_series"], result["pair_labels"], asset_names,
        returns.index, "2.2c) Dynamic correlations (Gaussian copula, GJR-N marginals)",
        fig_dir, "fig_2_2c_dcc_gjr.png", dpi)
    print(f"  Figure saved: {path}")

    return result


# -----------------------------------------------------------------
# 2.3  Compare dynamic correlations
# -----------------------------------------------------------------

def run_2_3(res_edf, res_garch, res_gjr, asset_names, dates, fig_dir, dpi):
    print(f"\n{'=' * 78}")
    print("2.3) Comparison of dynamic correlation series")
    print("=" * 78)

    rho_dict = {
        "EDF": res_edf["rho_series"],
        "GARCH-N": res_garch["rho_series"],
        "GJR-N": res_gjr["rho_series"],
    }

    path = plot_correlation_comparison(
        rho_dict, res_edf["pair_labels"], asset_names, dates,
        fig_dir, "fig_2_3_comparison.png", dpi)
    print(f"  Figure saved: {path}")

    # Print summary statistics of dynamic correlations
    pair_names = [f"{asset_names[i]}-{asset_names[j]}"
                  for i, j in res_edf["pair_labels"]]

    for k, pname in enumerate(pair_names):
        print(f"\n  {pname}:")
        print(f"  {'Model':<12}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
        print(f"  {'-'*42}")
        for model_name, rho_s in rho_dict.items():
            col = rho_s[:, k]
            print(f"  {model_name:<12}{np.mean(col):>10.4f}{np.std(col):>10.4f}"
                  f"{np.min(col):>10.4f}{np.max(col):>10.4f}")

    return path


# -----------------------------------------------------------------
# 2.4  DCC-t copula with EDF marginals
# -----------------------------------------------------------------

def run_2_4(u_edf, asset_names, cfg, dates, fig_dir, dpi):
    """
    u_edf: pseudo-observations from EDF (shape T x n), already computed in 2.2a.
    """
    print(f"\n{'=' * 78}")
    print("2.4) DCC-Student t copula with EDF marginals")
    print("=" * 78)

    x0 = tuple(cfg["exercise2"]["dcc"]["x0_ab_nu"])
    maxiter = cfg["exercise2"]["dcc"]["maxiter"]
    result = estimate_dcc_t(u_edf, x0=x0, maxiter=maxiter)

    print(f"  a = {result['a']:.6f}")
    print(f"  b = {result['b']:.6f}")
    print(f"  nu = {result['nu']:.4f}")
    print(f"  a + b = {result['a'] + result['b']:.6f}")
    print(f"  LL = {result['log_likelihood']:.2f}")
    print(f"  Converged = {result['success']}")

    path = plot_dynamic_correlations(
        result["rho_series"], result["pair_labels"], asset_names,
        dates, "2.4) Dynamic correlations (t copula, EDF marginals)",
        fig_dir, "fig_2_4_dcc_t.png", dpi)
    print(f"  Figure saved: {path}")

    return result


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    cfg = load_config()
    ex2 = cfg["exercise2"]
    out = cfg["output"]
    os.makedirs(out["figures_dir"], exist_ok=True)

    asset_names = ex2["asset_names"]
    returns = load_returns(ex2["data_path"], asset_names)
    print(f"Loaded {len(returns)} daily returns for {asset_names}, "
          f"{returns.index[0]} to {returns.index[-1]}\n")

    # 2.1
    run_2_1(returns, asset_names)

    # 2.2a
    res_edf, u_edf = run_2_2a(returns, asset_names, cfg,
                                out["figures_dir"], out["dpi"])

    # 2.2b
    res_garch = run_2_2b(returns, asset_names, cfg,
                          out["figures_dir"], out["dpi"])

    # 2.2c
    res_gjr = run_2_2c(returns, asset_names, cfg,
                        out["figures_dir"], out["dpi"])

    # 2.3
    run_2_3(res_edf, res_garch, res_gjr, asset_names,
            returns.index, out["figures_dir"], out["dpi"])

    # 2.4
    run_2_4(u_edf, asset_names, cfg, returns.index,
            out["figures_dir"], out["dpi"])

    print(f"\n{'=' * 78}")
    print(f"Exercise 2 completed. Figures saved in '{out['figures_dir']}/'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
