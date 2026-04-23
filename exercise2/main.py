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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from exercise2.data import load_returns
from exercise2.descriptive import compute_descriptive_stats
from exercise2.garch import estimate_garch, estimate_gjr
from exercise2.dcc import edf, estimate_dcc_gaussian, estimate_dcc_t
from exercise2.plotting import plot_dynamic_correlations, plot_correlation_comparison
from report_utils import TexWriter, fnum, fsig, generated_dir


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT_DIR, "config.json")
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

def _run_2_2b_with_capture(returns, asset_names, cfg, fig_dir, dpi):
    """Same as :func:`run_2_2b` but also returns the per-asset GARCH fits."""
    r = returns.values
    T, n = r.shape
    maxiter_garch = cfg["exercise2"]["garch"]["maxiter"]

    print(f"\n{'=' * 78}")
    print("2.2b) DCC-Gaussian copula with GARCH(1,1)-N marginals (zero mean)")
    print("=" * 78)

    margins = []
    z_all = np.zeros((T, n))
    for i in range(n):
        res = estimate_garch(r[:, i], maxiter=maxiter_garch)
        margins.append(res)
        z_all[:, i] = res["z"]
        print(f"  {asset_names[i]}: omega={res['omega']:.2e}, alpha={res['alpha']:.4f}, "
              f"beta={res['beta']:.4f}, persist={res['persistence']:.4f}, "
              f"LL={res['log_likelihood']:.2f}")

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
    return result, margins


def _run_2_2c_with_capture(returns, asset_names, cfg, fig_dir, dpi):
    """Same as :func:`run_2_2c` but also returns the per-asset GJR fits."""
    r = returns.values
    T, n = r.shape
    maxiter_garch = cfg["exercise2"]["garch"]["maxiter"]

    print(f"\n{'=' * 78}")
    print("2.2c) DCC-Gaussian copula with GJR(1,1)-N marginals (zero mean)")
    print("=" * 78)

    margins = []
    z_all = np.zeros((T, n))
    for i in range(n):
        res = estimate_gjr(r[:, i], maxiter=maxiter_garch)
        margins.append(res)
        z_all[:, i] = res["z"]
        print(f"  {asset_names[i]}: omega={res['omega']:.2e}, alpha={res['alpha']:.4f}, "
              f"beta={res['beta']:.4f}, gamma={res['gamma']:.4f}, "
              f"persist={res['persistence']:.4f}, LL={res['log_likelihood']:.2f}")

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
    return result, margins


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
    data_path = ex2["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_DIR, data_path)
    returns = load_returns(data_path, asset_names)
    print(f"Loaded {len(returns)} daily returns for {asset_names}, "
          f"{returns.index[0]} to {returns.index[-1]}\n")

    # 2.1
    stats, corr = run_2_1(returns, asset_names)

    # 2.2a
    res_edf, u_edf = run_2_2a(returns, asset_names, cfg,
                                out["figures_dir"], out["dpi"])

    # 2.2b (with per-asset marginal estimates recaptured for the report)
    res_garch, garch_margins = _run_2_2b_with_capture(
        returns, asset_names, cfg, out["figures_dir"], out["dpi"])

    # 2.2c
    res_gjr, gjr_margins = _run_2_2c_with_capture(
        returns, asset_names, cfg, out["figures_dir"], out["dpi"])

    # 2.3
    run_2_3(res_edf, res_garch, res_gjr, asset_names,
            returns.index, out["figures_dir"], out["dpi"])

    # 2.4
    res_t = run_2_4(u_edf, asset_names, cfg, returns.index,
                    out["figures_dir"], out["dpi"])

    # ------------------------------------------------------------------
    # Emit LaTeX-consumable results
    # ------------------------------------------------------------------
    tex = TexWriter("Exercise 2 — auto-generated results")
    # Raw number (no $...$) so ``$T=\ExTwoT$'' composes without nested math.
    tex.cmd("ExTwoT",           f"{len(returns):,}".replace(",", r"\,"))
    tex.cmd("ExTwoDateStart",   returns.index[0].strftime("%d %B %Y"))
    tex.cmd("ExTwoDateEnd",     returns.index[-1].strftime("%d %B %Y"))

    # 2.1 descriptives table
    rows = []
    for name in asset_names:
        s = stats[name]
        rows.append(
            f"{name} & {fnum(s['mean'], 6)} & {fnum(s['std'], 6)} "
            f"& {fnum(s['skewness'])} & {fnum(s['kurtosis'])}\\\\")
    tex.body("ExTwoDescriptivesBody", rows)

    # 2.1 sample correlations (inline)
    pair_idx = {(0, 1): "BTCETH", (0, 2): "BTCADA", (1, 2): "ETHADA"}
    for (i, j), suffix in pair_idx.items():
        tex.cmd(f"ExTwoCorr{suffix}", fnum(corr[i, j]))

    # 2.2a DCC-Gaussian with EDF
    tex.cmd("ExTwoAA",    fnum(res_edf["a"], 6))
    tex.cmd("ExTwoAB",    fnum(res_edf["b"], 6))
    tex.cmd("ExTwoAPers", fnum(res_edf["a"] + res_edf["b"], 6))
    tex.cmd("ExTwoALL",   fnum(res_edf["log_likelihood"], 2))
    for (i, j), suffix in pair_idx.items():
        tex.cmd(f"ExTwoAOmega{suffix}", fnum(res_edf["Omega"][i, j]))

    # 2.2b GARCH marginals + DCC
    gm_rows = []
    for i, name in enumerate(asset_names):
        m = garch_margins[i]
        gm_rows.append(
            f"{name} & {fsig(m['omega'], 2)} & {fnum(m['alpha'])} "
            f"& {fnum(m['beta'])} & {fnum(m['persistence'])} "
            f"& {fnum(m['log_likelihood'], 2)}\\\\")
    tex.body("ExTwoBGarchBody", gm_rows)
    tex.cmd("ExTwoBA",    fnum(res_garch["a"], 6))
    tex.cmd("ExTwoBB",    fnum(res_garch["b"], 6))
    tex.cmd("ExTwoBPers", fnum(res_garch["a"] + res_garch["b"], 6))
    tex.cmd("ExTwoBLL",   fnum(res_garch["log_likelihood"], 2))

    # 2.2c GJR marginals + DCC
    gj_rows = []
    for i, name in enumerate(asset_names):
        m = gjr_margins[i]
        gj_rows.append(
            f"{name} & {fnum(m['alpha'])} & {fnum(m['beta'])} "
            f"& {fnum(m['gamma'])} & {fnum(m['persistence'])} "
            f"& {fnum(m['log_likelihood'], 2)}\\\\")
    tex.body("ExTwoCGjrBody", gj_rows)
    tex.cmd("ExTwoCA",    fnum(res_gjr["a"], 6))
    tex.cmd("ExTwoCB",    fnum(res_gjr["b"], 6))
    tex.cmd("ExTwoCPers", fnum(res_gjr["a"] + res_gjr["b"], 6))
    tex.cmd("ExTwoCLL",   fnum(res_gjr["log_likelihood"], 2))

    # 2.3 comparison table: Mean/Std/Min/Max per pair per model
    rho_dict = {"EDF": res_edf["rho_series"],
                "GARCH-N": res_garch["rho_series"],
                "GJR-N": res_gjr["rho_series"]}
    pair_labels = res_edf["pair_labels"]
    cmp_rows = []
    for k, (i, j) in enumerate(pair_labels):
        pair = f"{asset_names[i]}--{asset_names[j]}"
        first = True
        for model, series in rho_dict.items():
            col = series[:, k]
            lead = (f"\\multirow{{3}}{{*}}{{{pair}}} " if first else " ")
            first = False
            cmp_rows.append(
                f"{lead}& {model} & {fnum(np.mean(col))} & {fnum(np.std(col))} "
                f"& {fnum(np.min(col))} & {fnum(np.max(col))}\\\\")
        if (i, j) != pair_labels[-1]:
            cmp_rows.append(r"\midrule")
    tex.body("ExTwoCompareBody", cmp_rows)

    # 2.4 DCC-t
    tex.cmd("ExTwoDA",    fnum(res_t["a"], 6))
    tex.cmd("ExTwoDB",    fnum(res_t["b"], 6))
    tex.cmd("ExTwoDNu",   fnum(res_t["nu"], 4))
    tex.cmd("ExTwoDPers", fnum(res_t["a"] + res_t["b"], 6))
    tex.cmd("ExTwoDLL",   fnum(res_t["log_likelihood"], 2))
    tex.cmd("ExTwoDLLGain",
            fnum(res_t["log_likelihood"] - res_edf["log_likelihood"], 2))

    out_path = os.path.join(generated_dir(cfg), "ex2.tex")
    tex.save(out_path)
    print(f"\n  LaTeX macros saved: {out_path}")

    print(f"\n{'=' * 78}")
    print(f"Exercise 2 completed. Figures saved in '{out['figures_dir']}/'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
