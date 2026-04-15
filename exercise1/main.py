"""
Exercise 1 — Main orchestrator.

Loads config.json, runs parts 1.a through 1.d, prints results, saves figures.

Usage:
    python -m exercise1.main          (from PS2 directory)
    python exercise1/main.py          (from PS2 directory)
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import OrderedDict

# Ensure PS2 root is on sys.path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise1.copula_sim import (
    sim_clayton, sim_surv_clayton, sim_gaussian,
    sim_student_t, sim_fgm, sim_mixture
)
from exercise1.copula_likelihood import ll_gaussian_copula, ll_t_copula
from exercise1.estimation import edf, estimate_gaussian_copula, estimate_t_copula
from exercise1.portfolio import compute_portfolio_returns, compute_stats
from exercise1.plotting import (
    plot_copula_scatter, plot_portfolio_histograms, plot_mixture_scatter
)
from report_utils import TexWriter, fnum, generated_dir


# Short LaTeX labels for each copula (used in the generated tables).
_COP_LABELS = [
    r"Clayton $\theta=10$",
    r"Surv.\ Clayton $\theta=5$",
    r"Gaussian $\rho=-0.7$",
    r"Student-$t$ $\rho=-0.7,\nu=5$",
    r"FGM $\lambda=-0.5$",
    r"FGM $\lambda=0$",
]


def load_config(path="config.json"):
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_copula_registry(cfg):
    """
    Build an ordered dict of copula name -> simulation callable.

    Each callable takes (v1, v2) and returns (u1, u2).
    Parameters are read from config, not hardcoded.
    """
    cop = cfg["copulas"]
    return OrderedDict([
        (r"i) Clayton $\theta$={}".format(cop["clayton"]["theta"]),
         lambda v1, v2, p=cop["clayton"]: sim_clayton(v1, v2, p["theta"])),

        (r"ii) Surv.Clayton $\theta$={}".format(cop["survival_clayton"]["theta"]),
         lambda v1, v2, p=cop["survival_clayton"]: sim_surv_clayton(v1, v2, p["theta"])),

        (r"iii) Gaussian $\rho$={}".format(cop["gaussian"]["rho"]),
         lambda v1, v2, p=cop["gaussian"]: sim_gaussian(v1, v2, p["rho"])),

        (r"iv) Student-t $\rho$={} $\nu$={}".format(cop["student_t"]["rho"], cop["student_t"]["nu"]),
         lambda v1, v2, p=cop["student_t"]: sim_student_t(v1, v2, p["rho"], p["nu"])),

        (r"v) FGM $\lambda$={}".format(cop["fgm_neg"]["lambda"]),
         lambda v1, v2, p=cop["fgm_neg"]: sim_fgm(v1, v2, p["lambda"])),

        (r"vi) FGM $\lambda$={}".format(cop["fgm_zero"]["lambda"]),
         lambda v1, v2, p=cop["fgm_zero"]: sim_fgm(v1, v2, p["lambda"])),
    ])


# ─────────────────────────────────────────────────────────────
# Part 1.a
# ─────────────────────────────────────────────────────────────

def run_1a(v1, v2, copula_registry, fig_dir, dpi):
    """
    Simulate dependent U(0,1) draws for each copula.
    Compute Pearson, Spearman, Kendall.
    Plot scatter plots.

    Returns dict of copula_name -> (u1, u2).
    """
    print("=" * 78)
    print("1.a) Dependence measures for simulated copulas")
    print("=" * 78)
    print(f"{'Copula':<45}{'Pearson':>11}{'Spearman':>11}{'Kendall':>11}")
    print("-" * 78)

    draws = OrderedDict()
    for name, sim_func in copula_registry.items():
        u1, u2 = sim_func(v1, v2)
        draws[name] = (u1, u2)
        pr = pearsonr(u1, u2)[0]
        sr = spearmanr(u1, u2)[0]
        kt = kendalltau(u1, u2)[0]
        # Strip LaTeX for console output (ASCII-safe for Windows cp1252)
        plain = name.replace("$\\theta$", "th").replace("$\\rho$", "rho") \
                     .replace("$\\nu$", "nu").replace("$\\lambda$", "lam")
        print(f"{plain:<45}{pr:>11.4f}{sr:>11.4f}{kt:>11.4f}")

    # Collect for the report
    rows = []
    dep_values = []
    for i, (name, (u1, u2)) in enumerate(draws.items()):
        pr = pearsonr(u1, u2)[0]
        sr = spearmanr(u1, u2)[0]
        kt = kendalltau(u1, u2)[0]
        dep_values.append((pr, sr, kt))
        rows.append(
            f"{_COP_LABELS[i]} & {fnum(pr)} & {fnum(sr)} & {fnum(kt)}\\\\"
        )

    path = plot_copula_scatter(draws, fig_dir, dpi)
    print(f"\n  Figure saved: {path}")
    return draws, rows, dep_values


# ─────────────────────────────────────────────────────────────
# Part 1.b
# ─────────────────────────────────────────────────────────────

def run_1b(draws, cfg, fig_dir, dpi):
    """
    Portfolio analysis: compute stats, VaR, correlation matrix, histograms.
    """
    weights = cfg["portfolio"]["weights"]
    var_levels = cfg["portfolio"]["var_levels"]
    m1_spec = cfg["marginals"]["stock1"]
    m2_spec = cfg["marginals"]["stock2"]
    Ns = cfg["simulation"]["Ns"]

    print(f"\n{'=' * 95}")
    print(f"1.b) Portfolio statistics  (w={weights}, "
          f"R1~{m1_spec['distribution']}{m1_spec['params']}, "
          f"R2~{m2_spec['distribution']}{m2_spec['params']})")
    print("=" * 95)

    var_hdr = "".join(f"{'VaR'+str(int(q*100))+'%':>9}" for q in var_levels)
    print(f"{'Copula':<45}{'Mean':>8}{'Std':>8}{'Skew':>8}{'Kurt':>8}{var_hdr}")
    print("-" * (77 + 9 * len(var_levels)))

    Rp_all = np.zeros((Ns, len(draws)))
    port_returns = OrderedDict()

    for j, (name, (u1, u2)) in enumerate(draws.items()):
        rp, _, _ = compute_portfolio_returns(u1, u2, m1_spec, m2_spec, weights)
        Rp_all[:, j] = rp
        port_returns[name] = rp
        stats = compute_stats(rp, var_levels)

        plain = name.replace("$\\theta$", "th").replace("$\\rho$", "rho") \
                     .replace("$\\nu$", "nu").replace("$\\lambda$", "lam")
        var_vals = "".join(f"{stats['var'][q]:>9.4f}" for q in var_levels)
        print(f"{plain:<45}{stats['mean']:>8.4f}{stats['std']:>8.4f}"
              f"{stats['skewness']:>8.4f}{stats['kurtosis']:>8.4f}{var_vals}")

    # Correlation matrix
    corr_mat = np.corrcoef(Rp_all.T)
    short = ["Cla", "SCla", "Gauss", "t", "FGM-", "FGM-0"]
    print(f"\n  Correlation matrix of portfolio returns:")
    print(f"  {'':>8}" + "".join(f"{s:>9}" for s in short))
    for i, s in enumerate(short):
        print(f"  {s:>8}" + "".join(f"{corr_mat[i, j]:>9.4f}" for j in range(len(short))))

    # Build report rows for the 1.b portfolio table
    rows = []
    for i, (name, (u1, u2)) in enumerate(draws.items()):
        rp = port_returns[name]
        stats = compute_stats(rp, var_levels)
        cells = [_COP_LABELS[i],
                 fnum(stats['mean']), fnum(stats['std']),
                 fnum(stats['skewness']), fnum(stats['kurtosis'])]
        for q in var_levels:
            cells.append(fnum(stats['var'][q]))
        rows.append(" & ".join(cells) + "\\\\")

    path = plot_portfolio_histograms(port_returns, fig_dir, dpi)
    print(f"\n  Figure saved: {path}")
    return port_returns, rows


# ─────────────────────────────────────────────────────────────
# Part 1.c
# ─────────────────────────────────────────────────────────────

def run_1c(draws, cfg):
    """
    IFM estimation of Gaussian copula from returns generated under
    the Gaussian copula. Marginals estimated via EDF.
    """
    true_rho = cfg["copulas"]["gaussian"]["rho"]
    m1_spec = cfg["marginals"]["stock1"]
    m2_spec = cfg["marginals"]["stock2"]
    bounds = cfg["estimation"]["gaussian_bounds"]

    print(f"\n{'=' * 78}")
    print(f"1.c) IFM/CML estimation of Gaussian copula  (true rho = {true_rho})")
    print(f"     Marginals estimated via EDF")
    print("=" * 78)

    # Identify the Gaussian copula draws (index 2 = iii)
    gauss_key = list(draws.keys())[2]
    u1_g, u2_g = draws[gauss_key]

    # Transform to returns using the specified marginals
    from exercise1.portfolio import compute_portfolio_returns
    _, r1, r2 = compute_portfolio_returns(u1_g, u2_g, m1_spec, m2_spec, [0.5, 0.5])

    # Step I: EDF
    u1h = edf(r1)
    u2h = edf(r2)

    # Step II: maximize Gaussian copula LL
    result = estimate_gaussian_copula(u1h, u2h, bounds=tuple(bounds))

    print(f"  Estimated rho   = {result['rho']:.6f}")
    print(f"  True rho        = {true_rho}")
    print(f"  Log-likelihood  = {result['log_likelihood']:.2f}")
    print(f"  Converged       = {result['success']}")
    return result


# ─────────────────────────────────────────────────────────────
# Part 1.d
# ─────────────────────────────────────────────────────────────

def run_1d(cfg, fig_dir, dpi):
    """
    Mixture copula simulation + t copula IFM estimation.

    Mixture: p·Clayton(θ_a) + (1-p)·SurvClayton(θ_b).
    Marginals: both N(0,1) as specified in config.
    Estimation: t copula via IFM/CML + EDF.
    """
    mix = cfg["mixture"]
    sim_cfg = cfg["simulation"]
    est_cfg = cfg["estimation"]
    m1_spec = cfg["marginals_1d"]["stock1"]
    m2_spec = cfg["marginals_1d"]["stock2"]

    print(f"\n{'=' * 78}")
    print(f"1.d) Mixture copula: {mix['prob_clayton']}*Clayton({mix['clayton_theta']}) "
          f"+ {1 - mix['prob_clayton']}*SurvClayton({mix['surv_clayton_theta']})")
    print(f"     Marginals: N(0,1).  Estimating t copula via IFM + EDF.")
    print("=" * 78)

    # Simulate mixture
    np.random.seed(sim_cfg["seed_mixture"])
    Vm = np.random.rand(sim_cfg["Ns"], 2)
    vm1 = np.clip(Vm[:, 0], sim_cfg["eps"], 1 - sim_cfg["eps"])
    vm2 = np.clip(Vm[:, 1], sim_cfg["eps"], 1 - sim_cfg["eps"])
    indicator = np.random.rand(sim_cfg["Ns"]) < mix["prob_clayton"]

    u1m, u2m = sim_mixture(
        vm1, vm2, indicator,
        sim_clayton, {"theta": mix["clayton_theta"]},
        sim_surv_clayton, {"theta": mix["surv_clayton_theta"]}
    )

    # Dependence measures
    pr = pearsonr(u1m, u2m)[0]
    sr = spearmanr(u1m, u2m)[0]
    kt = kendalltau(u1m, u2m)[0]
    print(f"  Dependence:  Pearson={pr:.4f}  Spearman={sr:.4f}  Kendall={kt:.4f}")

    # Scatter plot
    title = (f"1.d) Mixture: {mix['prob_clayton']}*Clayton({mix['clayton_theta']}) "
             f"+ {1 - mix['prob_clayton']}*SurvClayton({mix['surv_clayton_theta']})")
    path = plot_mixture_scatter(u1m, u2m, title, fig_dir, dpi=dpi)
    print(f"  Figure saved: {path}")

    # Transform to returns (both N(0,1))
    from exercise1.portfolio import compute_portfolio_returns
    _, r1m, r2m = compute_portfolio_returns(u1m, u2m, m1_spec, m2_spec, [0.5, 0.5])

    # EDF
    u1hm = edf(r1m)
    u2hm = edf(r2m)

    # Estimate t copula
    res_t = estimate_t_copula(
        u1hm, u2hm,
        x0=tuple(est_cfg["t_copula_x0"]),
        maxiter=est_cfg["t_copula_maxiter"]
    )
    print(f"\n  t copula estimation (IFM + EDF):")
    print(f"    rho_hat  = {res_t['rho']:.6f}")
    print(f"    nu_hat   = {res_t['nu']:.4f}")
    print(f"    LL       = {res_t['log_likelihood']:.2f}")
    print(f"    Converged = {res_t['success']}")

    # Gaussian copula for comparison
    res_g = estimate_gaussian_copula(
        u1hm, u2hm,
        bounds=tuple(est_cfg["gaussian_bounds"])
    )
    print(f"\n  Gaussian copula estimation (IFM + EDF) for comparison:")
    print(f"    rho_hat  = {res_g['rho']:.6f}")
    print(f"    LL       = {res_g['log_likelihood']:.2f}")

    return res_t, res_g


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    out_cfg = cfg["output"]

    # Create output directory
    os.makedirs(out_cfg["figures_dir"], exist_ok=True)

    # Reproducible independent U(0,1) draws
    np.random.seed(sim_cfg["seed_main"])
    V = np.random.rand(sim_cfg["Ns"], 2)
    v1 = np.clip(V[:, 0], sim_cfg["eps"], 1 - sim_cfg["eps"])
    v2 = np.clip(V[:, 1], sim_cfg["eps"], 1 - sim_cfg["eps"])

    # Build copula registry from config
    copula_registry = build_copula_registry(cfg)

    # Run all parts
    draws, dep_rows, dep_values = run_1a(v1, v2, copula_registry,
                                          out_cfg["figures_dir"], out_cfg["dpi"])
    _, port_rows = run_1b(draws, cfg, out_cfg["figures_dir"], out_cfg["dpi"])
    res_1c = run_1c(draws, cfg)
    res_1d_t, res_1d_g = run_1d(cfg, out_cfg["figures_dir"], out_cfg["dpi"])

    # ------------------------------------------------------------------
    # Emit LaTeX-consumable results
    # ------------------------------------------------------------------
    # Recover 1.d dependence measures for the report (seed-reproducible)
    mix = cfg["mixture"]; sim_cfg = cfg["simulation"]
    np.random.seed(sim_cfg["seed_mixture"])
    Vm = np.random.rand(sim_cfg["Ns"], 2)
    vm1 = np.clip(Vm[:, 0], sim_cfg["eps"], 1 - sim_cfg["eps"])
    vm2 = np.clip(Vm[:, 1], sim_cfg["eps"], 1 - sim_cfg["eps"])
    indicator = np.random.rand(sim_cfg["Ns"]) < mix["prob_clayton"]
    u1m, u2m = sim_mixture(
        vm1, vm2, indicator,
        sim_clayton, {"theta": mix["clayton_theta"]},
        sim_surv_clayton, {"theta": mix["surv_clayton_theta"]})
    mix_pearson = pearsonr(u1m, u2m)[0]
    mix_spearman = spearmanr(u1m, u2m)[0]
    mix_kendall = kendalltau(u1m, u2m)[0]

    tex = TexWriter("Exercise 1 — auto-generated results")
    # Raw number (no $...$) so ``$N=\ExOneNs$'' composes without nested math.
    tex.cmd("ExOneNs", f"{sim_cfg['Ns']:,}".replace(",", r"\,"))
    tex.body("ExOneDependenceBody", dep_rows)
    tex.body("ExOnePortfolioBody", port_rows)

    # 1.c IFM Gaussian
    tex.cmd("ExOneICRhoTrue",
            fnum(cfg["copulas"]["gaussian"]["rho"], d=1))
    tex.cmd("ExOneICRhoHat",  fnum(res_1c["rho"], d=6))
    tex.cmd("ExOneICLogLik",  fnum(res_1c["log_likelihood"], d=2))

    # 1.d mixture dependence and t-copula / Gaussian IFM
    tex.cmd("ExOneIDMixPearson",  fnum(mix_pearson))
    tex.cmd("ExOneIDMixSpearman", fnum(mix_spearman))
    tex.cmd("ExOneIDMixKendall",  fnum(mix_kendall))
    tex.cmd("ExOneIDTRhoHat",   fnum(res_1d_t["rho"], d=6))
    tex.cmd("ExOneIDTNuHat",    fnum(res_1d_t["nu"], d=4))
    tex.cmd("ExOneIDTLogLik",   fnum(res_1d_t["log_likelihood"], d=2))
    tex.cmd("ExOneIDGRhoHat",   fnum(res_1d_g["rho"], d=6))
    tex.cmd("ExOneIDGLogLik",   fnum(res_1d_g["log_likelihood"], d=2))
    tex.cmd("ExOneIDLLGain",
            fnum(res_1d_t["log_likelihood"] - res_1d_g["log_likelihood"], d=2))

    out_path = os.path.join(generated_dir(cfg), "ex1.tex")
    tex.save(out_path)
    print(f"\n  LaTeX macros saved: {out_path}")

    print(f"\n{'=' * 78}")
    print(f"Exercise 1 completed. Figures saved in '{out_cfg['figures_dir']}/'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
