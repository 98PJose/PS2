"""
Exercise 5 orchestrator — Berger (2016) simulation study.

DGP: bivariate Clayton copula with N(0,1) margins, three parameter values
(theta in {0.5, 1.5, 2.5} as in Berger Table 1).
Candidate models: Gaussian, Student t, and Clayton copulas with N(0,1)
margins.
Evaluation: Kupiec unconditional-coverage test applied to the aggregated
breach counts from ``n_rep`` Monte-Carlo replications.

Run with:
    cd PS2
    python -m exercise5.main
"""

from __future__ import annotations

import json
import os

import numpy as np

from .plotting import plot_failure_rates
from .simulation import ALPHAS, MODELS, run_scenario
from report_utils import TexWriter, fnum, generated_dir


def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _format_rate(kp):
    """'6.12%**' style formatting, matching Berger Table 2."""
    rate = 100.0 * kp["hit_rate"]
    mark = "**" if kp["reject_99"] else ("*" if kp["reject_95"] else "")
    return f"{rate:5.2f}%{mark:<2}"


def format_table(results, alphas):
    alphas = sorted(alphas, reverse=True)  # 0.05 then 0.01 (95% then 99% VaR)
    lines = []
    lines.append("=" * 78)
    lines.append("Empirical VaR failure rates — Clayton DGP, 3 candidate models")
    lines.append("  Rejections: *  -> 95% significance,  ** -> 99% significance")
    lines.append("=" * 78)

    for a in alphas:
        title = f"{int((1 - a) * 100)}% VaR  (nominal alpha = {a:.0%})"
        lines.append("")
        lines.append(title)
        lines.append(f"  {'DGP theta':<12}" + "".join(f"  {m:>10s}" for m in MODELS))
        for r in results:
            row = f"  theta = {r['theta_true']:<4g}"
            for m in MODELS:
                row += f"  {_format_rate(r['kupiec'][(m, a)]):>10s}"
            lines.append(row)
    return "\n".join(lines)


def format_fit_summary(results):
    lines = []
    lines.append("=" * 78)
    lines.append("Average fitted parameters across replications")
    lines.append("=" * 78)
    for r in results:
        lines.append("")
        lines.append(f"  DGP Clayton theta = {r['theta_true']}")
        g_rhos = [f["rho"] for f in r["fits"]["gaussian"]]
        t_rhos = [f["rho"] for f in r["fits"]["t"]]
        t_nus = [f["nu"] for f in r["fits"]["t"]]
        c_thetas = [f["theta"] for f in r["fits"]["clayton"]]
        lines.append(f"    Gaussian: rho_hat   = {np.mean(g_rhos):+.4f}  "
                     f"(sd {np.std(g_rhos):.4f})")
        lines.append(f"    t       : rho_hat   = {np.mean(t_rhos):+.4f}  "
                     f"(sd {np.std(t_rhos):.4f})  "
                     f"nu_hat = {np.mean(t_nus):.2f}  (sd {np.std(t_nus):.2f})")
        lines.append(f"    Clayton : theta_hat = {np.mean(c_thetas):+.4f}  "
                     f"(sd {np.std(c_thetas):.4f})")
    return "\n".join(lines)


def main():
    cfg = load_config()
    ex5 = cfg["exercise5"]

    print("=" * 78)
    print("Exercise 5 — Berger (2016) simulation study")
    print(f"  DGP = Clayton, thetas = {ex5['theta_values']}")
    print(f"  n_obs = {ex5['n_obs']}, n_sim = {ex5['n_sim']}, "
          f"n_rep = {ex5['n_rep']}, n_jobs = {ex5.get('n_jobs', 1)}")
    print("=" * 78)

    results = []
    base_seed = ex5["base_seed"]
    for idx, theta in enumerate(ex5["theta_values"]):
        print(f"\n[scenario {idx + 1}/{len(ex5['theta_values'])}] "
              f"theta_true = {theta} — running {ex5['n_rep']} replications...")
        out = run_scenario(
            theta_true=theta,
            n_obs=ex5["n_obs"],
            n_sim=ex5["n_sim"],
            n_rep=ex5["n_rep"],
            weights=tuple(ex5["weights"]),
            base_seed=base_seed + idx * 1_000_000,
            n_jobs=ex5.get("n_jobs", 1))
        results.append(out)

    table_txt = format_table(results, alphas=ALPHAS)
    fits_txt = format_fit_summary(results)
    print("\n" + table_txt)
    print("\n" + fits_txt)

    figures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", cfg["output"]["figures_dir"])
    os.makedirs(figures_dir, exist_ok=True)
    out_txt = os.path.join(figures_dir, "tab_5_failure_rates.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(table_txt + "\n\n" + fits_txt + "\n")
    print(f"\n[exercise5] saved {out_txt}")

    plot_failure_rates(results, alphas=ALPHAS, figures_dir=figures_dir,
                       dpi=cfg["output"]["dpi"])

    # ------------------------------------------------------------------
    # Emit LaTeX-consumable results
    # ------------------------------------------------------------------
    def _rate_cell(kp):
        rate = 100.0 * kp["hit_rate"]
        if kp["reject_99"]:
            mark = r"^{**}"
        elif kp["reject_95"]:
            mark = r"^{*}"
        else:
            mark = ""
        return f"${rate:.2f}{mark}$"

    tex = TexWriter("Exercise 5 — auto-generated results")
    # Raw numbers (no $...$) so ``$R=\ExFiveNrep$'' composes without nested math.
    tex.cmd("ExFiveNobs",  f"{ex5['n_obs']:,}".replace(",", r"\,"))
    tex.cmd("ExFiveNsim",  f"{ex5['n_sim']:,}".replace(",", r"\,"))
    tex.cmd("ExFiveNrep",  f"{ex5['n_rep']:,}".replace(",", r"\,"))

    # Two breach-rate tables (95% then 99%)
    alphas_sorted = sorted(ALPHAS, reverse=True)
    for alpha, tag in zip(alphas_sorted, ("NinetyFive", "NinetyNine")):
        rows = []
        for r in results:
            cells = [f"${r['theta_true']}$"]
            for m in MODELS:
                cells.append(_rate_cell(r["kupiec"][(m, alpha)]))
            cells.append(f"${100*alpha:.2f}$")
            rows.append(" & ".join(cells) + r"\\")
        tex.body(f"ExFiveBreach{tag}Body", rows)

    # Average-fits table
    fit_rows = []
    for r in results:
        g_rhos = [f["rho"] for f in r["fits"]["gaussian"]]
        t_rhos = [f["rho"] for f in r["fits"]["t"]]
        t_nus  = [f["nu"]  for f in r["fits"]["t"]]
        c_ths  = [f["theta"] for f in r["fits"]["clayton"]]
        nu_mean = float(np.mean(t_nus))
        # Display very large nu estimates as infinity (t-copula degenerating
        # to Gaussian at low Clayton dependence).
        nu_cell = r"$\infty$" if nu_mean > 100.0 else fnum(nu_mean, 2)
        row = (
            f"${r['theta_true']}$ "
            f"& {fnum(np.mean(g_rhos))} & {fnum(np.std(g_rhos))} "
            f"& {fnum(np.mean(t_rhos))} & {nu_cell} "
            f"& {fnum(np.mean(c_ths))} & {fnum(np.std(c_ths))}\\\\"
        )
        fit_rows.append(row)
    tex.body("ExFiveFitsBody", fit_rows)

    out_path = os.path.join(generated_dir(cfg), "ex5.tex")
    tex.save(out_path)
    print(f"\n[exercise5] LaTeX macros saved: {out_path}")


if __name__ == "__main__":
    main()
