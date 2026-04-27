"""
Exercise 6 orchestrator -- Lo (2007), Section 4.3 stop-loss policy.

Steps:
    1. read parameters from ``config.json["exercise6"]``;
    2. evaluate eq. (26) closed-form on the (rho, zeta) grid that mirrors
       Lo's Table 1;
    3. simulate the AR(1) + stop-loss process (eqs. 23-25) on the same
       grid and contrast simulated AP decomposition against the
       closed-form values (annualized);
    4. emit figures, a plain-text dump and the LaTeX macros for the
       report.

Run with:
    cd PS2
    python -m exercise6.main
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .dgp import simulate_ar1
from .plotting import plot_sample_path, plot_table1
from .policy import apply_stop_loss
from .simulation import run_grid
from .theory import expected_return_closed_form
from report_utils import TexWriter, fnum, generated_dir


def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _annualise(x, freq):
    return x * freq


def format_console_tables(results, freq):
    """Pretty-print Lo's Table 1 (closed form) and the simulated copy."""
    rhos = sorted({c["rho"] for c in results})
    zetas = sorted({c["zeta"] for c in results})

    def _block(getter, title):
        lines = []
        lines.append("=" * 88)
        lines.append(title)
        lines.append("=" * 88)
        lines.append(
            f"  {'zeta':>6}  {'rho':>5}  {'E[R_p]':>8}  {'Active':>8}"
            f"  {'Passive':>8}  {'%Active':>9}  {'E[w]':>7}  {'1-E[w]':>7}"
        )
        lines.append("-" * 88)
        for rho in rhos:
            for zeta in zetas:
                c = next(c for c in results
                         if c["rho"] == rho and c["zeta"] == zeta)
                d = getter(c)
                lines.append(
                    f"  {zeta * 100:>+5.1f}%  {rho * 100:>+4.0f}%  "
                    f"{_annualise(d['E_Rp'], freq) * 100:>7.1f}%  "
                    f"{_annualise(d['active'], freq) * 100:>7.1f}%  "
                    f"{_annualise(d['passive'], freq) * 100:>7.1f}%  "
                    f"{d['pct_active'] * 100:>8.1f}%  "
                    f"{d['E_omega'] * 100:>6.1f}%  "
                    f"{(1.0 - d['E_omega']) * 100:>6.1f}%"
                )
            lines.append("")
        return "\n".join(lines)

    th = _block(lambda c: c["theory"],
                "Lo (2007) Table 1 -- closed form (eq. 26), annualized")
    sm = _block(lambda c: c["sim"],
                "Lo (2007) Table 1 -- Monte-Carlo simulation, annualized")
    return th + "\n" + sm


def _max_abs_diff(results, freq):
    """Largest |theory - sim| of E[R_pt] across the grid (annualized %)."""
    diffs = []
    for c in results:
        diffs.append(
            abs(c["theory"]["E_Rp"] - c["sim"]["E_Rp"]) * freq * 100.0
        )
    return max(diffs)


def main():
    cfg = load_config()
    ex6 = cfg["exercise6"]

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    freq = ex6["freq_per_year"]
    R_f = ex6["Rf_annual"] / freq
    mu = ex6["mu_annual"] / freq
    # Var[R_t] = sigma_annual^2 / freq, so unconditional sd:
    sigma = ex6["sigma_annual"] / math.sqrt(freq)
    rho_grid = ex6["rho_grid"]
    zeta_grid = ex6["zeta_grid"]
    n_obs = ex6["n_obs"]
    base_seed = ex6["base_seed"]

    print("=" * 88)
    print("Exercise 6 -- Lo (2007), eq. (26) verification by simulation")
    print(f"  R_f={R_f:.6f}  mu={mu:.6f}  sigma={sigma:.6f}  freq={freq}")
    print(f"  rho_grid  = {rho_grid}")
    print(f"  zeta_grid = {zeta_grid}")
    print(f"  n_obs = {n_obs:,}, base_seed = {base_seed}")
    print("=" * 88)

    # ------------------------------------------------------------------
    # Grid: closed-form + simulation
    # ------------------------------------------------------------------
    results = run_grid(
        mu=mu, sigma=sigma, R_f=R_f,
        rho_grid=rho_grid, zeta_grid=zeta_grid,
        n_obs=n_obs, base_seed=base_seed,
    )
    txt = format_console_tables(results, freq)
    print("\n" + txt)
    max_diff = _max_abs_diff(results, freq)
    print(f"\n[exercise6] max |theory - sim| of annualized E[R_pt] = "
          f"{max_diff:.3f} %")

    # ------------------------------------------------------------------
    # Save text dump + figures
    # ------------------------------------------------------------------
    figures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", cfg["output"]["figures_dir"],
    )
    os.makedirs(figures_dir, exist_ok=True)
    out_txt = os.path.join(figures_dir, "tab_6_table1.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt + f"\n\nmax|theory-sim| = {max_diff:.3f} %\n")
    print(f"[exercise6] saved {out_txt}")

    plot_table1(results, freq=freq, figures_dir=figures_dir,
                dpi=cfg["output"]["dpi"])

    # Single illustrative path (independent RNG)
    rng = np.random.default_rng(base_seed - 1)
    R_path = simulate_ar1(
        n_obs=ex6["sample_path_n"],
        mu=mu, rho=ex6["sample_path_rho"], sigma=sigma, rng=rng,
    )
    R_p, omega, _ = apply_stop_loss(
        R_path, zeta=ex6["sample_path_zeta"], R_f=R_f,
    )
    plot_sample_path(R=R_path, zeta=ex6["sample_path_zeta"], R_f=R_f,
                     omega=omega, R_p=R_p, figures_dir=figures_dir,
                     dpi=cfg["output"]["dpi"], rho=ex6["sample_path_rho"])

    # ------------------------------------------------------------------
    # LaTeX macros
    # ------------------------------------------------------------------
    tex = TexWriter("Exercise 6 -- auto-generated results")
    tex.cmd("ExSixNobs",     f"{n_obs:,}".replace(",", r"\,"))
    tex.cmd("ExSixSeed",     f"{base_seed}")
    tex.cmd("ExSixFreq",     f"{freq}")
    tex.cmd("ExSixRfPct",    f"{ex6['Rf_annual'] * 100:.1f}")
    tex.cmd("ExSixMuPct",    f"{ex6['mu_annual']  * 100:.1f}")
    tex.cmd("ExSixSigmaPct", f"{ex6['sigma_annual'] * 100:.1f}")
    tex.cmd("ExSixMaxDiff",  fnum(max_diff, 3))
    tex.cmd("ExSixSamplePathRho",
            f"{ex6['sample_path_rho']:.2f}")
    tex.cmd("ExSixSamplePathZetaPct",
            f"{ex6['sample_path_zeta'] * 100:.1f}")
    tex.cmd("ExSixSamplePathN",
            f"{ex6['sample_path_n']}")

    def _row(c, getter):
        d = getter(c)
        return (
            f"${c['zeta'] * 100:+.1f}\\%$ "
            f"& ${c['rho']  * 100:+.0f}\\%$ "
            f"& ${_annualise(d['E_Rp'],    freq) * 100:.1f}\\%$ "
            f"& ${_annualise(d['active'],  freq) * 100:.1f}\\%$ "
            f"& ${_annualise(d['passive'], freq) * 100:.1f}\\%$ "
            f"& ${d['pct_active'] * 100:.1f}\\%$ "
            f"& ${d['E_omega'] * 100:.1f}\\%$ "
            f"& ${(1.0 - d['E_omega']) * 100:.1f}\\%$\\\\"
        )

    rhos_sorted = sorted({c["rho"] for c in results})
    zetas_sorted = sorted({c["zeta"] for c in results})
    ordered = []
    for rho in rhos_sorted:
        for zeta in zetas_sorted:
            ordered.append(
                next(c for c in results
                     if c["rho"] == rho and c["zeta"] == zeta)
            )
        ordered.append("midrule")  # spacer between rho blocks

    theory_rows = []
    sim_rows = []
    for item in ordered:
        if item == "midrule":
            theory_rows.append(r"\midrule")
            sim_rows.append(r"\midrule")
        else:
            theory_rows.append(_row(item, lambda c: c["theory"]))
            sim_rows.append(_row(item, lambda c: c["sim"]))
    # remove the trailing midrule at the very end of each block
    if theory_rows and theory_rows[-1] == r"\midrule":
        theory_rows.pop()
    if sim_rows and sim_rows[-1] == r"\midrule":
        sim_rows.pop()

    tex.body("ExSixTableTheoryBody", theory_rows)
    tex.body("ExSixTableSimBody",    sim_rows)

    out_path = os.path.join(generated_dir(cfg), "ex6.tex")
    tex.save(out_path)
    print(f"\n[exercise6] LaTeX macros saved: {out_path}")


if __name__ == "__main__":
    main()
