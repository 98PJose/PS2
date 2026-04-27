"""
Monte-Carlo verification of Lo (2007), eq. (26).

For each (rho, zeta) cell of Lo's Table 1 we
    1. simulate a long AR(1) path of strategy returns (eq. 23),
    2. apply the stop-loss switch (eqs. 24-25),
    3. estimate the AP decomposition by sample moments
       (Cov[omega, R], E[omega], 1-E[omega]),
    4. report E[R_pt] together with a Monte-Carlo standard error,
and compare the results with the closed-form prediction in
``theory.expected_return_closed_form``.
"""

from __future__ import annotations

import math

import numpy as np

from .dgp import simulate_ar1
from .policy import apply_stop_loss
from .theory import expected_return_closed_form


def simulate_one_cell(mu, rho, sigma, zeta, R_f, n_obs, rng):
    """
    Simulate a single (rho, zeta) cell and return both the simulated and
    the closed-form moments.

    The active / passive components on the simulation side use Lo's
    sample-moment identity (22a):
        Cov[omega, R] + E[omega] E[R] + (1 - E[omega]) R_f.
    """
    R = simulate_ar1(n_obs=n_obs, mu=mu, rho=rho, sigma=sigma, rng=rng)
    R_p, omega, R_t = apply_stop_loss(R, zeta=zeta, R_f=R_f)

    n = R_p.size
    mean_Rp = float(R_p.mean())
    sd_Rp = float(R_p.std(ddof=1))
    se_Rp = sd_Rp / math.sqrt(n)

    E_omega_hat = float(omega.mean())
    E_R_hat = float(R_t.mean())
    cov_omega_R = float(((omega - E_omega_hat) * (R_t - E_R_hat)).mean())

    active_sim = cov_omega_R
    passive_sim = E_omega_hat * E_R_hat + (1.0 - E_omega_hat) * R_f
    pct_active_sim = active_sim / mean_Rp if mean_Rp != 0.0 else float("nan")

    theory = expected_return_closed_form(
        mu=mu, rho=rho, sigma=sigma, zeta=zeta, R_f=R_f,
    )

    return {
        "rho": rho,
        "zeta": zeta,
        "n": n,
        # simulation side
        "sim": {
            "E_Rp": mean_Rp,
            "se_Rp": se_Rp,
            "active": active_sim,
            "passive": passive_sim,
            "E_omega": E_omega_hat,
            "pct_active": pct_active_sim,
        },
        # closed-form side (eq. 26)
        "theory": theory,
    }


def run_grid(mu, sigma, R_f, rho_grid, zeta_grid, n_obs, base_seed):
    """
    Loop over the (rho, zeta) grid that mirrors Lo's Table 1.

    Returns a list of cell dicts in row-major order:
    rho varies slowest, zeta varies fastest -- matching Lo's printed
    ordering.
    """
    results = []
    cell_id = 0
    for rho in rho_grid:
        for zeta in zeta_grid:
            rng = np.random.default_rng(base_seed + cell_id)
            cell = simulate_one_cell(
                mu=mu, rho=rho, sigma=sigma, zeta=zeta, R_f=R_f,
                n_obs=n_obs, rng=rng,
            )
            results.append(cell)
            cell_id += 1
    return results
