"""
Monte-Carlo driver for Exercise 5.

One Monte-Carlo replication:
  1. Simulate n_obs bivariate Clayton(theta_true) draws with N(0,1) margins
     (Berger 2016 Table 1 sets n_obs = 1001).
  2. Take r_{in} = first n_obs - 1 observations as the in-sample window.
  3. Fit Gaussian / Student-t / Clayton copulas on r_{in} (N(0,1) margins
     known, so this is a direct MLE — no marginal estimation step).
  4. From each fitted copula simulate n_sim forecast portfolio returns and
     read off the 5% and 1% quantiles.
  5. Record whether r_{p, n_obs} (the 1001st realisation) breaches each VaR
     forecast.

Across M replications we accumulate breach counts per (copula, alpha) pair
and feed them to the Kupiec UC test. A small ``n_jobs > 1`` runs the outer
loop in parallel.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .backtesting import kupiec_uc
from .dgp import simulate_clayton_returns
from .estimation import estimate_clayton, estimate_gaussian, estimate_t
from .var_forecast import simulate_portfolio_returns, var_quantiles

MODELS = ("gaussian", "t", "clayton")
ALPHAS = (0.01, 0.05)  # left-tail probabilities (99% and 95% VaR)


@dataclass
class ReplicationResult:
    breaches: dict               # {(model, alpha): int}
    fit_params: dict             # {model: dict of params}


def _one_replication(seed, theta_true, n_obs, n_sim, weights):
    rng = np.random.default_rng(seed)

    r, _ = simulate_clayton_returns(n_obs, theta_true, rng)
    r_in = r[:-1]
    r1, r2 = r_in[:, 0], r_in[:, 1]

    # Realisation we will check against (the "1001st" observation).
    r_out = r[-1]
    w1, w2 = weights
    r_p_out = w1 * r_out[0] + w2 * r_out[1]

    # Fit each model.
    g_fit = estimate_gaussian(r1, r2)
    t_fit = estimate_t(r1, r2)
    c_fit = estimate_clayton(r1, r2)
    fits = {
        "gaussian": {"rho": g_fit["rho"]},
        "t": {"rho": t_fit["rho"], "nu": t_fit["nu"]},
        "clayton": {"theta": c_fit["theta"]},
    }

    breaches = {}
    for name, params in fits.items():
        rp_sim = simulate_portfolio_returns(name, params, n_sim, rng, weights)
        qs = var_quantiles(rp_sim, ALPHAS)
        for a in ALPHAS:
            breaches[(name, a)] = int(r_p_out < qs[a])

    return ReplicationResult(breaches=breaches, fit_params=fits)


def _run_worker(args):
    return _one_replication(*args)


def run_scenario(theta_true, *, n_obs, n_sim, n_rep, weights, base_seed, n_jobs=1):
    """
    Execute n_rep replications for a given true Clayton theta.

    Returns a dict keyed by (model, alpha) with aggregated breach count and
    Kupiec test output, plus arrays of fitted parameters.
    """
    seeds = [base_seed + k for k in range(n_rep)]
    jobs = [(s, theta_true, n_obs, n_sim, weights) for s in seeds]

    counters = {(m, a): 0 for m in MODELS for a in ALPHAS}
    fits_history = {m: [] for m in MODELS}

    if n_jobs <= 1:
        for args in jobs:
            rep = _run_worker(args)
            for k, v in rep.breaches.items():
                counters[k] += v
            for m, pars in rep.fit_params.items():
                fits_history[m].append(pars)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = [pool.submit(_run_worker, a) for a in jobs]
            for f in as_completed(futs):
                rep = f.result()
                for k, v in rep.breaches.items():
                    counters[k] += v
                for m, pars in rep.fit_params.items():
                    fits_history[m].append(pars)

    kupiec = {}
    for m in MODELS:
        for a in ALPHAS:
            kupiec[(m, a)] = kupiec_uc(counters[(m, a)], n_rep, a)

    return {
        "theta_true": theta_true,
        "n_rep": n_rep,
        "counters": counters,
        "kupiec": kupiec,
        "fits": fits_history,
    }
