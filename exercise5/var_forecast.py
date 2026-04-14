"""
VaR forecasting via copula simulation (Berger 2016, Section 2.1.1).

For each fitted copula:
  1. Simulate N_sim bivariate (u_1, u_2) draws from the fitted copula.
  2. Transform to returns r_i = Phi^{-1}(u_i)  (N(0,1) marginals).
  3. Form the equal-weighted portfolio return  r_p = 0.5 r_1 + 0.5 r_2.
  4. The alpha VaR forecast is the empirical alpha-quantile of r_p.

By convention VaR is reported as a signed quantile (negative in the loss
tail). A VaR breach is recorded whenever the realised portfolio return on
day t+1 falls below the forecast.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from exercise1.copula_sim import sim_clayton, sim_gaussian, sim_student_t


def simulate_portfolio_returns(copula_name, params, n_sim, rng, weights=(0.5, 0.5)):
    """Return n_sim equal-weighted portfolio returns from the fitted copula."""
    v1 = rng.random(n_sim)
    v2 = rng.random(n_sim)

    if copula_name == "gaussian":
        u1, u2 = sim_gaussian(v1, v2, params["rho"])
    elif copula_name == "t":
        u1, u2 = sim_student_t(v1, v2, params["rho"], params["nu"])
    elif copula_name == "clayton":
        u1, u2 = sim_clayton(v1, v2, params["theta"])
    else:
        raise ValueError(f"unknown copula: {copula_name}")

    eps = 1e-12
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    r1 = norm.ppf(u1)
    r2 = norm.ppf(u2)
    w1, w2 = weights
    return w1 * r1 + w2 * r2


def var_quantiles(portfolio_returns, alphas):
    """Empirical alpha-quantiles (signed; negative in the loss tail)."""
    return {a: float(np.quantile(portfolio_returns, a)) for a in alphas}
