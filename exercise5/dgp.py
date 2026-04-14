"""
Data generating process for Exercise 5 — bivariate Clayton copula with
N(0,1) marginals.

For each Monte-Carlo replication we generate ``n_obs`` independent Clayton
draws (u_1, u_2) at the true theta and transform to returns via
    r_i = Phi^{-1}(u_i),  i = 1, 2.

The first n_obs - 1 observations form the "in-sample" window for fitting the
candidate copulas; the last observation is the one-step-ahead realisation
against which each VaR forecast is checked (Berger 2016, Section 3.1).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from exercise1.copula_sim import sim_clayton


def simulate_clayton_returns(n_obs, theta, rng, eps=1e-12):
    """
    Simulate ``n_obs`` bivariate draws from Clayton(theta) with N(0,1)
    marginals.

    Parameters
    ----------
    n_obs : int
    theta : float
        Clayton dependence parameter (> 0).
    rng : numpy Generator
    eps : float
        Clipping applied to uniforms before Phi^{-1} to keep returns finite.

    Returns
    -------
    r : (n_obs, 2) ndarray of N(0,1) returns
    u : (n_obs, 2) ndarray of dependent U(0,1) draws
    """
    v1 = rng.random(n_obs)
    v2 = rng.random(n_obs)
    u1, u2 = sim_clayton(v1, v2, theta)
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    r = np.column_stack([norm.ppf(u1), norm.ppf(u2)])
    u = np.column_stack([u1, u2])
    return r, u
