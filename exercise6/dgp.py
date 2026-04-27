"""
Data generating process for Exercise 6 — Lo (2007), eq. (23).

The risky asset follows a stationary AR(1) with Gaussian innovations:

    R_t = mu + rho (R_{t-1} - mu) + eps_t,    eps_t ~ IID N(0, sigma_e^2),

with unconditional mean ``mu`` and unconditional variance
``sigma**2 = sigma_e**2 / (1 - rho**2)``. To avoid a transient burn-in we
draw the initial state from the stationary distribution:

    R_0 ~ N(mu, sigma**2).
"""

from __future__ import annotations

import numpy as np


def simulate_ar1(n_obs, mu, rho, sigma, rng):
    """
    Simulate ``n_obs`` consecutive observations of an AR(1) process started
    in its stationary distribution.

    Parameters
    ----------
    n_obs : int
        Length of the series.
    mu : float
        Unconditional mean of R_t.
    rho : float
        AR(1) coefficient (|rho| < 1).
    sigma : float
        Unconditional standard deviation of R_t (NOT the innovation sd).
    rng : numpy.random.Generator

    Returns
    -------
    R : (n_obs,) ndarray
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie strictly inside (-1, 1) for stationarity")
    sigma_e = sigma * np.sqrt(1.0 - rho * rho)
    eps = rng.normal(loc=0.0, scale=sigma_e, size=n_obs)
    R = np.empty(n_obs)
    R[0] = mu + rng.normal(loc=0.0, scale=sigma)        # stationary R_0
    for t in range(1, n_obs):
        R[t] = mu + rho * (R[t - 1] - mu) + eps[t]
    return R
