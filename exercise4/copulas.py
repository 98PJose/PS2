"""
Bivariate shock generators with N(0,1) marginals under alternative copulas.

Each function returns an (N, 2) array of standardised shocks (Z_1, Z_2) with
unit-variance Normal marginals and the requested dependence structure. These
are the `eta` variables used in the slides (page 14): the stock returns are

    R_i = (mu_i - sigma_i^2 / 2) * T + sigma_i * sqrt(T) * Z_i,
    S_iT = S_i0 * exp(R_i).

Copula models covered (Exercise 4):
  * Gaussian (4.1, 4.2): direct Cholesky-style decomposition.
  * Student t (4.3): simulated via the conditional method in
    exercise1.copula_sim.sim_student_t, then Z_i = Phi^{-1}(U_i).
  * Clayton (4.4): simulated via exercise1.copula_sim.sim_clayton, then
    Z_i = Phi^{-1}(U_i).

The Gaussian implementation matches the MATLAB reference ccK_simuRet2.m
exactly (linear combination of two independent N(0,1) draws).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from exercise1.copula_sim import sim_clayton, sim_student_t


def _draw_uniforms(rng, n):
    """Two independent U(0,1) vectors (avoids exact 0/1 for Phi^{-1})."""
    v1 = rng.random(n)
    v2 = rng.random(n)
    return v1, v2


def gaussian_shocks(n, rho, rng):
    """
    (Z_1, Z_2) with Gaussian copula and N(0,1) marginals.

    Uses the explicit Cholesky recombination
        Z_1 = eps_1, Z_2 = rho * eps_1 + sqrt(1 - rho^2) * eps_2
    with eps_i ~ N(0,1) iid (matches MATLAB ccK_simuRet2.m).
    """
    eps1 = rng.standard_normal(n)
    eps2 = rng.standard_normal(n)
    z1 = eps1
    z2 = rho * eps1 + np.sqrt(1.0 - rho**2) * eps2
    return np.column_stack([z1, z2])


def t_shocks(n, rho, nu, rng, eps=1e-12):
    """
    (Z_1, Z_2) with Student t copula (rho, nu) and N(0,1) marginals.

    Simulates dependent uniforms via sim_student_t, then applies Phi^{-1}.
    The uniforms are clipped to (eps, 1-eps) to keep Phi^{-1} finite.
    """
    v1, v2 = _draw_uniforms(rng, n)
    u1, u2 = sim_student_t(v1, v2, rho, nu)
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    return np.column_stack([norm.ppf(u1), norm.ppf(u2)])


def clayton_shocks(n, theta, rng, eps=1e-12):
    """
    (Z_1, Z_2) with Clayton copula (theta) and N(0,1) marginals.

    Simulates dependent uniforms via sim_clayton, then applies Phi^{-1}.
    """
    v1, v2 = _draw_uniforms(rng, n)
    u1, u2 = sim_clayton(v1, v2, theta)
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    return np.column_stack([norm.ppf(u1), norm.ppf(u2)])


def sample_shocks(kind, n, rng, **params):
    """
    Dispatcher keyed by copula name.

    Parameters
    ----------
    kind : {"gaussian", "t", "clayton"}
    n : int
    rng : numpy Generator
    params : dict
        Copula-specific parameters (rho for gaussian; rho, nu for t; theta for
        clayton).
    """
    if kind == "gaussian":
        return gaussian_shocks(n, params["rho"], rng)
    if kind == "t":
        return t_shocks(n, params["rho"], params["nu"], rng)
    if kind == "clayton":
        return clayton_shocks(n, params["theta"], rng)
    raise ValueError(f"unknown copula kind: {kind}")
