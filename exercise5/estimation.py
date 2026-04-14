"""
Copula MLE with KNOWN N(0,1) marginals (Berger 2016, Section 2.1).

Because the exercise fixes the marginal distribution to N(0,1), the IFM
two-step method collapses to a single step: transform returns to uniforms
via u_i = Phi(r_i), then maximise the copula log-likelihood over its
parameter(s). This is different from Exercise 1.c / 1.d where marginals were
estimated via the EDF. Here no marginal estimation error enters the copula
parameter estimator, matching the design of Berger 2016 whose stated goal is
to "isolate the impact of competing copula approaches" from marginal
misspecification.

Gaussian and Student-t log-likelihoods reuse the formulas from
``exercise1.copula_likelihood``. The Clayton log-likelihood is implemented
here because Exercise 1 simulated Clayton draws but never estimated them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from exercise1.copula_likelihood import ll_gaussian_copula, ll_t_copula


# ------------------------------------------------------------------ Clayton

def clayton_log_density(u1, u2, theta, eps=1e-12):
    """
    Log density of the bivariate Clayton copula.

    c^Cla(u_1, u_2; theta) = (1 + theta) (u_1 u_2)^{-(1 + theta)}
                             * (u_1^{-theta} + u_2^{-theta} - 1)^{-(2 + 1/theta)}
    =>
    ln c = ln(1 + theta) - (1 + theta) (ln u_1 + ln u_2)
           - (2 + 1/theta) * ln(u_1^{-theta} + u_2^{-theta} - 1).

    The formula is defined for theta > 0. ``eps`` clips the uniforms to
    (eps, 1 - eps) for numerical stability.
    """
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    t = theta
    A = u1**(-t) + u2**(-t) - 1.0
    return (
        np.log1p(t)
        - (1.0 + t) * (np.log(u1) + np.log(u2))
        - (2.0 + 1.0 / t) * np.log(A)
    )


def clayton_neg_ll(theta, u1, u2):
    return -float(np.sum(clayton_log_density(u1, u2, theta)))


def estimate_clayton(r1, r2, bounds=(0.01, 30.0)):
    """MLE of Clayton theta from returns (N(0,1) marginals)."""
    u1 = norm.cdf(r1)
    u2 = norm.cdf(r2)
    res = minimize_scalar(
        clayton_neg_ll, args=(u1, u2),
        bounds=bounds, method="bounded",
        options={"xatol": 1e-6})
    return {"theta": float(res.x), "ll": float(-res.fun), "success": res.success}


# ------------------------------------------------------------------ Gaussian

def estimate_gaussian(r1, r2, bounds=(-0.9999, 0.9999)):
    """MLE of Gaussian rho from returns (N(0,1) marginals)."""
    u1 = norm.cdf(r1)
    u2 = norm.cdf(r2)

    def neg_ll(rho):
        return -float(ll_gaussian_copula(rho, u1, u2))

    res = minimize_scalar(neg_ll, bounds=bounds, method="bounded",
                          options={"xatol": 1e-6})
    return {"rho": float(res.x), "ll": float(-res.fun), "success": res.success}


# ------------------------------------------------------------------ Student t

def estimate_t(r1, r2, x0=(0.5, 8.0)):
    """
    MLE of Student-t copula (rho, nu) from returns (N(0,1) marginals).

    Uses Nelder-Mead on the transform (rho, ln(nu - 2)) to enforce rho in
    (-1, 1) and nu > 2 (finite variance).
    """
    u1 = norm.cdf(r1)
    u2 = norm.cdf(r2)

    def to_params(x):
        rho = np.tanh(x[0])
        nu = 2.0 + np.exp(x[1])
        return rho, nu

    def neg_ll(x):
        rho, nu = to_params(x)
        return -float(ll_t_copula(rho, nu, u1, u2))

    x0_t = (np.arctanh(x0[0]), np.log(x0[1] - 2.0))
    res = minimize(neg_ll, x0_t, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
    rho, nu = to_params(res.x)
    return {"rho": float(rho), "nu": float(nu), "ll": float(-res.fun),
            "success": bool(res.success)}
