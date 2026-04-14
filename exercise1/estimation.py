"""
Estimation routines for copula models.

Implements:
- Empirical Distribution Function (EDF) for marginal estimation.
- IFM (Inference Functions for Margins) / CML (Canonical ML) estimation
  of Gaussian and Student t copula parameters.

The IFM method (Joe, 1997) proceeds in two steps:
    Step I:  Estimate marginals (here via EDF, making this effectively CML).
    Step II: Maximize copula log-likelihood given pseudo-observations.

References: Copula_slides.pdf, sections 11.2 and 11.3.
"""

import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar, minimize
from exercise1.copula_likelihood import ll_gaussian_copula, ll_t_copula


def edf(x):
    """
    Empirical Distribution Function using the Weibull plotting position.

    For a sample x₁, ..., x_T, the EDF assigns to the k-th order statistic:
        F̂(x_{(k)}) = k / (T + 1)

    This avoids boundary values 0 and 1, which would cause Φ^{-1}(0) = -∞
    or t_ν^{-1}(1) = +∞ in the copula likelihood evaluation.

    Alternative conventions exist (e.g., k/T from slide 84, or (k-0.5)/T
    Hazen position), but rank/(T+1) is standard in copula estimation
    literature (Genest & Favre, 2007).

    Parameters
    ----------
    x : ndarray
        Sample of observations, shape (T,).

    Returns
    -------
    ndarray
        Pseudo-observations in (0, 1), shape (T,).
    """
    return rankdata(x) / (len(x) + 1)


def estimate_gaussian_copula(u1, u2, bounds=(-0.9999, 0.9999)):
    """
    Estimate the Gaussian copula parameter ρ via maximum likelihood.

    This is Step II of IFM/CML: given pseudo-observations (u1, u2),
    maximize LL(ρ) = Σ ln c^G(u1_t, u2_t; ρ) over ρ ∈ (-1, 1).

    Uses scipy.optimize.minimize_scalar with bounded method, which
    applies Brent's algorithm within the interval.

    Parameters
    ----------
    u1, u2 : ndarray
        Pseudo-observations in (0,1), shape (T,).
    bounds : tuple of float
        Search bounds for ρ.

    Returns
    -------
    dict
        Keys: 'rho' (estimated ρ), 'log_likelihood' (LL at optimum),
        'success' (convergence flag).
    """
    result = minimize_scalar(
        lambda rho: -ll_gaussian_copula(rho, u1, u2),
        bounds=bounds,
        method="bounded"
    )
    return {
        "rho": result.x,
        "log_likelihood": -result.fun,
        "success": result.success
    }


def estimate_t_copula(u1, u2, x0=(0.5, 5.0), maxiter=10000):
    """
    Estimate the Student t copula parameters (ρ, ν) via maximum likelihood.

    This is Step II of IFM/CML: given pseudo-observations (u1, u2),
    maximize LL(ρ, ν) = Σ ln c^t(u1_t, u2_t; ρ, ν) over ρ ∈ (-1,1), ν > 2.

    Uses Nelder-Mead (derivative-free) because the objective involves
    tdist.ppf which is not analytically differentiable. The parameter space
    is enforced by returning a large penalty for invalid parameters.

    Parameters
    ----------
    u1, u2 : ndarray
        Pseudo-observations in (0,1), shape (T,).
    x0 : tuple of float
        Initial guess (ρ₀, ν₀).
    maxiter : int
        Maximum iterations for Nelder-Mead.

    Returns
    -------
    dict
        Keys: 'rho', 'nu', 'log_likelihood', 'success'.
    """
    def neg_ll(params):
        rho, nu = params
        if abs(rho) >= 0.9999 or nu < 2.01:
            return 1e15
        return -ll_t_copula(rho, nu, u1, u2)

    result = minimize(
        neg_ll, x0=list(x0), method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8}
    )
    return {
        "rho": result.x[0],
        "nu": result.x[1],
        "log_likelihood": -result.fun,
        "success": result.success
    }
