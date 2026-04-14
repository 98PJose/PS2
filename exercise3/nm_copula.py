"""
Bivariate Normal Mixture (NM) copula — conditional CDF and q-quantile curves.

Mathematical definition
-----------------------
The NM copula is a finite mixture of two Gaussian copulas sharing the same
uniform margins but with different correlations rho_1 and rho_2 and mixing
weight pi in (0, 1):

    C^NM(u_1, u_2) = pi * C^G(u_1, u_2; rho_1) + (1 - pi) * C^G(u_1, u_2; rho_2).

Because conditioning (partial differentiation) is linear, the conditional CDF
of U_1 given U_2 = u_2 is also a mixture (problem statement, Exercise 3):

    C^NM(u_1 | u_2) =
          pi     * Phi( (Phi^{-1}(u_1) - rho_1 * Phi^{-1}(u_2)) / sqrt(1 - rho_1^2) )
      + (1-pi)   * Phi( (Phi^{-1}(u_1) - rho_2 * Phi^{-1}(u_2)) / sqrt(1 - rho_2^2) ).

q-quantile curves
-----------------
Following the convention of the course slides (Section 9 "Quantile curves of
copulas"), a q-quantile curve of a bivariate copula C is the locus of points
(u_1, u_2) in the unit square that satisfy

    C(u_1 | u_2) = q,

where u_2 is treated as the conditioning variable (x-axis) and u_1 as the
solved-for variable (y-axis). For each Gaussian-family copula discussed in the
slides, inverting the conditional CDF gives a closed-form expression for
u_1(u_2 ; q). For the NM copula, the right-hand side is a weighted sum of two
Phi terms and the inverse cannot be obtained in closed form; a numerical
root-finder is used instead.

A note on the problem statement: the Exercise 3 text phrases the inversion as
"back out a value of u_2 from C^NM(u_1 | u_2) = q from each u_1 and q".
Taken literally (fix u_1, solve for u_2) with pi = 0.3, rho_1 = -0.7,
rho_2 = 0.4 the map u_2 |-> C^NM(u_1 | u_2) is NOT monotone onto (0, 1); in
fact its range is (0.3, 0.7), so q in {0.05, 0.25, 0.75, 0.95} admits no
solution. Inverting in u_1 instead is strictly monotone on (0, 1) (the
coefficient 1/sqrt(1-rho_i^2) of Phi^{-1}(u_1) is strictly positive in both
components), gives a unique solution for every q in (0, 1), and matches the
convention adopted in the slides for all other copula families. This is the
convention implemented below and is also the one under which the plots in
Section 9 of the slides were produced.

When both marginals are N(0,1) (the Exercise 3 setting), the curves on the
return scale are r_i = Phi^{-1}(u_i), so the same root-finder delivers both
the unit-square curve and the corresponding curve in the (r_1, r_2) plane.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def nm_conditional_cdf(u1, u2, pi, rho1, rho2):
    """
    Evaluate the Normal Mixture conditional CDF C^NM(u_1 | u_2).

    Parameters
    ----------
    u1, u2 : array_like
        Points in (0, 1). Broadcasting is supported.
    pi : float
        Mixing weight of the first component, 0 < pi < 1.
    rho1, rho2 : float
        Component correlations, each in (-1, 1).

    Returns
    -------
    ndarray
        C^NM(u_1 | u_2) evaluated point-wise.
    """
    z1 = norm.ppf(u1)
    z2 = norm.ppf(u2)
    s1 = np.sqrt(1.0 - rho1**2)
    s2 = np.sqrt(1.0 - rho2**2)
    return pi * norm.cdf((z1 - rho1 * z2) / s1) + (1 - pi) * norm.cdf((z1 - rho2 * z2) / s2)


def nm_quantile_u1(u2, q, pi, rho1, rho2, eps=1e-10):
    """
    Invert C^NM(u_1 | u_2) = q for u_1 at a single (u_2, q).

    Uses Brent's method on (eps, 1 - eps). The conditional CDF is strictly
    increasing in u_1 (each Phi term is increasing in Phi^{-1}(u_1) and the
    weights are positive), so a unique root exists for any q in (0, 1).

    Parameters
    ----------
    u2 : float
        Conditioning value in (0, 1).
    q : float
        Target probability in (0, 1).
    pi, rho1, rho2 : float
        NM copula parameters.
    eps : float
        Bracket padding away from 0 and 1, to keep Phi^{-1} finite.

    Returns
    -------
    float
        u_1 such that C^NM(u_1 | u_2) = q.
    """
    def f(u1):
        return nm_conditional_cdf(u1, u2, pi, rho1, rho2) - q

    return brentq(f, eps, 1.0 - eps)


def nm_quantile_curve(u2_grid, q, pi, rho1, rho2, eps=1e-10):
    """
    Compute the q-quantile curve u_1(u_2 ; q) on a grid of u_2 values.

    Parameters
    ----------
    u2_grid : ndarray
        Grid of conditioning values in (0, 1).
    q : float
        Target probability in (0, 1).
    pi, rho1, rho2 : float
        NM copula parameters.
    eps : float
        Bracket padding for Brent's method.

    Returns
    -------
    ndarray
        Array of u_1 values, same shape as ``u2_grid``.
    """
    u1 = np.empty_like(u2_grid, dtype=float)
    for i, u2 in enumerate(u2_grid):
        u1[i] = nm_quantile_u1(float(u2), q, pi, rho1, rho2, eps=eps)
    return u1
