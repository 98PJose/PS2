"""
Closed-form expected return under the stop-loss policy — Lo (2007), eq. (26).

The decomposition Lo states in (22a) carries over to the stop-loss case:

    E[R_pt] = Cov[omega_t, R_t] + E[omega_t] E[R_t] + (1 - E[omega_t]) R_f.

Under the AR(1) + Gaussian innovations of (23) and the indicator (24),
``R_{t-1}`` is N(mu, sigma**2) and the conditional mean is

    E[R_t | R_{t-1}] = mu + rho (R_{t-1} - mu).

Plugging in and using the truncated-Normal identity

    E[(R_{t-1} - mu) * 1{R_{t-1} > zeta}] = sigma * phi((zeta - mu) / sigma),

Lo arrives at the closed form (26):

    E[R_pt] = rho sigma phi((zeta - mu) / sigma)            <- ACTIVE
            + mu (1 - Phi((zeta - mu) / sigma))             ----+
            + R_f Phi((zeta - mu) / sigma).                 ----+ PASSIVE

This module evaluates that expression and the matching active / passive
split for a (mu, rho, sigma, zeta, R_f) tuple, and helpers to compute the
"%active" and 1 - E[omega] columns reported in Lo's Table 1.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def expected_return_closed_form(mu, rho, sigma, zeta, R_f):
    """
    Evaluate eq. (26) and the AP decomposition.

    Returns
    -------
    dict with keys
        E_Rp        : E[R_pt]
        active      : rho sigma phi((zeta - mu) / sigma)
        passive     : mu (1 - Phi((zeta - mu) / sigma)) + R_f Phi(.)
        E_omega     : 1 - Phi((zeta - mu) / sigma) = P(R_{t-1} > zeta)
        pct_active  : active / E_Rp  (Lo's "%Active" column)
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    z = (zeta - mu) / sigma
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)

    active = rho * sigma * phi_z
    passive = mu * (1.0 - Phi_z) + R_f * Phi_z
    E_Rp = active + passive
    E_omega = 1.0 - Phi_z

    if E_Rp != 0.0 and not math.isclose(E_Rp, 0.0, abs_tol=1e-15):
        pct_active = active / E_Rp
    else:
        pct_active = float("nan")

    return {
        "E_Rp": E_Rp,
        "active": active,
        "passive": passive,
        "E_omega": E_omega,
        "pct_active": pct_active,
    }
