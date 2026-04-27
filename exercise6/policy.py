"""
Stop-loss switch — Lo (2007) eqs. (24)–(25).

Given a return series ``R`` and a threshold ``zeta``, the binary indicator

    omega_t = 1{ R_{t-1} > zeta }                                     (24)

decides whether to stay invested in the strategy or move to the risk-free
asset for date ``t``:

    R_pt = omega_t R_t + (1 - omega_t) R_f.                           (25)

Convention: with ``len(R) = n_obs`` the indicator can only be evaluated
for ``t = 1, ..., n_obs - 1`` (it depends on ``R_{t-1}``). The function
returns the ``(n_obs - 1,)`` arrays aligned at date ``t``.
"""

from __future__ import annotations

import numpy as np


def apply_stop_loss(R, zeta, R_f):
    """
    Apply the stop-loss policy to an AR(1) return path.

    Parameters
    ----------
    R : (n_obs,) ndarray of strategy returns.
    zeta : float
        Stop-loss threshold (R_{t-1} <= zeta -> switch to risk free).
    R_f : float
        Risk-free rate (same period as ``R``).

    Returns
    -------
    R_p : (n_obs - 1,) ndarray
        Portfolio-plus-stop-loss returns aligned at date ``t``.
    omega : (n_obs - 1,) ndarray of {0, 1}
        Stop-loss indicator at date ``t``.
    R_t : (n_obs - 1,) ndarray
        Risky returns aligned at date ``t`` (i.e. ``R[1:]``).
    """
    R = np.asarray(R)
    omega = (R[:-1] > zeta).astype(float)
    R_t = R[1:]
    R_p = omega * R_t + (1.0 - omega) * R_f
    return R_p, omega, R_t
