"""
Black–Scholes closed-form prices for European calls and puts.

Port of the course MATLAB routine BS.m:

    d1 = (ln(S0/K) + (r + sigma^2/2) T) / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    C  = S0 * Phi(d1)           - K exp(-r T) * Phi(d2)
    P  = K exp(-r T) * Phi(-d2) - S0         * Phi(-d1)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call_put(S0, r, sigma, T, K):
    """
    European call and put prices under Black–Scholes.

    Parameters
    ----------
    S0, r, sigma, T : float
        Spot price, risk-free rate, volatility, maturity (all scalar).
    K : float or array_like
        Strike price(s).

    Returns
    -------
    C, P : same shape as K
        European call and put prices.
    """
    K = np.asarray(K, dtype=float)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    C = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    P = K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return C, P
