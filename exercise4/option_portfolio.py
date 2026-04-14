"""
Covered-call (CC) and protective-put (PP) portfolio returns.

The "Option-based portfolios" slides (Section 2) consider four strategies
over a single holding period equal to the option maturity T:

    R_CC(K, T)  = [S_2T - max(S_1T - K, 0)] / [S_20 - C_0(K, T)] - 1,   (eq 4)
    R_PP(K, T)  = [S_2T + max(K - S_1T, 0)] / [S_20 + P_0(K, T)] - 1.   (eq 5)

When the held asset and the option underlying are the same (S_1T == S_2T),
these reduce to the classic single-asset CC and PP. When they differ (the
"CC2"/"PP2" variants of ccK_simuRet2.m), the dependence between S_1T and S_2T
enters explicitly, which is why Exercise 4 studies these statistics under
alternative copula models.

All arithmetic is vectorised over Monte-Carlo paths, and the wrappers below
also vectorise over the strike grid by returning (N, NK) arrays.
"""

from __future__ import annotations

import numpy as np

from .black_scholes import bs_call_put


def cc_return(S1T, S2T, K, S20, C0):
    """Covered-call return given call-on-asset-1 paid on sale of asset 2."""
    call_payoff = np.maximum(S1T - K, 0.0)
    return (S2T - call_payoff) / (S20 - C0) - 1.0


def pp_return(S1T, S2T, K, S20, P0):
    """Protective-put return: long asset 2, long put on asset 1."""
    put_payoff = np.maximum(K - S1T, 0.0)
    return (S2T + put_payoff) / (S20 + P0) - 1.0


def simulate_terminal_prices(shocks, mu, sigma, T, S0):
    """
    Geometric Brownian Motion terminal prices from (Z_1, Z_2) shocks.

    Parameters
    ----------
    shocks : (N, 2) ndarray
        Columns are Z_1, Z_2.
    mu, sigma : float or (2,) array_like
        Drift and volatility per asset (annualised).
    T : float
    S0 : float or (2,) array_like
        Initial price.

    Returns
    -------
    (N, 2) ndarray
        Terminal prices (S_1T, S_2T).
    """
    mu = np.broadcast_to(np.asarray(mu, float), (2,))
    sigma = np.broadcast_to(np.asarray(sigma, float), (2,))
    S0 = np.broadcast_to(np.asarray(S0, float), (2,))
    drift = (mu - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    R = drift + vol * shocks
    return S0 * np.exp(R)


def portfolio_returns_over_strikes(S1T, S2T, K_grid, S0, r, sigma, T):
    """
    Compute cc, cc2, pp, pp2 return matrices over a strike grid.

    For 4.1–4.4 the slide parameters are symmetric (S_10 = S_20, sigma_1 =
    sigma_2), so the BS call and put prices are shared between the "same
    asset" and "different asset" variants. The simulation distinguishes them
    through the choice of underlying: cc/pp use S_1T as both held and option
    leg; cc2/pp2 use S_2T as the held leg and S_1T as the option underlying.

    Parameters
    ----------
    S1T, S2T : (N,) ndarray
        Terminal prices of the two assets.
    K_grid : (NK,) ndarray
    S0, r, sigma, T : float

    Returns
    -------
    dict with keys {"cc", "cc2", "pp", "pp2"} mapping to (N, NK) arrays
    and key "prices" mapping to (C0, P0) over the strike grid.
    """
    C0, P0 = bs_call_put(S0, r, sigma, T, K_grid)

    # Broadcast: (N, 1) underlyings vs (NK,) grid  ->  (N, NK).
    S1T_col = S1T[:, None]
    S2T_col = S2T[:, None]
    K_row = K_grid[None, :]
    C0_row = C0[None, :]
    P0_row = P0[None, :]

    cc = cc_return(S1T_col, S1T_col, K_row, S0, C0_row)
    cc2 = cc_return(S1T_col, S2T_col, K_row, S0, C0_row)
    pp = pp_return(S1T_col, S1T_col, K_row, S0, P0_row)
    pp2 = pp_return(S1T_col, S2T_col, K_row, S0, P0_row)

    return {"cc": cc, "cc2": cc2, "pp": pp, "pp2": pp2, "C0": C0, "P0": P0}
