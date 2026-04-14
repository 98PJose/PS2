"""
Portfolio construction and risk analysis.

Given dependent uniform draws (u1, u2) from a copula, transforms them into
asset returns via specified marginal distributions (inverse CDF method),
constructs a weighted portfolio, and computes descriptive statistics and VaR.
"""

import numpy as np
from scipy.stats import norm, t as tdist, skew, kurtosis


def get_marginal_ppf(spec):
    """
    Return the inverse CDF (percent point function) for a marginal specification.

    Parameters
    ----------
    spec : dict
        Must contain 'distribution' (str) and 'params' (dict).
        Supported distributions: 'norm', 't'.

    Returns
    -------
    callable
        Function u -> r mapping uniforms to returns.
    """
    dist_name = spec["distribution"]
    params = spec["params"]

    if dist_name == "norm":
        return lambda u: norm.ppf(u, loc=params.get("loc", 0),
                                  scale=params.get("scale", 1))
    elif dist_name == "t":
        return lambda u: tdist.ppf(u, df=params["df"])
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")


def compute_portfolio_returns(u1, u2, marginal1_spec, marginal2_spec, weights):
    """
    Transform copula draws to portfolio returns.

    Steps:
        1. r1 = F1^{-1}(u1)  — inversion method for marginal 1
        2. r2 = F2^{-1}(u2)  — inversion method for marginal 2
        3. r_p = w1·r1 + w2·r2

    Parameters
    ----------
    u1, u2 : ndarray
        Dependent U(0,1) draws from a copula, shape (Ns,).
    marginal1_spec, marginal2_spec : dict
        Marginal distribution specs (see get_marginal_ppf).
    weights : list or ndarray
        Portfolio weights [w1, w2], must sum to 1.

    Returns
    -------
    rp : ndarray
        Portfolio returns, shape (Ns,).
    r1 : ndarray
        Stock 1 returns.
    r2 : ndarray
        Stock 2 returns.
    """
    ppf1 = get_marginal_ppf(marginal1_spec)
    ppf2 = get_marginal_ppf(marginal2_spec)
    r1 = ppf1(u1)
    r2 = ppf2(u2)
    rp = weights[0] * r1 + weights[1] * r2
    return rp, r1, r2


def compute_stats(rp, var_levels):
    """
    Compute descriptive statistics and VaR for a return series.

    Parameters
    ----------
    rp : ndarray
        Portfolio returns, shape (Ns,).
    var_levels : list of float
        Quantile levels for VaR (e.g. [0.01, 0.05]).

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'skewness', 'kurtosis', 'var'
        where 'var' is a dict mapping level -> VaR value.
        Kurtosis is the regular (not excess) kurtosis: Normal = 3.
    """
    return {
        "mean": float(np.mean(rp)),
        "std": float(np.std(rp, ddof=1)),
        "skewness": float(skew(rp)),
        "kurtosis": float(kurtosis(rp, fisher=False)),
        "var": {q: float(np.quantile(rp, q)) for q in var_levels},
    }
