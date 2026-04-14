"""
Kupiec (1995) unconditional coverage (UC) test.

Let N be the number of VaR breaches observed out of T forecasts, and p the
theoretical left-tail level (e.g. p = 0.05 for a 95% VaR). Under H_0 the
breaches are i.i.d. Bernoulli(p), so the likelihood ratio statistic is

    LR_UC = -2 ln[ (1-p)^{T-N} p^N ]
            + 2 ln[ (1 - N/T)^{T-N} (N/T)^N ]                     (eq 11)

            = 2 { N ln(N/(Tp)) + (T-N) ln( (T-N) / (T(1-p)) ) }    (equivalent).

LR_UC is asymptotically chi-squared with one degree of freedom; H_0 is
rejected when LR_UC exceeds the chi-square critical value at the chosen
significance level.

In our setting T is the number of Monte-Carlo replications (one forecast
per replication, compared against the 1001st simulated observation).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def kupiec_uc(n_breaches, n_forecasts, p):
    """
    Compute the Kupiec unconditional-coverage LR statistic and p-value.

    Parameters
    ----------
    n_breaches : int
        Observed number of VaR breaches (N).
    n_forecasts : int
        Total number of forecasts (T).
    p : float
        Theoretical left-tail level (e.g. 0.05 or 0.01).

    Returns
    -------
    dict with keys:
        ``breaches``    — N
        ``forecasts``   — T
        ``hit_rate``    — N / T
        ``expected``    — p
        ``LR_UC``       — test statistic
        ``p_value``     — 1 - F_{chi2_1}(LR_UC)
        ``reject_95``   — True if p_value < 0.05
        ``reject_99``   — True if p_value < 0.01
    """
    N = int(n_breaches)
    T = int(n_forecasts)
    if T <= 0:
        raise ValueError("n_forecasts must be positive")
    hit = N / T

    # Log-likelihood under H_0 (restricted: rate fixed at p)
    ll0 = (T - N) * np.log1p(-p) + N * np.log(p) if 0 < p < 1 else 0.0

    # Log-likelihood unrestricted (MLE at N/T); handle the boundary cases
    # N = 0 and N = T where ln 0 appears. Those contributions are 0 in the
    # limit (0 * ln 0 := 0), so we compute term-by-term and skip zeros.
    if N == 0:
        ll1 = (T - N) * np.log(1 - 0 + 1e-300)  # essentially 0
    elif N == T:
        ll1 = N * np.log(1 - 1e-16)
    else:
        ll1 = (T - N) * np.log(1 - hit) + N * np.log(hit)

    LR = 2.0 * (ll1 - ll0)
    p_value = 1.0 - chi2.cdf(LR, df=1)

    return {
        "breaches": N,
        "forecasts": T,
        "hit_rate": hit,
        "expected": p,
        "LR_UC": float(LR),
        "p_value": float(p_value),
        "reject_95": bool(p_value < 0.05),
        "reject_99": bool(p_value < 0.01),
    }
