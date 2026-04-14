"""
Descriptive statistics used by Exercise 4.

MATLAB's built-in skewness() and kurtosis() return the BIASED (moment-based,
uncorrected) estimators, and kurtosis() returns the CLASSICAL (non-excess)
coefficient with Normal reference 3. To reproduce the slide graphs we follow
the same convention here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew


def strike_stats(R):
    """
    Column-wise mean, std, skewness, kurtosis over an (N, NK) matrix.

    Uses MATLAB defaults:
      * std with divisor N (ddof=0).
      * skew biased (bias=True).
      * kurtosis non-excess (fisher=False), biased.

    Returns
    -------
    dict with keys "mean", "std", "skew", "kurt", each a (NK,) ndarray.
    """
    return {
        "mean": R.mean(axis=0),
        "std": R.std(axis=0, ddof=0),
        "skew": _skew(R, axis=0, bias=True),
        "kurt": _kurtosis(R, axis=0, fisher=False, bias=True),
    }
