"""
Descriptive statistics for Exercise 2.1.

Computes mean, standard deviation, skewness, kurtosis (regular, Normal=3),
and the sample Pearson correlation matrix.
"""

import numpy as np
from scipy.stats import skew, kurtosis


def compute_descriptive_stats(returns):
    """
    Compute descriptive statistics for each return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series, shape (T, n).

    Returns
    -------
    stats : dict
        Keys are asset names; values are dicts with
        'mean', 'std', 'skewness', 'kurtosis'.
    corr_matrix : np.ndarray
        Sample Pearson correlation matrix, shape (n, n).
    """
    stats = {}
    for col in returns.columns:
        r = returns[col].values
        stats[col] = {
            "mean": float(np.mean(r)),
            "std": float(np.std(r, ddof=1)),
            "skewness": float(skew(r)),
            "kurtosis": float(kurtosis(r, fisher=False)),
        }

    corr_matrix = np.corrcoef(returns.values.T)

    return stats, corr_matrix
