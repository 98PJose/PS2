"""
Data loading for Exercise 2.

Reads the crypto.xlsx spreadsheet and extracts the three daily return series
(BTC, ETH, ADA). The spreadsheet has a two-row header; returns start at
row index 2 (0-based) and columns 5, 6, 7. The first return observation
is NaN (no return on the first date) and is dropped.
"""

import pandas as pd
import numpy as np


def load_returns(data_path, asset_names):
    """
    Load daily return series from the crypto spreadsheet.

    Parameters
    ----------
    data_path : str
        Path to the Excel file (crypto.xlsx).
    asset_names : list of str
        Asset labels, e.g. ["BTC", "ETH", "ADA"].

    Returns
    -------
    returns : pd.DataFrame
        DataFrame of shape (T, n) with columns = asset_names, index = dates.
        NaN rows (first observation) are dropped.
    """
    raw = pd.read_excel(data_path, header=None)

    # Extract date column (column 1) and return columns (columns 5, 6, 7)
    # Data rows start at index 2 (rows 0-1 are headers)
    dates = pd.to_datetime(raw.iloc[2:, 1])
    ret = raw.iloc[2:, 5:8].copy()
    ret.columns = asset_names
    ret.index = dates.values
    ret = ret.astype(float)

    # Drop first row (NaN — no return on first date)
    ret = ret.dropna()

    return ret
