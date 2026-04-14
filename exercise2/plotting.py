"""
Plotting functions for Exercise 2.

Generates:
- 2.2a) Dynamic correlation plots for a single DCC model.
- 2.3)  Comparison of dynamic correlations across models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dynamic_correlations(rho_series, pair_labels, asset_names, dates,
                              title, output_dir, filename, dpi=150):
    """
    Plot time series of pairwise dynamic correlations from a DCC model.

    Parameters
    ----------
    rho_series : ndarray
        Dynamic correlations, shape (T, n_pairs).
    pair_labels : list of tuple
        Index pairs, e.g. [(0,1), (0,2), (1,2)].
    asset_names : list of str
        Asset labels, e.g. ["BTC", "ETH", "ADA"].
    dates : array-like
        Date index for x-axis.
    title : str
        Figure title.
    output_dir : str
        Directory to save figure.
    filename : str
        Output filename.
    dpi : int
        Figure resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    n_pairs = len(pair_labels)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.5 * n_pairs), sharex=True)
    fig.suptitle(title, fontsize=13)

    if n_pairs == 1:
        axes = [axes]

    for k, (i, j) in enumerate(pair_labels):
        ax = axes[k]
        label = f"{asset_names[i]}-{asset_names[j]}"
        ax.plot(dates, rho_series[:, k], linewidth=0.6, label=label)
        ax.set_ylabel(f"$\\rho_{{t}}$ ({label})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_correlation_comparison(rho_dict, pair_labels, asset_names, dates,
                                output_dir, filename="fig_2_3_comparison.png",
                                dpi=150):
    """
    Compare dynamic correlations across multiple DCC models.

    Parameters
    ----------
    rho_dict : dict
        Keys are model labels (str), values are rho_series arrays (T, n_pairs).
    pair_labels : list of tuple
        Index pairs.
    asset_names : list of str
        Asset labels.
    dates : array-like
        Date index.
    output_dir : str
        Directory to save figure.
    filename : str
        Output filename.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    n_pairs = len(pair_labels)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.5 * n_pairs), sharex=True)
    fig.suptitle("2.3) Comparison of dynamic correlations across marginal models",
                 fontsize=13)

    if n_pairs == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for k, (i, j) in enumerate(pair_labels):
        ax = axes[k]
        pair_name = f"{asset_names[i]}-{asset_names[j]}"
        for idx, (model_name, rho_s) in enumerate(rho_dict.items()):
            ax.plot(dates, rho_s[:, k], linewidth=0.6,
                    color=colors[idx % len(colors)], label=model_name, alpha=0.8)
        ax.set_ylabel(f"$\\rho_{{t}}$ ({pair_name})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
