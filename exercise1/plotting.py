"""
Plotting functions for Exercise 1.

Generates:
- 1.a) Scatter plots of dependent U(0,1) draws for each copula.
- 1.b) Histograms of portfolio returns with Normal PDF overlay.
- 1.d) Scatter plot of mixture copula draws.
"""

import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_copula_scatter(draws_dict, output_dir, dpi=150):
    """
    Plot scatter plots of bivariate U(0,1) draws for multiple copulas.

    Parameters
    ----------
    draws_dict : dict
        Keys are copula labels (str), values are (u1, u2) tuples of ndarray.
    output_dir : str
        Directory to save the figure.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    n = len(draws_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

    if nrows == 1:
        axes = [axes]

    for idx, (label, (u1, u2)) in enumerate(draws_dict.items()):
        ax = axes[idx // ncols][idx % ncols]
        ax.scatter(u1[::5], u2[::5], s=0.3, alpha=0.2)
        ax.set(xlabel="$u_1$", ylabel="$u_2$",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_1a_scatter.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_portfolio_histograms(returns_dict, output_dir, dpi=150):
    """
    Plot histograms of portfolio returns with a Normal PDF overlay.

    The Normal PDF uses the same mean and std as the simulated portfolio,
    allowing visual comparison of tail behavior.

    Parameters
    ----------
    returns_dict : dict
        Keys are copula labels, values are portfolio return arrays.
    output_dir : str
        Directory to save the figure.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    n = len(returns_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

    if nrows == 1:
        axes = [axes]

    for idx, (label, rp) in enumerate(returns_dict.items()):
        ax = axes[idx // ncols][idx % ncols]
        mu, sd = np.mean(rp), np.std(rp)
        ax.hist(rp, bins=120, density=True, alpha=0.55,
                color="steelblue", edgecolor="none", label="Simulated")
        xg = np.linspace(mu - 4.5 * sd, mu + 4.5 * sd, 400)
        ax.plot(xg, norm.pdf(xg, mu, sd), "r-", lw=1.5, label="Normal PDF")
        ax.set(xlabel="$r_p$", ylabel="Density")
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_1b_histograms.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_mixture_scatter(u1, u2, title, output_dir, filename="fig_1d_mixture.png",
                         dpi=150):
    """
    Scatter plot of bivariate U(0,1) draws from a single copula model.

    Parameters
    ----------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    title : str
        Unused, kept for backwards compatibility.
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u1[::5], u2[::5], s=0.3, alpha=0.2, c="purple")
    ax.set(xlabel="$u_1$", ylabel="$u_2$",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
