"""Exercise 4 — plotting of mean, volatility, skewness, kurtosis over K."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt


# Colour / linestyle convention matches the slides:
#   cc   : blue dashed
#   cc2  : blue solid
#   pp   : red dashed
#   pp2  : red solid
_STYLE = {
    "cc": ("--", "tab:blue", "cc"),
    "cc2": ("-", "tab:blue", "cc2"),
    "pp": ("--", "tab:red", "pp"),
    "pp2": ("-", "tab:red", "pp2"),
}


def plot_strategy_statistics(K, stats_by_strategy, title_suffix, filename, figures_dir, dpi=150):
    """
    Four-panel figure (mean, volatility, skewness, kurtosis) vs K.

    Parameters
    ----------
    K : (NK,) ndarray
    stats_by_strategy : dict
        {strategy_name -> {"mean": .., "std": .., "skew": .., "kurt": ..}}
    title_suffix : str
        Appended to each panel title (e.g. "Gaussian copula, rho = 0.9").
    filename : str
        Output filename (saved in ``figures_dir``).
    """
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("mean", "Mean (yearly)", "mean of portfolio"),
        ("std",  "Volatility (yearly)", "volatility of portfolio"),
        ("skew", "Skewness (yearly)", "skewness of portfolio"),
        ("kurt", "Kurtosis (yearly)", "kurtosis of portfolio"),
    ]

    for ax, (key, subtitle, ylabel) in zip(axes.ravel(), metrics):
        for strat, stats in stats_by_strategy.items():
            ls, color, label = _STYLE[strat]
            ax.plot(K, stats[key], linestyle=ls, color=color, label=label, lw=1.6)
        ax.set_xlim(K[0], K[-1])
        ax.set_xlabel("Exercise price, K")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Option strategy: {subtitle}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Option-based portfolios — {title_suffix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(figures_dir, filename)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[exercise4] saved {out}")
    return out
