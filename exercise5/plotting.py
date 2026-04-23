"""Exercise 5 — summary plots of empirical failure rates across scenarios."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

_MODEL_COLORS = {"gaussian": "tab:blue", "t": "tab:green", "clayton": "tab:red"}
_MODEL_LABELS = {"gaussian": "Gaussian", "t": "Student t", "clayton": "Clayton"}


def plot_failure_rates(results, alphas, figures_dir, dpi=150):
    """
    Grouped bar chart of empirical failure rates across the three Clayton
    DGP scenarios, overlaid with the nominal alpha line.

    Parameters
    ----------
    results : list[dict]
        Output of ``run_scenario`` for each theta_true, in DGP-parameter order.
    alphas : iterable of float
        Left-tail levels to plot (e.g. (0.01, 0.05)).
    """
    os.makedirs(figures_dir, exist_ok=True)

    thetas = [f"theta = {r['theta_true']}" for r in results]
    x = np.arange(len(results))
    width = 0.25

    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 5),
                             squeeze=False)
    for ax, a in zip(axes[0], alphas):
        for i, model in enumerate(("gaussian", "t", "clayton")):
            rates = [r["kupiec"][(model, a)]["hit_rate"] for r in results]
            ax.bar(x + (i - 1) * width, rates, width=width,
                   color=_MODEL_COLORS[model], label=_MODEL_LABELS[model])
            # Annotate rejections.
            for xi, r in zip(x + (i - 1) * width, results):
                kp = r["kupiec"][(model, a)]
                mark = ""
                if kp["reject_99"]:
                    mark = "**"
                elif kp["reject_95"]:
                    mark = "*"
                if mark:
                    ax.text(xi, kp["hit_rate"], mark,
                            ha="center", va="bottom", fontsize=11)
        ax.axhline(a, color="k", lw=1, linestyle="--",
                   label=f"nominal = {a:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels(thetas)
        ax.set_ylabel("empirical failure rate")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_5_failure_rates.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[exercise5] saved {out}")
    return out
