"""
Log-likelihood functions for bivariate copula models.

These are used in the second step of the IFM (Inference Functions for Margins)
estimation method: given pseudo-observations (u1, u2) from the EDF, maximize
the copula log-likelihood to estimate the copula parameters.

References: Copula_slides.pdf, equations cited in each docstring.
"""

import numpy as np
from scipy.stats import norm, t as tdist
from scipy.special import gammaln


def ll_gaussian_copula(rho, u1, u2):
    """
    Log-likelihood of the bivariate Gaussian copula.

    Derivation (eq 31):
        The Gaussian copula PDF is:
            c^G(u1,u2) = (det Ψ)^{-1/2} exp(-½ η'(Ψ^{-1} - I₂) η)

        where η = (Φ^{-1}(u1), Φ^{-1}(u2))' and Ψ = [[1,ρ],[ρ,1]].

        Expanding:
            det Ψ = 1 - ρ²

            Ψ^{-1} = 1/(1-ρ²) [[1, -ρ], [-ρ, 1]]

            Ψ^{-1} - I₂ = 1/(1-ρ²) [[ρ², -ρ], [-ρ, ρ²]]

            η'(Ψ^{-1} - I₂)η = (ρ²(x₁² + x₂²) - 2ρx₁x₂) / (1-ρ²)

        Therefore:
            ln c^G = -½ ln(1-ρ²) - [ρ²(x₁² + x₂²) - 2ρx₁x₂] / [2(1-ρ²)]

        The total log-likelihood is LL = Σ_{t=1}^{T} ln c^G(u_{1t}, u_{2t}).

    Parameters
    ----------
    rho : float
        Correlation coefficient, -1 < ρ < 1.
    u1, u2 : ndarray
        Pseudo-observations in (0,1), shape (T,).

    Returns
    -------
    float
        Total log-likelihood.
    """
    x1 = norm.ppf(u1)
    x2 = norm.ppf(u2)
    log_density = (
        -0.5 * np.log(1 - rho**2)
        - (rho**2 * (x1**2 + x2**2) - 2 * rho * x1 * x2)
        / (2 * (1 - rho**2))
    )
    return np.sum(log_density)


def ll_t_copula(rho, nu, u1, u2):
    """
    Log-likelihood of the bivariate Student t copula.

    Derivation (eq 38):
        The t copula PDF is:
            c^t(u1,u2) = f_{ν,ρ}(t_ν^{-1}(u1), t_ν^{-1}(u2))
                         / [f_ν(t_ν^{-1}(u1)) · f_ν(t_ν^{-1}(u2))]

        where f_{ν,ρ} is the bivariate Student t PDF (eq 37):
            f_{ν,ρ}(x,y) = Γ((ν+2)/2) / [νπ Γ(ν/2) √(1-ρ²)]
                           · (1 + (x² - 2ρxy + y²)/[ν(1-ρ²)])^{-(ν+2)/2}

        and f_ν is the univariate Student t PDF (eq 36):
            f_ν(x) = Γ((ν+1)/2) / [√(νπ) Γ(ν/2)] · (1 + x²/ν)^{-(ν+1)/2}

        Taking logs and simplifying:
            ln c^t = ln Γ((ν+2)/2) + ln Γ(ν/2) - 2 ln Γ((ν+1)/2)
                     - ½ ln(1-ρ²)
                     - (ν+2)/2 · ln(1 + Q/[ν(1-ρ²)])
                     + (ν+1)/2 · [ln(1 + y₁²/ν) + ln(1 + y₂²/ν)]

        where Q = y₁² - 2ρy₁y₂ + y₂² and yᵢ = t_ν^{-1}(uᵢ).

        Step-by-step constant cancellation:
            ln f_{ν,ρ} contributes: ln Γ((ν+2)/2) - ln ν - ln π - ln Γ(ν/2)
            2·ln f_ν contributes:   2 ln Γ((ν+1)/2) - ln ν - ln π - 2 ln Γ(ν/2)

            Subtracting:
            ln Γ((ν+2)/2) - ln Γ(ν/2) - 2 ln Γ((ν+1)/2) + 2 ln Γ(ν/2)
            = ln Γ((ν+2)/2) + ln Γ(ν/2) - 2 ln Γ((ν+1)/2)

    Parameters
    ----------
    rho : float
        Correlation coefficient, -1 < ρ < 1.
    nu : float
        Degrees of freedom, ν > 2.
    u1, u2 : ndarray
        Pseudo-observations in (0,1), shape (T,).

    Returns
    -------
    float
        Total log-likelihood.
    """
    y1 = tdist.ppf(u1, nu)
    y2 = tdist.ppf(u2, nu)
    Q = y1**2 - 2 * rho * y1 * y2 + y2**2

    log_density = (
        gammaln((nu + 2) / 2) + gammaln(nu / 2)
        - 2 * gammaln((nu + 1) / 2)
        - 0.5 * np.log(1 - rho**2)
        - (nu + 2) / 2 * np.log(1 + Q / (nu * (1 - rho**2)))
        + (nu + 1) / 2 * (np.log(1 + y1**2 / nu) + np.log(1 + y2**2 / nu))
    )
    return np.sum(log_density)
