"""
Copula simulation functions using the conditional (inverse CDF) method.

Each function takes two arrays of independent U(0,1) draws (v1, v2) and copula
parameters, and returns a tuple (u1, u2) of dependent U(0,1) draws.

The conditional method sets u2 = v2 and solves v1 = C(u1 | u2) for u1,
where C(u1 | u2) = dC(u1, u2)/du2 is the conditional copula CDF.

References: Copula_slides.pdf, equations cited in each docstring.
"""

import numpy as np
from scipy.stats import norm, t as tdist


def sim_clayton(v1, v2, theta):
    """
    Clayton copula conditional simulation.

    Derivation (eq 43 -> eq 45):
        The Clayton copula CDF is  C(u1,u2) = (u1^{-θ} + u2^{-θ} - 1)^{-1/θ}.
        The conditional CDF is:
            C(u1|u2) = ∂C/∂u2 = u2^{-(1+θ)} (u1^{-θ} + u2^{-θ} - 1)^{-1/θ - 1}
        Setting v1 = C(u1|u2) and solving for u1:
            u1 = [1 + u2^{-θ} (v1^{-θ/(1+θ)} - 1)]^{-1/θ}
        with u2 = v2.

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws, shape (Ns,).
    theta : float
        Clayton parameter, θ > 0.

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    """
    u2 = v2
    u1 = (1 + u2**(-theta) * (v1**(-theta / (1 + theta)) - 1))**(-1 / theta)
    return u1, u2


def sim_surv_clayton(v1, v2, theta):
    """
    Survival Clayton copula conditional simulation.

    Derivation (eq 48 -> eq 49 -> eq 51):
        The survival Clayton CDF is:
            C^S(u1,u2) = u1 + u2 - 1 + C^Cla(1-u1, 1-u2)
        The conditional CDF is (eq 49):
            C^S(u1|u2) = 1 - (1-u2)^{-(1+θ)} [(1-u1)^{-θ} + (1-u2)^{-θ} - 1]^{-1/θ - 1}
        Setting v1 = C^S(u1|u2), u2 = v2, and solving for u1 gives (eq 51):
            u1 = 1 - [(v̄1 · v̄2^{1+θ})^{-θ/(1+θ)} - v̄2^{-θ} + 1]^{-1/θ}
        where v̄i = 1 - vi.

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws.
    theta : float
        Clayton parameter, θ > 0.

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    """
    vb1 = 1 - v1
    vb2 = 1 - v2
    inner = (vb1 * vb2**(1 + theta))**(-theta / (1 + theta)) - vb2**(-theta) + 1
    u1 = 1 - inner**(-1 / theta)
    u2 = v2
    return u1, u2


def sim_gaussian(v1, v2, rho):
    """
    Gaussian copula conditional simulation.

    Derivation (eq 30 -> eq 34):
        The Gaussian copula CDF is C^G(u1,u2) = Φ_ρ(Φ^{-1}(u1), Φ^{-1}(u2)).
        The conditional CDF is (eq 30):
            C^G(u1|u2) = Φ( (Φ^{-1}(u1) - ρ Φ^{-1}(u2)) / √(1-ρ²) )
        Setting v1 = C^G(u1|u2) and solving for u1:
            Φ^{-1}(u1) = ρ Φ^{-1}(u2) + √(1-ρ²) Φ^{-1}(v1)
            u1 = Φ( ρ Φ^{-1}(v2) + √(1-ρ²) Φ^{-1}(v1) )
        with u2 = v2.

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws.
    rho : float
        Correlation coefficient, -1 < ρ < 1.

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    """
    u1 = norm.cdf(rho * norm.ppf(v2) + np.sqrt(1 - rho**2) * norm.ppf(v1))
    u2 = v2
    return u1, u2


def sim_student_t(v1, v2, rho, nu):
    """
    Student t copula conditional simulation.

    Derivation (eq 39 -> eq 41):
        The conditional CDF of the t copula is (eq 39):
            C^t(u1|u2) = t_{ν+1}( (t_ν^{-1}(u1) - ρ t_ν^{-1}(u2)) / √((1-ρ²)(ν+[t_ν^{-1}(u2)]²)/(ν+1)) )

        Note the inner term uses t_{ν+1} (not t_ν) because conditioning in the
        bivariate t reduces degrees of freedom by the conditioning structure.

        Setting v1 = C^t(u1|u2), u2 = v2, let y2 = t_ν^{-1}(v2). Then:
            t_{ν+1}^{-1}(v1) = (y1 - ρ y2) / √((1-ρ²)(ν + y2²)/(ν+1))

        Solving for y1:
            y1 = ρ y2 + t_{ν+1}^{-1}(v1) · √( (1-ρ²)(ν + y2²)/(ν+1) )
            u1 = t_ν(y1)

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws.
    rho : float
        Correlation coefficient, -1 < ρ < 1.
    nu : float
        Degrees of freedom, ν > 0.

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    """
    y2 = tdist.ppf(v2, nu)
    y1 = rho * y2 + tdist.ppf(v1, nu + 1) * np.sqrt(
        (1 - rho**2) * (nu + y2**2) / (nu + 1))
    u1 = tdist.cdf(y1, nu)
    u2 = v2
    return u1, u2


def sim_fgm(v1, v2, lam):
    """
    FGM (Farlie-Gumbel-Morgenstern) copula conditional simulation.

    Derivation (eq 52 -> eq 53):
        The FGM copula CDF is:
            C(u1,u2) = u1 u2 [1 + λ(1-u1)(1-u2)],  λ ∈ [-1, 1].
        The conditional CDF is (eq 52):
            C(u1|u2) = u1 [1 + λ(1-u1)(1-2u2)]
        Let ψ(u2) = λ(1-2u2). Then C(u1|u2) = u1(1 + ψ) - u1²ψ.
        Setting v1 = C(u1|u2), this is a quadratic in u1:
            ψ u1² - (1+ψ) u1 + v1 = 0
        Solving (taking the root in [0,1]):
            u1 = [(1+ψ) - √((1+ψ)² - 4v1ψ)] / (2ψ)

        When ψ → 0 (λ = 0 or u2 = 0.5), L'Hôpital gives u1 → v1.

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws.
    lam : float
        FGM parameter, λ ∈ [-1, 1].

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws.
    """
    psi = lam * (1 - 2 * v2)
    # Replace near-zero psi with 1.0 before division to avoid RuntimeWarning;
    # np.where will discard these results in favour of v1.
    safe_psi = np.where(np.abs(psi) < 1e-10, 1.0, psi)
    u1 = np.where(
        np.abs(psi) < 1e-10,
        v1,
        ((1 + safe_psi) - np.sqrt((1 + safe_psi)**2 - 4 * v1 * safe_psi))
        / (2 * safe_psi))
    u2 = v2
    return u1, u2


def sim_mixture(v1, v2, indicator, sim_func_a, params_a, sim_func_b, params_b):
    """
    Mixture of two copulas.

    For each draw k, if indicator[k] is True, draw from copula A; else copula B.
    Both components are evaluated on the same (v1, v2), and the result is selected
    per-draw by the indicator. This is valid because each component produces a
    valid conditional draw from (v1[k], v2[k]).

    Parameters
    ----------
    v1, v2 : ndarray
        Independent U(0,1) draws.
    indicator : ndarray of bool
        True selects copula A, False selects copula B.
    sim_func_a, sim_func_b : callable
        Copula simulation functions.
    params_a, params_b : dict
        Parameters passed to each simulation function.

    Returns
    -------
    u1, u2 : ndarray
        Dependent U(0,1) draws from the mixture.
    """
    u1_a, u2_a = sim_func_a(v1, v2, **params_a)
    u1_b, u2_b = sim_func_b(v1, v2, **params_b)
    u1 = np.where(indicator, u1_a, u1_b)
    u2 = np.where(indicator, u2_a, u2_b)
    return u1, u2
