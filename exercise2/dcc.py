"""
DCC (Dynamic Conditional Correlation) model estimation.

Implements the DCC model under both Gaussian and Student t copulas,
following the IFM approach (slides eqs 68-76).

The DCC dynamics (eq 69):
    Q_t = (1 - a - b) * Omega_hat + a * z_{t-1} z_{t-1}' + b * Q_{t-1}

where:
    - z_t is the (n x 1) vector of standardized residuals from Step I
    - Omega_hat = sample correlation of z_t (unconditional correlation)
    - Q_1 = Omega_hat (initialization)

The time-varying correlation matrix (eq 68):
    Psi_t = Lambda_t^{-1/2} Q_t Lambda_t^{-1/2}

where Lambda_t = diag(q_{11,t}, ..., q_{nn,t}).

Gaussian copula DCC log-likelihood (eq 72):
    LL_G = -1/2 sum_t [ ln|Psi_t| + eta_t' (Psi_t^{-1} - I_n) eta_t ]

where eta_t = Phi^{-1}(u_t) for EDF marginals, or eta_t = z_t for
GARCH/GJR marginals under Normal innovations.

Student t copula DCC log-likelihood (eq 76):
    LL_t = T/2 ln A(v) - 1/2 sum_t ln|Psi_t|
           - (v+n)/2 sum_t ln[1 + v^{-1} eta_t' Psi_t^{-1} eta_t]
           + (v+1)/2 sum_t sum_i ln[1 + v^{-1} eta_{it}^2]

where A(v) = Gamma((v+n)/2) * Gamma(v/2)^{n-1} / Gamma((v+1)/2)^n
and eta_t = (t_v^{-1}(u_{1t}), ..., t_v^{-1}(u_{nt}))'.
"""

import numpy as np
from scipy.stats import norm, t as tdist, rankdata
from scipy.special import gammaln
from scipy.optimize import minimize


def edf(x):
    """Empirical distribution function: rank/(T+1)."""
    return rankdata(x) / (len(x) + 1)


def _dcc_dynamics(a, b, z, Omega):
    """
    Compute the DCC Q_t and Psi_t sequences.

    Parameters
    ----------
    a, b : float
        DCC parameters, a >= 0, b >= 0, a + b < 1.
    z : ndarray
        Standardized residuals, shape (T, n).
    Omega : ndarray
        Unconditional correlation matrix, shape (n, n).

    Returns
    -------
    Psi : list of ndarray
        Time-varying correlation matrices, length T. Each is (n, n).
    Q_series : list of ndarray
        Raw Q_t matrices (before normalization), length T.
    """
    T, n = z.shape
    c = 1 - a - b  # weight on unconditional correlation

    Q_prev = Omega.copy()
    Psi_list = []
    Q_list = []

    for t in range(T):
        if t == 0:
            Q_t = Omega.copy()
        else:
            zt_1 = z[t - 1].reshape(-1, 1)
            Q_t = c * Omega + a * (zt_1 @ zt_1.T) + b * Q_prev

        # Normalize Q_t to correlation matrix Psi_t
        diag_sqrt = np.sqrt(np.diag(Q_t))
        diag_sqrt_inv = 1.0 / diag_sqrt
        Psi_t = Q_t * np.outer(diag_sqrt_inv, diag_sqrt_inv)

        Psi_list.append(Psi_t)
        Q_list.append(Q_t)
        Q_prev = Q_t

    return Psi_list, Q_list


def _ll_dcc_gaussian(params, z, Omega):
    """
    Negative log-likelihood of the DCC model under Gaussian copula.

    From eq 72:
        LL = -1/2 sum_t [ ln|Psi_t| + eta_t'(Psi_t^{-1} - I) eta_t ]

    For GARCH/GJR-Normal marginals, eta_t = z_t.
    For EDF marginals, z_t = Phi^{-1}(EDF(r_t)) is precomputed and passed as z.
    """
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e15

    T, n = z.shape
    Psi_list, _ = _dcc_dynamics(a, b, z, Omega)
    I_n = np.eye(n)

    ll = 0.0
    for t in range(T):
        Psi_t = Psi_list[t]
        eta_t = z[t]
        try:
            sign, logdet = np.linalg.slogdet(Psi_t)
            if sign <= 0:
                return 1e15
            Psi_inv = np.linalg.solve(Psi_t, I_n)
        except np.linalg.LinAlgError:
            return 1e15

        # eta'(Psi^{-1} - I)eta = eta'Psi^{-1}eta - eta'eta
        quad = eta_t @ Psi_inv @ eta_t - eta_t @ eta_t
        ll += -0.5 * (logdet + quad)

    return -ll  # return negative for minimization


def _ll_dcc_t(params, u, Omega_z, n):
    """
    Negative log-likelihood of the DCC model under Student t copula.

    Parameters
    ----------
    params : array-like
        [a, b, nu] where nu is degrees of freedom.
    u : ndarray
        Pseudo-observations in (0,1) from EDF, shape (T, n).
    Omega_z : ndarray
        Unconditional correlation of initial z (used for Q dynamics).
    n : int
        Number of assets.

    From eq 76:
        LL = T/2 ln A(v)
             - 1/2 sum_t ln|Psi_t|
             - (v+n)/2 sum_t ln[1 + v^{-1} eta_t' Psi_t^{-1} eta_t]
             + (v+1)/2 sum_t sum_i ln[1 + v^{-1} eta_{it}^2]

    where eta_t = (t_v^{-1}(u_{1t}), ..., t_v^{-1}(u_{nt}))'
    and A(v) = Gamma((v+n)/2) Gamma(v/2)^{n-1} / Gamma((v+1)/2)^n.

    Note: The DCC dynamics (Q_t recursion) run on z_t, but the copula
    likelihood is evaluated on eta_t. For the t copula, eta depends on v,
    so we recompute it inside the likelihood.

    Implementation detail:
        We run the Q_t dynamics on the Phi^{-1}(u) transform (same z as
        Gaussian case) to keep the dynamics comparable, and evaluate the
        copula density using t_v^{-1}(u) for the quadratic forms.
    """
    a, b, nu = params
    if a < 0 or b < 0 or a + b >= 0.9999 or nu <= 2.01:
        return 1e15

    T_obs = u.shape[0]

    # eta for t copula: t_v^{-1}(u)
    eta = tdist.ppf(u, nu)

    # z for DCC dynamics: Phi^{-1}(u) — same as Gaussian case
    z = norm.ppf(u)
    Psi_list, _ = _dcc_dynamics(a, b, z, Omega_z)

    # A(v) = Gamma((v+n)/2) * Gamma(v/2)^{n-1} / Gamma((v+1)/2)^n
    ln_A = (gammaln((nu + n) / 2)
            + (n - 1) * gammaln(nu / 2)
            - n * gammaln((nu + 1) / 2))

    I_n = np.eye(n)
    ll = T_obs * ln_A

    for t in range(T_obs):
        Psi_t = Psi_list[t]
        eta_t = eta[t]
        try:
            sign, logdet = np.linalg.slogdet(Psi_t)
            if sign <= 0:
                return 1e15
            Psi_inv = np.linalg.solve(Psi_t, I_n)
        except np.linalg.LinAlgError:
            return 1e15

        quad_joint = eta_t @ Psi_inv @ eta_t
        quad_marg = np.sum(eta_t ** 2)

        ll += -0.5 * logdet
        ll += -(nu + n) / 2 * np.log(1 + quad_joint / nu)
        ll += (nu + 1) / 2 * np.sum(np.log(1 + eta_t ** 2 / nu))

    return -ll


def estimate_dcc_gaussian(z, x0=(0.02, 0.95), maxiter=10000):
    """
    Estimate DCC parameters (a, b) under Gaussian copula.

    Parameters
    ----------
    z : ndarray
        Standardized residuals, shape (T, n). These are either:
        - Phi^{-1}(EDF(r)) for EDF marginals, or
        - r / sqrt(h) for GARCH/GJR-Normal marginals.
    x0 : tuple
        Initial guess for (a, b).
    maxiter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'Psi' (list of correlation matrices),
        'rho_series' (T x n_pairs array of pairwise correlations),
        'log_likelihood', 'success'.
    """
    Omega = np.corrcoef(z.T)

    result = minimize(
        _ll_dcc_gaussian, x0, args=(z, Omega),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-10}
    )

    a, b = result.x
    Psi_list, _ = _dcc_dynamics(a, b, z, Omega)

    # Extract pairwise dynamic correlations
    T, n = z.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rho_series = np.zeros((T, len(pairs)))
    for t in range(T):
        for k, (i, j) in enumerate(pairs):
            rho_series[t, k] = Psi_list[t][i, j]

    return {
        "a": a, "b": b,
        "Psi": Psi_list,
        "rho_series": rho_series,
        "pair_labels": pairs,
        "Omega": Omega,
        "log_likelihood": -result.fun,
        "success": result.success,
    }


def estimate_dcc_t(u, x0=(0.02, 0.95, 8.0), maxiter=10000):
    """
    Estimate DCC parameters (a, b, nu) under Student t copula.

    Parameters
    ----------
    u : ndarray
        Pseudo-observations in (0,1) from EDF, shape (T, n).
    x0 : tuple
        Initial guess for (a, b, nu).
    maxiter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'nu', 'Psi', 'rho_series',
        'log_likelihood', 'success'.
    """
    n = u.shape[1]

    # For Q dynamics, use z = Phi^{-1}(u)
    z = norm.ppf(u)
    Omega_z = np.corrcoef(z.T)

    result = minimize(
        _ll_dcc_t, x0, args=(u, Omega_z, n),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-10}
    )

    a, b, nu = result.x
    Psi_list, _ = _dcc_dynamics(a, b, z, Omega_z)

    T = u.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rho_series = np.zeros((T, len(pairs)))
    for t in range(T):
        for k, (i, j) in enumerate(pairs):
            rho_series[t, k] = Psi_list[t][i, j]

    return {
        "a": a, "b": b, "nu": nu,
        "Psi": Psi_list,
        "rho_series": rho_series,
        "pair_labels": pairs,
        "Omega": Omega_z,
        "log_likelihood": -result.fun,
        "success": result.success,
    }
