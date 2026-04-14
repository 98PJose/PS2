"""
GARCH(1,1) and GJR(1,1) estimation with Normal innovations and zero mean.

Both models are estimated by maximum likelihood. The conditional variance
is recursed forward, and the Gaussian log-likelihood is maximized.

GARCH(1,1) (eq 67 from slides):
    h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

    Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.
    Unconditional variance: sigma^2 = omega / (1 - alpha - beta).

GJR(1,1) (eq 71 from slides):
    h_t = omega + (alpha + gamma * D_{t-1}) * r_{t-1}^2 + beta * h_{t-1}

    where D_{t-1} = 1 if r_{t-1} < 0, 0 otherwise.
    Constraints: omega > 0, alpha >= 0, beta >= 0, gamma >= 0,
                 alpha + beta + gamma/2 < 1.
    Unconditional variance: sigma^2 = omega / (1 - alpha - beta - gamma/2).

In both cases, zero mean is assumed: r_t = sqrt(h_t) * z_t, z_t ~ N(0,1).
The log-likelihood (ignoring constant -T/2 ln(2*pi)):
    LL = -1/2 * sum[ ln(h_t) + r_t^2 / h_t ]
"""

import numpy as np
from scipy.optimize import minimize


def _garch_variance(params, r, model="garch"):
    """
    Compute the conditional variance series h_t.

    Parameters
    ----------
    params : array-like
        GARCH: [omega, alpha, beta].
        GJR:   [omega, alpha, beta, gamma].
    r : ndarray
        Return series, shape (T,).
    model : str
        "garch" or "gjr".

    Returns
    -------
    h : ndarray
        Conditional variance series, shape (T,).
    """
    T = len(r)
    h = np.zeros(T)

    if model == "garch":
        omega, alpha, beta = params
        # Initialize with unconditional variance
        h[0] = omega / max(1 - alpha - beta, 1e-6)
        for t in range(1, T):
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    elif model == "gjr":
        omega, alpha, beta, gamma = params
        h[0] = omega / max(1 - alpha - beta - gamma / 2, 1e-6)
        for t in range(1, T):
            leverage = gamma * (r[t - 1] < 0)
            h[t] = omega + (alpha + leverage) * r[t - 1] ** 2 + beta * h[t - 1]

    # Floor to avoid log(0) or division by zero
    h = np.maximum(h, 1e-12)
    return h


def _neg_log_likelihood(params, r, model):
    """
    Negative Gaussian log-likelihood for GARCH/GJR.

    LL = -1/2 * sum[ ln(h_t) + r_t^2 / h_t ]
    We return -LL for minimization.
    """
    h = _garch_variance(params, r, model)
    ll = -0.5 * np.sum(np.log(h) + r ** 2 / h)
    return -ll


def estimate_garch(r, maxiter=5000):
    """
    Estimate GARCH(1,1) parameters by MLE.

    Parameters
    ----------
    r : ndarray
        Return series (zero mean assumed), shape (T,).
    maxiter : int
        Maximum optimizer iterations.

    Returns
    -------
    dict
        Keys: 'omega', 'alpha', 'beta', 'h' (variance series),
        'z' (standardized residuals), 'log_likelihood', 'success'.

    Mathematical derivation:
        The GARCH(1,1) model specifies:
            r_t | F_{t-1} ~ N(0, h_t)
            h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

        The conditional log-likelihood of observation t is:
            l_t = -1/2 [ln(2*pi) + ln(h_t) + r_t^2/h_t]

        Summing and dropping the constant:
            LL = -1/2 sum[ln(h_t) + r_t^2/h_t]

        We maximize LL over (omega, alpha, beta) subject to:
            omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.
    """
    var0 = np.var(r)
    # Initial guess: omega = var0*0.05, alpha = 0.05, beta = 0.90
    x0 = [var0 * 0.05, 0.05, 0.90]
    bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]

    result = minimize(
        _neg_log_likelihood, x0, args=(r, "garch"),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12}
    )

    omega, alpha, beta = result.x
    h = _garch_variance(result.x, r, "garch")
    z = r / np.sqrt(h)

    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "persistence": alpha + beta,
        "h": h, "z": z,
        "log_likelihood": -result.fun,
        "success": result.success,
    }


def estimate_gjr(r, maxiter=5000):
    """
    Estimate GJR(1,1) parameters by MLE.

    Parameters
    ----------
    r : ndarray
        Return series (zero mean assumed), shape (T,).
    maxiter : int
        Maximum optimizer iterations.

    Returns
    -------
    dict
        Keys: 'omega', 'alpha', 'beta', 'gamma', 'h', 'z',
        'log_likelihood', 'success'.

    Mathematical derivation:
        The GJR(1,1) model extends GARCH with an asymmetric leverage term:
            h_t = omega + (alpha + gamma * D_{t-1}) * r_{t-1}^2 + beta * h_{t-1}

        where D_{t-1} = 1{r_{t-1} < 0} is the leverage indicator.

        The term gamma captures the asymmetric response to negative returns
        ("leverage effect"): negative shocks increase variance more than
        positive shocks of the same magnitude.

        The news impact curve is:
            h_t = omega + beta*h_{t-1} + alpha*eps^2           if eps >= 0
            h_t = omega + beta*h_{t-1} + (alpha+gamma)*eps^2   if eps < 0

        Constraints: omega > 0, alpha >= 0, beta >= 0, gamma >= 0,
                     alpha + beta + gamma/2 < 1 (stationarity).
    """
    var0 = np.var(r)
    x0 = [var0 * 0.05, 0.03, 0.90, 0.04]
    bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999), (0.0, 0.999)]

    result = minimize(
        _neg_log_likelihood, x0, args=(r, "gjr"),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12}
    )

    omega, alpha, beta, gamma = result.x
    h = _garch_variance(result.x, r, "gjr")
    z = r / np.sqrt(h)

    return {
        "omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma,
        "persistence": alpha + beta + gamma / 2,
        "h": h, "z": z,
        "log_likelihood": -result.fun,
        "success": result.success,
    }
