# Changelog

## [2.0.0] - 2026-04-14

### Added
- `exercise2/data.py`: Loads crypto.xlsx, extracts BTC/ETH/ADA daily returns (T=1827, 2020-01-02 to 2025-01-01).
- `exercise2/descriptive.py`: Computes mean, std, skewness, kurtosis (regular), and sample Pearson correlation matrix.
- `exercise2/garch.py`: GARCH(1,1) and GJR(1,1) MLE estimation from scratch with zero mean, Normal innovations. Uses L-BFGS-B optimizer with parameter bounds. Full mathematical derivations in docstrings.
- `exercise2/dcc.py`: DCC model implementation with Q_t dynamics (eq 69), normalization to Psi_t (eq 68). Gaussian copula LL (eq 72) and Student t copula LL (eq 76). Estimation via Nelder-Mead. Returns time-varying pairwise correlation series.
- `exercise2/plotting.py`: Dynamic correlation time series plots (3-panel) and multi-model comparison overlay plot.
- `exercise2/main.py`: Orchestrator for parts 2.1–2.4.
- `config.json`: Added `exercise2` section with data path, asset names, GARCH and DCC optimizer settings.
- `requirements.txt`: Added `pandas>=2.0` and `openpyxl>=3.1` dependencies.
- `readme.html`: Added full Exercise 2 documentation (sections 6–8).

## [1.0.1] - 2026-04-14

### Fixed
- `copula_sim.py`: Eliminated `RuntimeWarning` in `sim_fgm` when `psi` is near zero by using a `safe_psi` guard before division. `np.where` evaluates both branches; the guard prevents divide-by-zero in the unused branch.
- `main.py`: Replaced Unicode Greek letters (theta, rho, etc.) with ASCII equivalents in console output to avoid `UnicodeEncodeError` on Windows cp1252 terminals.

## [1.0.0] - 2026-04-14

### Added
- Initial project structure: `config.json`, `requirements.txt`, `readme.html`, `changelog.md`.
- `exercise1/copula_sim.py`: Simulation functions for Clayton, Survival Clayton, Gaussian, Student t, FGM copulas, and mixture copula. All use the conditional (inverse CDF) method with full derivations in docstrings.
- `exercise1/copula_likelihood.py`: Log-likelihood functions for Gaussian copula (eq 31) and Student t copula (eq 38) with step-by-step derivations.
- `exercise1/estimation.py`: EDF (rank/(T+1)), IFM/CML estimation wrappers for Gaussian (Brent optimizer) and t copula (Nelder-Mead).
- `exercise1/portfolio.py`: Portfolio return construction via inversion method, descriptive statistics (mean, std, skewness, kurtosis), VaR computation. Supports configurable marginal distributions.
- `exercise1/plotting.py`: Scatter plots (1.a), histogram with Normal overlay (1.b), mixture scatter (1.d). All save to configurable output directory.
- `exercise1/main.py`: Orchestrator loading `config.json`, running parts 1.a–1.d sequentially.
- `figures/` output directory for generated plots.
