# Changelog

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
