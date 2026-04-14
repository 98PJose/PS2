# Changelog

## [4.0.0] - 2026-04-14

### Added
- `exercise4/black_scholes.py`: `bs_call_put` — closed-form European call/put prices (port of the course routine `BS.m`). Vectorised over the strike grid.
- `exercise4/copulas.py`: Bivariate shock generators producing (Z_1, Z_2) with N(0,1) marginals under Gaussian / Student t / Clayton copulas. Gaussian uses the direct Cholesky recombination (matches `ccK_simuRet2.m`); t and Clayton reuse `exercise1.copula_sim` and apply Phi^{-1} to the dependent uniforms.
- `exercise4/option_portfolio.py`: `cc_return`, `pp_return` (eqs 4–5 from the Option-based portfolios slides) and `portfolio_returns_over_strikes` which broadcasts to yield (N, NK) return matrices for cc / cc2 / pp / pp2 in one call.
- `exercise4/stats.py`: Column-wise mean, std, skew, kurtosis over the strike grid with MATLAB defaults (biased estimators, non-excess kurtosis so Normal = 3).
- `exercise4/plotting.py`: 4-panel figure per scenario with slide-matching line styles (cc dashed blue, cc2 solid blue, pp dashed red, pp2 solid red).
- `exercise4/main.py`: Orchestrator running 4.1–4.4 and saving `fig_4_4_1_gaussian_high.png`, `fig_4_4_2_gaussian_low.png`, `fig_4_4_3_student_t.png`, `fig_4_4_4_clayton.png`. Prints an ATM (K = 100) summary table per scenario.
- `config.json`: Added `exercise4` block with N = 100_000, strike grid 85–115 step 1, GBM parameters (S0 = 100, r = 0.03, mu = 0.08, sigma = 0.2, T = 1), per-part copula parameters (rho = 0.9 / 0.3, (rho, nu) = (0.3, 5), theta = 10) and per-scenario RNG seeds.
- `readme.html`: Added Exercise 4 documentation (sections 14–19) plus an Exercise 1 section header. Project-structure tree reordered to put `exercise3/` and `exercise4/` after `exercise2/`.

## [3.0.0] - 2026-04-14

### Added
- `exercise3/nm_copula.py`: Bivariate Normal Mixture (NM) copula conditional CDF and numerical q-quantile inversion. `nm_conditional_cdf` evaluates C^NM(u_1|u_2) vectorised; `nm_quantile_u1` / `nm_quantile_curve` invert on u_1 with `scipy.optimize.brentq` (strictly monotone in u_1, so a unique root exists for every q in (0,1)).
- `exercise3/plotting.py`: Two-panel figure `fig_3_nm_quantile_curves.png` showing the q-quantile curves both on the unit square (u_2, u_1) and on the N(0,1) return scale (r_2 = Phi^{-1}(u_2), r_1 = Phi^{-1}(u_1)).
- `exercise3/main.py`: Orchestrator for Exercise 3. Reports verification that C^NM(hat{u}_1|u_2) equals q at the solved roots, plus tables of u_1 and r_1 at representative conditioning values.
- `config.json`: Added `exercise3` block with pi = 0.3, rho_1 = -0.7, rho_2 = 0.4, q ∈ {0.05, 0.25, 0.5, 0.75, 0.95}, grid size 401, and eps = 1e-4.
- `readme.html`: Added Exercise 3 section with the NM copula formula, q-quantile definition, module documentation, configuration table, and a note on the slide-consistent u_1-inversion convention adopted (the literal "back out u_2" reading of the problem text admits no solution for q outside (0.3, 0.7) with the given parameters).

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
