# Changelog

## [6.1.0] - 2026-04-15

### Added
- `report_utils.py`: Shared LaTeX-emission helpers. `TexWriter` accumulates `\newcommand` macros (scalar `cmd`) and multi-line tabular bodies (`body`) and saves them atomically; `fnum`, `fint`, `fpct`, `fsig` format numbers for math-mode insertion (NaN/inf handled); `generated_dir(cfg)` resolves the output folder from `config.json` (`output.tex_generated_dir`, default `report/generated`).
- `report/generated/ex{1..5}.tex`: Auto-generated macro files consumed by the LaTeX report. Regenerated on every `python main.py` run from each `exerciseN/main.py`.

### Changed
- `exercise1/main.py`, `exercise2/main.py`, `exercise3/main.py`, `exercise4/main.py`, `exercise5/main.py`: Every orchestrator now emits its numeric results through a `TexWriter` at the end of `main()`. The emitted macros cover dependence measures, portfolio statistics, IFM/CML estimates (Ex1), descriptive table, DCC a/b/persistence/LL for EDF/GARCH/GJR, t-copula DCC (Ex2), NM-quantile table (Ex3), ATM option-portfolio table (Ex4), breach-rate + fitted-parameter tables (Ex5).
- `report/sections/exercise{1..5}.tex`: All previously hardcoded numbers replaced by `\ExOne…`, `\ExTwo…`, `\ExThree…`, `\ExFour…`, `\ExFive…` macros (scalars and `*Body` tabular rows). Tables now reference a body macro + the toprule/midrule/bottomrule skeleton, so the .tex sources carry no numeric drift from the Python outputs.
- `report/main.tex`: Added `\input{generated/ex{1..5}.tex}` in the preamble so every macro is defined before `\begin{document}`.

### Fixed
- **Numeric drift between code and LaTeX**: prior to this refactor, sample statistics, parameter estimates, and table entries were hand-copied into the `.tex` sources. They are now sourced directly from Python at build time — a single `python main.py` regenerates `report/generated/` and recompiles the PDF, guaranteeing the report always prints the values produced by the code.

## [6.0.0] - 2026-04-15

### Added
- `main.py` (root): Top-level orchestrator. Sequentially runs `exercise{1..5}.main`, then compiles the LaTeX report under `report/` (two pdflatex passes for cross-refs / TOC). CLI flags: `--only N [...]`, `--skip-exercises`, `--skip-latex`, `--latex-engine`, `--latex-passes`. Captures per-stage success, wall-clock time, and prints a final pass/fail summary; exits with code 1 on any failure.
- `report/`: Full LaTeX report covering all five exercises.
  - `report/main.tex`: Master document (preamble, abstract, intro, conclusions). Includes `siunitx`, `booktabs`, `cleveref`, `listings`, custom math macros. Graphics path set to `../figures/`.
  - `report/sections/exercise1.tex`: Bivariate copula simulation, conditional inversion derivations (Clayton, Survival Clayton, Gaussian, Student-t, FGM), 1.a dependence-measure table and scatter, 1.b portfolio statistics + VaR table and histogram, 1.c Gaussian IFM (eq 31 LL derivation), 1.d mixture + t-copula IFM (eq 38 LL derivation).
  - `report/sections/exercise2.tex`: GARCH(1,1) eq 67, GJR(1,1) eq 71, DCC dynamics eqs 68-69, Gaussian DCC LL eq 72, Student-t DCC LL eq 76 — all with full derivations. Tables for 2.1 descriptives, 2.2a/b/c parameter estimates, 2.3 path comparison summary, 2.4 t-copula DCC.
  - `report/sections/exercise3.tex`: NM-copula conditional CDF derivation, Brent-inversion in u_1 with the slide-consistent convention remark, results table and figure.
  - `report/sections/exercise4.tex`: Black–Scholes pricing, CC/PP eqs 4–5, 4-scenario ATM table (4.1–4.4) + figures, copula-invariance discussion for cc/pp vs cc2/pp2.
  - `report/sections/exercise5.tex`: Berger Section 3.1 simulation design, Kupiec UC eq 11, Clayton log-density derivation, fitted-parameter and breach-rate tables (95%/99%), bar-chart figure, discussion linking misspecification to coverage failures.
- `changelog.md`: This entry.
- `readme.html`: Added `report/` documentation section.

## [5.0.0] - 2026-04-15

### Added
- `exercise5/dgp.py`: Clayton DGP with N(0,1) marginals (reuses `sim_clayton` from `exercise1.copula_sim`) matching Berger (2016) Section 3.1.
- `exercise5/estimation.py`: Copula MLE with marginals fixed at N(0,1). `estimate_gaussian` reuses `ll_gaussian_copula`, `estimate_t` reuses `ll_t_copula` on a tanh/log reparameterisation, and a new `clayton_log_density` + `estimate_clayton` derive the Clayton copula density analytically and optimise by bounded Brent.
- `exercise5/var_forecast.py`: Empirical portfolio VaR from 10 000 simulated returns drawn from the fitted copula (Berger eq 9). Uses `sim_gaussian`, `sim_student_t`, `sim_clayton` from `exercise1`.
- `exercise5/backtesting.py`: Kupiec unconditional-coverage test (eq 11) with chi-square(1) p-value and 95% / 99% rejection flags; handles N = 0 and N = T boundary cases.
- `exercise5/simulation.py`: Monte-Carlo driver; optionally parallel via `ProcessPoolExecutor`.
- `exercise5/plotting.py`: Grouped bar chart of empirical failure rates with Kupiec rejection annotations and nominal reference line.
- `exercise5/main.py`: Orchestrator producing a Berger-Table-2-style summary, `figures/tab_5_failure_rates.txt` (saved table + fitted-parameter summary), and `fig_5_failure_rates.png`.
- `config.json`: Added `exercise5` block (theta ∈ {0.5, 1.5, 2.5}, n_obs=1001, n_sim=10 000, n_rep=1000, weights=[0.5, 0.5]). Note: paper uses n_rep=10 000.
- `readme.html`: Added Exercise 5 section (simulation design, module documentation, configuration, usage).

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
