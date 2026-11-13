# WP-048: Bayesian optimization against gradient-based BFGS on the WP-045 problem

| | |
|---|---|
| Work package | WP-048 (goals A0, A1, A7) |
| Commits | the optimizer, its tests and the comparison script `a0f7fe5`; this report and the run's figure in the commit after it |
| Runs | `optim-bo_compare-20261109T135950` (commit `a0f7fe5`, clean, 4 threads, 5105 s). An earlier run, `optim-bo_compare-20261109T123027` (commit `6766bff`), had only ξ = 0.01; its BO and BFGS values are identical to this run's |
| Input hashes | SHA-256 of every input file in the run's manifest: `wp-048-bayesian-baseline/manifest.toml` |
| Environment | Julia 1.13.3, KernelAbstractions 0.10.4 CPU backend; Intel Xeon @ 2.10GHz (the model string the CPU reports), 4 vCPUs of a shared cloud VM |
| Figures and data | `wp-048-bayesian-baseline/bo_compare.png` with its sidecar `bo_compare.json`; every call (kind, point, value, seconds) and every fitted length scale in `results.json` |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## 1. Setup

- **The problem** is WP-045's, unchanged (`scripts/optim/wake2d_case.jl`; `reports/wp-045-wake-optimization.md` §1): maximize the RMS on-axis wake amplitude J of the small 2D LWFA case over the pulse duration τ ∈ [4, 9] and the focus x_f ∈ [60, 130], at a₀ = 1 and n₀ = 0.1, in the scaled coordinates z = (τ/6, x_f/90). Every call is a full 512-step run.
- **BFGS** is WP-045's projected BFGS (`maximize.jl`), with ForwardDiff gradients through the run and WP-045's stop (projected gradient below 10⁻⁴). Each call is one run with a dual and returns the value and the gradient.
- **Bayesian optimization** (`scripts/optim/bayes.jl`) is a textbook GP-EI loop, written out here so it adds no dependency:
  - A Gaussian process on the unit box with a Matérn-5/2 kernel, one length scale per parameter, a signal variance and a noise variance, on values standardized after every call.
  - Hyperparameters by maximum marginal likelihood (the WP-045 maximizer with ForwardDiff on the log-hyperparameters, from three fixed starts and the previous fit). Bounds: length scales 0.02 to 5 box widths, signal variance 0.01 to 100, noise variance 10⁻⁸ to 10⁻¹.
  - Expected improvement with margin ξ (standardized units), maximized over 2048 counter-based candidates and refined from the best five with gradients. ξ = 0 is the default; ξ = 0.01, a common default elsewhere, is run alongside (§2).
  - A 5-point Latin hypercube first, then one point per call. Each call is a plain run and returns the value only. There is no stopping rule; every run spends its budget of 30 calls.
  - The randomness is counter-based (splitmix64 of seed and counter), so a seed repeats bit for bit; the ξ = 0.01 runs here repeat the earlier run's values exactly.
- **Tests** (`scripts/optim/test/bayes.jl`, included from `runtests.jl`):
  - T-620: the Latin hypercube's stratification, distinct orders per coordinate, reproducibility.
  - T-621: the GP interpolates its data and fits a length scale in range; the fit improves on each of its fixed starts and keeps a good warm start; expected improvement's limits; its ForwardDiff gradient against central differences.
  - T-622: BO reaches the Branin minimum 5/(4π) within 2 × 10⁻⁴ in 40 calls from seeds 1 to 6 (measured 7 × 10⁻⁶ to 3.4 × 10⁻⁵; Branin spans about 300 on its box) and repeats bit for bit.
  - T-623: the proposal follows the seed; the guard keeps a proposal away from evaluated points (with ξ ≪ 0, where the refinement converges onto the best data point); `reach` counts calls and seconds.
- **The comparison** (`scripts/optim/bo_compare.jl`): five seeds. Seed s gives BO its Latin hypercube and BFGS its start (that design's first point), so the methods start from the same information. Both kinds of call are compiled before any timing. Every call is logged with its kind and wall time.
- **The measure.** J* is the largest value any of the fifteen runs evaluated. A run reaches tolerance ε at its first call whose best value so far is within ε·J* of J*. Time is the summed wall time of the simulation calls. The GP's own work is not counted: the whole run's wall time exceeds the summed calls by 16 s over 250 BO proposals.

## 2. Results

![Gap to the best value found against calls and against simulation time, for five runs each of BO with ξ = 0, BO with ξ = 0.01 and BFGS; the evaluated points over the box](wp-048-bayesian-baseline/bo_compare.png)

J* = 0.2586462, at (τ, x_f) = (7.742, 112.97), the point WP-045 reports. Each entry is calls / simulation seconds to reach the tolerance; "—" is not reached within the budget. "total" is the run's summed simulation time (BO: 30 calls; BFGS: until its own stop).

| method | seed | calls | 1 % | 0.1 % | 0.01 % | best J | at (τ, x_f) | 1 − best/J* | total |
|---|---|---|---|---|---|---|---|---|---|
| BO, ξ = 0 | 1 | 30 | 8 / 88 s | 8 / 88 s | 11 / 122 s | 0.2586462 | (7.742, 112.96) | 2.5e-09 | 338 s |
| BO, ξ = 0 | 2 | 30 | 8 / 80 s | 11 / 111 s | 11 / 111 s | 0.2586462 | (7.742, 112.97) | 1.2e-09 | 316 s |
| BO, ξ = 0 | 3 | 30 | 5 / 54 s | 9 / 96 s | 11 / 117 s | 0.2586461 | (7.742, 112.98) | 9.5e-09 | 314 s |
| BO, ξ = 0 | 4 | 30 | 2 / 22 s | 10 / 105 s | 13 / 137 s | 0.2586462 | (7.742, 112.96) | 2.5e-10 | 320 s |
| BO, ξ = 0 | 5 | 30 | 9 / 96 s | 9 / 96 s | 11 / 118 s | 0.2586462 | (7.742, 112.97) | 2.0e-09 | 328 s |
| BO, ξ = 0.01 | 1 | 30 | 10 / 115 s | 10 / 115 s | — | 0.2585931 | (7.739, 111.32) | 2.1e-04 | 348 s |
| BO, ξ = 0.01 | 2 | 30 | 8 / 91 s | 11 / 125 s | — | 0.2586117 | (7.798, 114.03) | 1.3e-04 | 334 s |
| BO, ξ = 0.01 | 3 | 30 | 5 / 51 s | 9 / 96 s | 14 / 148 s | 0.2586406 | (7.735, 113.48) | 2.1e-05 | 326 s |
| BO, ξ = 0.01 | 4 | 30 | 2 / 22 s | 10 / 111 s | — | 0.2585852 | (7.660, 114.21) | 2.4e-04 | 312 s |
| BO, ξ = 0.01 | 5 | 30 | 9 / 104 s | 9 / 104 s | 22 / 244 s | 0.2586410 | (7.774, 113.19) | 2.0e-05 | 326 s |
| BFGS | 1 | 8 | 3 / 149 s | 5 / 244 s | 6 / 292 s | 0.2586462 | (7.742, 112.97) | 1.7e-09 | 386 s |
| BFGS | 2 | 8 | 3 / 146 s | 4 / 196 s | 4 / 196 s | 0.2586461 | (7.742, 112.96) | 6.6e-09 | 392 s |
| BFGS | 3 | 7 | 3 / 144 s | 4 / 193 s | 4 / 193 s | 0.2586462 | (7.742, 112.97) | 0 | 339 s |
| BFGS | 4 | 8 | 3 / 140 s | 3 / 140 s | 4 / 186 s | 0.2586462 | (7.742, 112.97) | 1.8e-09 | 375 s |
| BFGS | 5 | 7 | 3 / 143 s | 3 / 143 s | 5 / 240 s | 0.2586461 | (7.744, 112.99) | 1.1e-07 | 336 s |

**Cost of a call.** The median plain run took 10.9 s and the median run with a dual (three partials) 48.1 s, a ratio of 4.4 (4.3 in the earlier run). The physics breaker measured 5.3 at one point in a separate process. WP-045's report gives "about 2.5"; WP-045's run records no plain-run timing, and from its own totals (49 grid runs and the other plain runs within 866 s, gradient calls at about 46 s) the ratio is about 5, so 2.5 is not supported by its record.

**What the numbers say.**
- **Calls.** BFGS needs fewer: 3 for 1 %, 3 to 5 for 0.1 %, 4 to 6 for 0.01 %. BO with ξ = 0 needs 2 to 9, 8 to 11 and 11 to 13.
- **Time.** A gradient call costs 4.4 plain runs, so BO with ξ = 0 is ahead at every tolerance: 0.01 % in 111 to 137 s against 186 to 292 s for BFGS, 0.1 % in 88 to 111 s against 140 to 244 s.
- **Location.** Both find the optimum to four digits from every seed; BO with ξ = 0 ends within 10⁻⁸ of J* in every run.
- **Stopping.** BFGS stops on its own once the gradient is small. BO has no stopping rule here, and after reaching 0.01 % it keeps sampling next to the optimum (11 to 15 of each run's last 17 calls lie within 0.1 in τ and 1.5 in x_f of it). A user who had to know when to stop would spend the full budget, about 320 s, against BFGS's 336 to 392 s to its own stop. The time to reach a tolerance is the fair measure only if something else says the tolerance has been reached.
- **The margin ξ = 0.01 stalls BO.** It is in standardized units, so it is 0.01 times the spread of the sampled values, which is 0.016 to 0.026 in these runs: 1.6 to 2.6 × 10⁻⁴ in J, 0.6 to 1 × 10⁻³ of J*. Expected improvement discounts any gain smaller than that. With it, BO reaches 0.01 % in 2 of 5 runs and ends 2 × 10⁻⁵ to 2.4 × 10⁻⁴ below J*, up to 0.09 in τ and 1.7 in x_f off. The physics breaker found the same on analytic functions: on Branin (seeds 1 to 10) the gaps were 0.006 to 0.13 with ξ = 0.01 and below 5 × 10⁻⁵ with ξ = 0.
- **The GP's view.** The fitted length scales end at 2.6 to 4.3 box widths with ξ = 0 (3.6 to 5 with ξ = 0.01): over the box the objective looks smooth and close to quadratic, which suits both methods.

**In short**, on this smooth two-parameter problem, BO with ξ = 0 reaches the optimum with 2 to 3 times as many calls as BFGS, but since each of its calls costs a quarter as much, it gets there in about 60 % of BFGS's time. Without a stopping rule the BO advantage disappears in total time. The comparison would shift with more parameters: forward-mode gradients cost about one partial per parameter (a reverse-mode gradient would not), while BO's calls to reach a tolerance usually grow quickly with dimension. Neither direction is measured here.

## 3. Open issues

- **The objective.** WP-051 found that the WP-045 objective's optimum is set by the read-out time (its gradient changes sign within ±5 c/ω₀ of it) and is replacing it with a lab-frame slab objective whose optimum lies at longer τ and further focus. This report compares the two methods on the WP-045 objective as merged; the comparison is valid as a method comparison on that function, not as a statement about the physical optimum. Rerunning `bo_compare.jl` on the slab objective once it lands is a follow-up.
- **Breadth.** One problem, two parameters, five seeds, one discretization. WP-045's optimum is also not converged in Δy.
- **BO variants.** No stopping rule, no gradient-enhanced GP, no batch proposals; each would change the balance. A stopping rule (for example, expected improvement below a threshold for several calls) is the most important for a fair total-time comparison.
- **The gradient's cost.** The dual carries three partials for two parameters (#375); after the KernelAbstractions bump, two partials should make a gradient call somewhat cheaper.
