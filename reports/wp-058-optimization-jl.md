# WP-058: the optimization scripts on Optimization.jl

The hand-written projected BFGS of `scripts/optim/maximize.jl` is gone. A case is a plain `OptimizationProblem` (`case_problem` in `scripts/optim/problem.jl`: scaled unknowns, `MaxSense`, `AutoForwardDiff`, the box as bounds) that the drivers `solve` with Optimization.jl's L-BFGS-B; a callback records the points and a `TimerOutput` times the runs. The Bayesian baseline is one more solver of the same interface (`BayesianEI`). Rerun at commit `f50a7bf`, the WP-045 and WP-052 drivers pass every check they assert and end at the reported optima: WP-045 at (τ, x_f) = (7.742, 112.97), J = 0.2586462, in 6 and 5 points (old optimizer: 6 and 7); WP-052 at (τ, n₀, x_c) = (4.9356, 0.072003, 36.556), J = 14.0537979, in 17 and 19 points (old: 10 and 15). MadNLP with a limited-memory Hessian solves the WP-045 problem to the same value in 6 and 5 iterations. Found on the way: OptimizationLBFGSB takes no `fg`, so a point costs a plain run and a run with duals (#440); the callback's objective has a solver-dependent sign under `MaxSense` (#441); a run with two dual partials costs 8 to 10 plain runs, because the 2D Reference current deposition is 14 times slower with duals (#442).

| | |
|---|---|
| Work package | WP-058 (goals A0, A1, A7); the cases are WP-045, WP-051 and WP-052's, the Bayesian baseline WP-048's |
| Commit | `f50a7bf` (clean) for every run below |
| Runs | `optim-wake2d-20261114T221031` (1154 s), `optim-witness2d-20261114T223035` (5474 s), `optim-converge2d-optimize-20261115T000231`, `optim-solvers-20261115T001124`, `optim-bo_compare-20261115T003007` (1 seed, budget 12, 349 s) |
| Input hashes | `wp-058-optimization-jl/wake2d-manifest.toml`, `witness2d-manifest.toml`, `solvers-manifest.toml` |
| Environment | Julia 1.13.4, KernelAbstractions CPU backend, 4 threads; Intel Xeon @ 2.10GHz (the model string the CPU reports), 4 vCPUs of a shared cloud VM |
| Figures | `wp-058-optimization-jl/wake2d.png` and `witness2d.png`, with their JSON sidecars (every number of the runs); `solvers.json` |
| Status | measured, not independently reviewed |

## Results

A point costs L-BFGS-B one plain run (the value) and one run with duals (the gradient). The old BFGS took both from one run with duals.

| case | solver | start | points (old BFGS) | points | end point | J | \|∇J\|∞ in z | reported (WP-045, WP-051, WP-052) |
|---|---|---|---|---|---|---|---|---|
| WP-045 wake | L-BFGS-B | (4.5, 70) | 6 | 6 | (7.7418, 112.966) | 0.25864615 | 4.1e-6 | (7.742, 112.96), 0.2586462 |
| WP-045 wake | L-BFGS-B | (8.5, 125) | 7 | 5 | (7.7419, 112.966) | 0.25864615 | 2.1e-6 | (7.742, 112.97), 0.2586462 |
| WP-045 wake | MadNLP, limited memory | (4.5, 70) | | 7 plain runs, 8 with duals | (7.7410, 112.950) | 0.25864614 | | as above |
| WP-045 wake | MadNLP, limited memory | (8.5, 125) | | 6 plain runs, 7 with duals | (7.7414, 112.954) | 0.25864615 | | as above |
| WP-051 slab, λ₀/16 × λ₀/4 | L-BFGS-B | (4.5, 70) | 5, from (7.742, 112.97) | 7 | (7.8937, 115.825) | 0.3403326 | 2.3e-6 | (7.894, 115.83), 0.34033 |
| WP-052 witness | L-BFGS-B | (4.5, 0.05, 36) | 10 | 17 | (4.93555, 0.0720026, 36.5560) | 14.0537979 | 8.1e-4 | (4.9356, 0.072002, 36.556), 14.0537979 |
| WP-052 witness | L-BFGS-B | (8.5, 0.095, 47) | 15 | 19 | (4.93557, 0.0720024, 36.5560) | 14.0537979 | 1.0e-5 | as above |
| WP-052 at fixed a₀ = 1.654 | L-BFGS-B | (4.5, 0.05, 36) | 47 | 15 | (9.0, 0.07120, 36.62), τ on its bound | 19.185 | | τ = 9, 19.19 |

Every solve returned `Success`. All 14 checks of `witness2d.jl` and all checks of `wake2d.jl` pass: the launched energy, the run with duals giving the plain run's value, AD against central differences (1.2 × 10⁻⁸ and 4.7 × 10⁻⁷), the two starts ending at the same point (2.0 × 10⁻⁵ and 3.8 × 10⁻⁶ apart in z), the grid scans (best cells 0.257380 and 14.0391), the slices, the optimum inside the box, and τ on its bound at fixed a₀.

## Not shown, and where the data is

- **The stop.** OptimizationLBFGSB does not pass L-BFGS-B's tolerance on the decrease of the objective, about 2 × 10⁻⁹ relative, and loses the stop message (#437). WP-052's first start stopped on it with |∇J|∞ = 8.1 × 10⁻⁴, above the 10⁻⁴ asked of the projected gradient. So the drivers check that both starts return `Success` at the same point; the gradient in the table is computed once more at the solution.
- **Points to the optimum.** The fixed-energy problem took more points than with the old BFGS. Its first L-BFGS-B step has length about 1 in z and goes to a corner of the box, where the old optimizer capped steps at 0.1. This costs runs, not results.
- **The cost of a point.** In the solver comparison a plain run took 4.3 to 4.7 s and a run with duals 34 to 40 s. The plain run that L-BFGS-B adds per point is therefore about 12% of the point. The ratio of 8 itself is #442.
- **MadNLP** stopped at its own tolerance (`abstol = 1e-4` on its scaled optimality error), 0.016 in x_f from L-BFGS-B's point and 8 × 10⁻⁹ below it in J. Its callback reports a scaled, negated objective once per iteration (#441), so its cost here is the timer's run count.
- **The Bayesian comparison** (`bo_compare.jl`) was run with one seed and a budget of 12 as a smoke test, not to repeat WP-048's numbers. `BayesianEI` with ξ = 0 came within 10⁻⁴ of J* at its 11th point, 54 s into the search; L-BFGS-B at its 4th point, 122 s in.
- The data: the JSON sidecars here, and the run directories named above (not committed: `scripts/runs/` is ignored).
