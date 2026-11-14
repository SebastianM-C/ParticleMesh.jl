# WP-058: the optimization scripts on Optimization.jl

The hand-written projected BFGS of `scripts/optim/maximize.jl` is replaced by Optimization.jl's bound-constrained L-BFGS-B (OptimizationLBFGSB, which calls the Fortran L-BFGS-B 3.0), behind a small layer in `scripts/optim/problem.jl`: a scaled box, an objective that logs every run, and one ForwardDiff run per point for the value and the gradient. Rerun at commit `172c47d`, the WP-045 and WP-052 drivers pass every check they assert and end at the reported optima. WP-045 ends at (τ, x_f) = (7.742, 112.97), J = 0.2586462, from both starts, in 6 and 5 runs where the old optimizer took 6 and 7. WP-052 ends at (τ, n₀, x_c) = (4.9356, 0.072003, 36.556), J = 14.0537979, in 17 and 19 runs where it took 10 and 15. One difference: OptimizationLBFGSB fixes L-BFGS-B's stop on a small decrease of the objective (#437), and WP-052's start A stopped there with |∇J|∞ = 8.1 × 10⁻⁴, above the gtol of 10⁻⁴. The drivers therefore check that both starts stop on one of L-BFGS-B's convergence tests at the same point, instead of checking the gradient.

| | |
|---|---|
| Work package | WP-058 (goals A0, A1, A7); the cases are WP-045, WP-051 and WP-052's, the Bayesian baseline WP-048's |
| Commit | `172c47d` (clean) for every run below |
| Runs | `optim-witness2d-20261114T002912` (4900 s), `optim-wake2d-20261114T015140` (924 s), `optim-converge2d-optimize-20261114T020744`, `optim-bo_compare-20261114T021439` (1 seed, budget 12, 353 s) |
| Input hashes | `wp-058-optimization-jl/wake2d-manifest.toml`, `witness2d-manifest.toml` |
| Environment | Julia 1.13.4, KernelAbstractions CPU backend, 4 threads; Intel Xeon @ 2.10GHz (the model string the CPU reports), 4 vCPUs of a shared cloud VM |
| Figures | `wp-058-optimization-jl/wake2d.png` and `witness2d.png`, with their JSON sidecars (every number of the runs) |
| Status | measured, not independently reviewed |

## Results

| case | start | runs (old BFGS) | runs (L-BFGS-B) | stop | end point | J | reported (WP-045, WP-051, WP-052) |
|---|---|---|---|---|---|---|---|
| WP-045 wake | (4.5, 70) | 6 | 6 | gtol | (7.7418, 112.966) | 0.25864615 | (7.742, 112.96), 0.2586462 |
| WP-045 wake | (8.5, 125) | 7 | 5 | gtol | (7.7419, 112.966) | 0.25864615 | (7.742, 112.97), 0.2586462 |
| WP-051 slab, λ₀/16 × λ₀/4 | (4.5, 70) | 5, from (7.742, 112.97) | 7 | gtol | (7.8937, 115.825) | 0.3403326 | (7.894, 115.83), 0.34033 |
| WP-052 witness | (4.5, 0.05, 36) | 10 | 17 | ftol, \|∇J\|∞ = 8.1e-4 | (4.93555, 0.0720026, 36.5560) | 14.0537979 | (4.9356, 0.072002, 36.556), 14.0537979 |
| WP-052 witness | (8.5, 0.095, 47) | 15 | 19 | gtol | (4.93557, 0.0720024, 36.5560) | 14.0537979 | as above |
| WP-052 at fixed a₀ = 1.654 | (4.5, 0.05, 36) | 47 | 15 | ftol, on τ's bound | (9.0, 0.07120, 36.62) | 19.185 | τ = 9, 19.19 |

The two WP-052 end points differ by at most 3.8 × 10⁻⁶ in z. Every other check of the two runs passes as before: the launched energy, AD against central differences (4.7 × 10⁻⁷), the grid scan (best cell 14.0391), the slices, the optimum inside the box, and τ on its bound at fixed a₀.

## Not shown, and where the data is

- **Calls to the optimum.** The fixed-energy problem took more runs than with the old BFGS. Its first L-BFGS-B step has length about 1 in z and goes to a corner of the box (from (4.5, 0.05, 36) to (9, 0.1, 30)), where the old optimizer capped steps at 0.1. Every corner of WP-052's box is well defined (the physics breaker evaluated all of them), so this costs runs, not results.
- **The Bayesian comparison** (`bo_compare.jl`) was run with one seed and a budget of 12 as a smoke test, not to repeat WP-048's numbers. BO with ξ = 0 came within 10⁻⁴ of J* in 53 s of simulation, and L-BFGS-B within 10⁻⁴ in 4 runs, 154 s. A plain run took 4.9 s and a run with duals 38 s on this machine; WP-048's machine had a ratio of 4.4.
- **Independent reruns** (the physics breaker, from the pull request): WP-045 from three other starts, two near corners, ends within 7 × 10⁻⁵ in z of the point above, in 5 or 6 runs against the old optimizer's 8; WP-051's slab from (6, 95) ends at (7.8943, 115.823); WP-052 from (8.8, 0.042, 32) ends at (4.93557, 0.072003, 36.55597) in 15 runs.
- The data: the JSON sidecars here, and the run directories named above (not committed: `scripts/runs/` is ignored).
