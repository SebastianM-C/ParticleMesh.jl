# scripts/optim: optimizing the 2D LWFA cases

User-layer optimization (F3) of the small 2D LWFA run. Each driver reproduces one report; its run directory goes to `scripts/runs/optim-<name>-<stamp>/` with a manifest. Run them as `julia +1.13 --project=scripts -t 4 scripts/optim/<driver>.jl`; the header of each driver lists its arguments.

## The layer (WP-058)

`problem.jl` makes a case a plain `OptimizationProblem`, and the drivers call `solve` on it themselves.

```julia
prob = case_problem(wake_amplitude, WAKE_BOX, (4.5, 70.0))    # J, the box, the start in physical units
rec = Record(prob, LBFGSB())
sol = solve(prob, LBFGSB(); callback = rec, maxiters = 25, reltol = 1e-4)
sol.u .* WAKE_BOX.scale                                       # the optimum; sol.objective is J there
```

- A box is a named tuple `(; lo, hi, scale)` in physical units. The problem's unknown is `z = x ./ scale`, which is O(1) (A1), with `lb` and `ub` from the box, `sense = MaxSense` and gradients by `AutoForwardDiff()`.
- `Record` is a callback that keeps every point the solver reports, the objective there and the wall time.
- The problem's parameters hold a `TimerOutput` (`prob.p.timer`) that counts and times the runs of the case: "plain run" and "run with duals". `runs(timer, kind)` reads it.
- Any solver of Optimization.jl that takes bounds solves the same problem. OptimizationLBFGSB asks for the value and the gradient separately and does not take `fg` (#440), so a point costs it a plain run and a run with duals.
- `bayes.jl` is the Bayesian optimizer of WP-048, a Gaussian process with expected improvement, written out because Julia has no maintained one to take instead. It is one more solver, `solve(prob, BayesianEI(); maxiters = 30, callback = rec)`, with one plain run per point.

## The cases

| File | Case | Free parameters | Objective |
|---|---|---|---|
| `wake2d_case.jl` | WP-045: 512 steps, a₀ = 1, n₀ = 0.1 | τ ∈ [4, 9], focus ∈ [60, 130] (`WAKE_BOX`) | `wake_amplitude`, the RMS on-axis wake read at one time |
| `slab.jl` (with `wake2d_case.jl`) | WP-051 | as WP-045 | `slab_amplitude`, a wake amplitude that does not depend on when it is read |
| `witness2d_case.jl` | WP-052: 830 steps, fixed laser energy, a witness bunch | τ ∈ [4, 9], n₀ ∈ [0.04, 0.10], witness centre x_c ∈ [30, 50] (`WITNESS_BOX`) | `witness_gain`, the witness bunch's mean energy gain |

## The drivers

| Driver | Report | What it shows |
|---|---|---|
| `wake2d.jl` | `reports/wp-045-wake-optimization.md` | The first 2D optimum, against a grid scan, finite differences, slices and 1D theory |
| `bo_compare.jl` | `reports/wp-048-bayesian-baseline.md` | Calls and seconds to the optimum for Bayesian optimization and L-BFGS-B |
| `converge2d.jl` | `reports/wp-051-converged-optimization.md` | The optimum under refinement in Δx, Δy and read-out time, with the slab objective |
| `witness2d.jl` | `reports/wp-052-fixed-energy-optimization.md` | An interior optimum at fixed laser energy; at fixed a₀ the duration runs to its bound |
| `witness2d_resolution.jl` | the same report | The fixed-energy optimum at a finer grid (no optimizer) |
| `solvers.jl` | `reports/wp-058-optimization-jl.md` | The WP-045 problem solved by L-BFGS-B and by MadNLP with a limited-memory Hessian |

The reports of WP-045 to WP-052 record runs made with the hand-written projected BFGS that Optimization.jl replaced, so their call counts and paths are that optimizer's.

Tests: `julia +1.13 --project=scripts scripts/optim/test/runtests.jl` (T-760 to T-763 for the layer, T-660 to T-665 and T-680 to T-682 for the cases, and T-620 to T-623 for the Bayesian optimizer).
