# scripts/optim: optimizing the 2D LWFA cases

User-layer optimization (F3) of the small 2D LWFA run. Each driver reproduces one report; its run directory goes to `scripts/runs/optim-<name>-<stamp>/` with a manifest. Run them as `julia +1.13 --project=scripts -t 4 scripts/optim/<driver>.jl`; the header of each driver lists its arguments.

## The layer (WP-058)

`problem.jl` connects a case to Optimization.jl.

- `ScaledBox(lo, hi, scale)` is the box of the physical parameters. The optimizers see `z = x ./ scale`, which is O(1) (A1).
- `Objective(J, box)` runs the case `J(x...)` and logs every run with its kind and wall time. `o(z)` is a plain run. `value_gradient(o, z)` is one run with ForwardDiff duals, one partial per parameter, and returns the value and the gradient in `z`.
- `maximize(o, z0)` maximizes over the box with L-BFGS-B (OptimizationLBFGSB), at one run with duals per point. `maximize_box(fg, z0, lo, hi)` does the same for any `fg(z) -> (f, ∇f)`. Any other solver of Optimization.jl that takes bounds can be passed as `solver`.
- `bayes.jl` is the Bayesian optimizer of WP-048, a Gaussian process with expected improvement, written out because Julia has no maintained one to take instead. `bayesopt(o)` takes the same objective.

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

The reports record runs made before WP-058 with the hand-written projected BFGS that `problem.jl` replaced, so their call counts and paths are that optimizer's.

Tests: `julia +1.13 --project=scripts scripts/optim/test/runtests.jl` (T-760 and T-761 for the layer, T-660 to T-665 and T-680 to T-682 for the cases, and T-620 to T-623 for the Bayesian optimizer).
