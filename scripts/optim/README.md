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
- Any solver of Optimization.jl that takes bounds solves the same problem. OptimizationLBFGSB asks for the value and the gradient separately and does not take `fg` (#440), so a point costs it a plain run and a run with duals. Its `reltol` is L-BFGS-B's `pgtol`, an absolute tolerance on the largest component of the projected gradient in z.
- `bayes.jl` is the Bayesian optimizer of WP-048, a Gaussian process with expected improvement, written out because Julia has no maintained one to take instead. It is one more solver, `solve(prob, BayesianEI(); maxiters = 30, callback = rec)`, with one plain run per point.

## The cases

| File | Case | Free parameters | Objective |
|---|---|---|---|
| `wake2d_case.jl` | WP-045: 512 steps, a₀ = 1, n₀ = 0.1 | τ ∈ [4, 9], focus ∈ [60, 130] (`WAKE_BOX`) | `wake_amplitude`, the RMS on-axis wake read at one time |
| `slab.jl` (with `wake2d_case.jl`) | WP-051 | as WP-045 | `slab_amplitude`, a wake amplitude that does not depend on when it is read |
| `witness2d_case.jl` | WP-052: 830 steps, fixed laser energy, a witness bunch | τ ∈ [4, 9], n₀ ∈ [0.04, 0.10], witness centre x_c ∈ [30, 50] (`WITNESS_BOX`) | `witness_gain`, the witness bunch's mean energy gain |
| `born2d_case.jl` | WP-064: 900 steps at λ₀/16, PMLs, a nitrogen K-shell dopant in a short section (WP-059) | a₀ ∈ [1.5, 2.4] (`BORN_BOX` in `born2d.jl`, a scan, no optimizer yet) | `born_energy`, the kinetic energy of the ionization-born electrons ahead of the window's back, with `plasma_peak_energy` and `plasma_fast_weight` as the guard on the plasma's own electrons |
| `trapped2d_case.jl` | WP-065: WP-064's case at a fixed laser energy (a₀ from duration and waist), the dopant at 1 % of the density, `ny` refining y | duration ∈ [4, 9], waist ∈ [6, 14], focus ∈ [64, 110], density ∈ [0.05, 0.10] (`TRAPPED_BOX` over `TCASE.free`; the delay stays at `TCASE.x0[4]`); `gather = CompensatedGather()` passes WP-067's gather to the numerics | `trapped_energy`, the kinetic energy of every electron of the first bucket (a position weight rising from 0 at window x = 12 to 1 at 24) with u_x above about 4 (a smooth weight from 3 to 5), born or the plasma's own |
| `pulse2d_case.jl` | WP-087: WP-065's case at its base density with a `ShapedLaser` of fixed energy, a super-Gaussian near field of the base Gaussian's focal FWHM | focus ∈ [64, 110], Legendre modes P₃, P₄ ∈ [−0.3, 0.3] waves, GDD ∈ [−12, 12], TOD ∈ [−30, 30] (`PULSE_BOX`) | WP-065's `trapped_energy` |
| `pulse3d_case.jl` (with `trapped3d_case.jl`) | WP-087's 3D half: WP-084's 3D case at its base density with a 3D `ShapedLaser` of fixed energy (WP-085), a round super-Gaussian near field of the round base Gaussian's focal FWHM | focus ∈ [64, 110], eleven Zernike modes (OSA j = 3, 5 to 14) ∈ [−0.1, 0.1] waves rms, GDD ∈ [−12, 12], TOD ∈ [−30, 30] (`PULSE3_BOX`) | `trapped_energy3` |

## The drivers

| Driver | Report | What it shows |
|---|---|---|
| `wake2d.jl` | `reports/wp-045-wake-optimization.md` | The first 2D optimum, against a grid scan, finite differences, slices and 1D theory |
| `bo_compare.jl` | `reports/wp-048-bayesian-baseline.md` | Calls and seconds to the optimum for Bayesian optimization and L-BFGS-B |
| `converge2d.jl` | `reports/wp-051-converged-optimization.md` | The optimum under refinement in Δx, Δy and read-out time, with the slab objective |
| `witness2d.jl` | `reports/wp-052-fixed-energy-optimization.md` | An interior optimum at fixed laser energy; at fixed a₀ the duration runs to its bound |
| `witness2d_resolution.jl` | the same report | The fixed-energy optimum at a finer grid (no optimizer) |
| `solvers.jl` | `reports/wp-058-optimization-jl.md` | The WP-045 problem solved by L-BFGS-B and by MadNLP with a limited-memory Hessian |
| `born2d.jl` | `reports/wp-064-born-electron-objective.md` | Whether the born-electron objective can be optimized on: a run that traps, a scan in a₀ with ForwardDiff tangents, finite differences, the cost per run |
| `trapped2d.jl` | `reports/wp-065-trapped-resolution.md` | The trapped-electron objective and its five-parameter gradient from λ₀/16 to λ₀/64, against central differences and along lines in the energy budget and the density; campaigns of independent runs, several at a time, resumable |
| `trapped2d_opt.jl` | `reports/wp-068-first-gpu-optimizations.md` | L-BFGS-B on the trapped-electron objective from four starts at λ₀/32 with the compensated gather in Float32 on the GPU, every point a run with duals recorded with J, its gradient, a₀ and the dark current (resumable), and a check of the optimum at λ₀/48 and against Float64 central differences |
| `pulse2d_opt.jl` | `reports/wp-087-pulse-shape-study.md` | L-BFGS-B over the pulse's shape from five random starts (WP-068's store), every point a run with duals in five directions with its Strehl ratio and launched power, and a Float64 check against central differences |
| `pulse3d_opt.jl` | `reports/wp-087-pulse-shape-study.md` | L-BFGS-B over the 3D pulse's fourteen parameters from four random starts at λ₀/16 in Float32 on the GPU, every point four runs with Dual{4} (the gradient in chunks) recorded with its Strehl ratio and launched share, and a check by central differences along the gradient and random directions |

The reports of WP-045 to WP-052 record runs made with the hand-written projected BFGS that Optimization.jl replaced, so their call counts and paths are that optimizer's.

Tests: `julia +1.13 --project=scripts scripts/optim/test/runtests.jl` (T-760 to T-763 for the layer, T-660 to T-665 and T-680 to T-682, T-830 to T-835 and T-910 to T-914 for the cases, T-970 to T-977 for the WP-068 driver, T-1180 to T-1187 for WP-087's case and driver, T-1230 to T-1237 for its 3D half in `test/pulse3d.jl` (runnable alone), and T-620 to T-623 for the Bayesian optimizer).
