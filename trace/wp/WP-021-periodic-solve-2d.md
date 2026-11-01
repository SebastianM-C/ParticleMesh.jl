+++
id = "WP-021"
status = "done"
goals = ["A4", "A7", "A8", "A9", "A10"]
depends_on = ["WP-018"]
adds_dependencies = []

[checked]
A10 = "30906c6b96b63dab1b6d4eab77c5f25dcb8e1213"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-021: a periodic inverse of the discrete Laplacian at N = 2, for the initial field

`docs/plan/architecture-draft.md` §0.7 contract 14, written from this package's code. The 1D `initialize!` takes Ex from `cumsum_periodic!`; an oblique 2D perturbation needs a genuine 2D solve for the initial electrostatic field. A spike in the A0 sense. The choices taken with the owner (the Dual comparison, the staggering, the sign, the API) are in `questions/wp-021-periodic-solve.md`.

## Scope

`lib/ParticleMeshCore` (src and test), §0.7 and §0.8 in `docs/plan/architecture-draft.md`, `questions/`, a timing script in `scripts/engine2d/` and its report.

- Physics-free engine functions that, given a periodic source s on `PeriodicGrid{2}` (and N = 1, where it must agree with the existing route), return the zero-mean φ with the discrete Laplacian of φ equal to s − mean s, and its gradient. s and φ sit on ρ's nodes; gradient component d is the forward difference G_d[j] = (φ[j + e_d] − φ[j])/Δ_d stored at index j, the face half a cell to the right (Center on axis d: for N = 2 the §0.8 item 5 layout of Ex and Ey, for N = 1 that of Ex in the 1D module), and its backward-difference divergence Σ_d (G_d[j] − G_d[j − e_d])/Δ_d equals s − mean s. Sign: ∇²φ = s − mean s and G = +∇φ, so the module takes E = G for div E = ρ. The engine names no equation (A9, A10); the module decides that s is ρ.
- Per-axis DFTs written as KernelAbstractions kernels with a fixed association order (one work item per output entry, summing its line serially in increasing order), the Yee-consistent symbol Σ_d (2 sin(k_d Δ_d/2)/Δ_d)², so that the backward-difference divergence of the gradient equals s − mean s to roundoff. O(n) work per mode is acceptable at CPU test sizes; the report says up to which size, and what a production solver would need (it is only called once, at initialization).
- No new dependency: FFTW does not take `Dual`, and Krylov solvers give no fixed order, so neither is on the Reference path.
- Every kernel within §0.4, no float literals, `workgroupsize = nothing`.

## Acceptance

Tests T-310 to T-316, Float32, Float64 and `Dual{Float32,3}`:

- T-310 the backward-difference divergence of the gradient equals s − mean s at every node, and mean φ is zero, to C·eps(T) scaled by (Σ_d n_d)·(max|s| + ‖L‖·max|φ|) with ‖L‖ = Σ_d 4/Δ_d², a scale under which the measured ratio does not grow with the grid size [A7, WP-021].
- T-311 a single oblique Fourier mode (k_x ≠ 0, k_y ≠ 0, a non-square grid with Δx ≠ Δy) is reproduced against its closed form φ = −s/K² with the Yee-consistent symbol, and its gradient, to C·eps(T) [A7, WP-021].
- T-312 bitwise across reruns, thread counts (1 and 4) and launch shapes (`workgroupsize` `nothing`, 7 and 64) [A4, WP-021].
- T-313 at N = 1 the gradient agrees with the `cumsum_periodic!` route to C·eps(T) [A7, WP-021].
- T-314 `Dual`: the partials of parameters in s are the Float32 solve of the partials of s to C·eps(T) (the solve is linear in s); the primal agrees with the Float32 run to C·eps(T), and whether it is bitwise is recorded, not asserted, since A4 does not make bitwise agreement across element types a contract [A4, A8, WP-021].
- T-315 allocation per call does not grow with the grid size, measured at two sizes after warm-up as T-268 does [A7, WP-021].
- T-316 a workspace of another size or element type, a grid without guards and a gradient written into φ are refused [A7, WP-021].
- `tolerances.toml` entries `status = "proposed"`; existing tests pass; `build.jl --check` passes; CPU CI green.

## Non-goals

No electromagnetic code, no change to `initialize!` (WP-023 wires it in), no GPU run, no `Project.toml` change.

## Depends on

WP-018. Runs in parallel with WP-020 and WP-022. Test ids T-310 to T-319.
