+++
id = "WP-021"
status = "proposed"
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

`docs/plan/architecture-draft.md` §0.8 item 3. The 1D `initialize!` takes Ex from `cumsum_periodic!`; an oblique 2D perturbation needs a genuine 2D solve for the initial electrostatic field. A spike in the A0 sense: start from the §0.8 choice and change it if the code shows a better one.

## Scope

`lib/ParticleMeshCore` (src and test), §0.7 and §0.8 item 3 in `docs/plan/architecture-draft.md`, `questions/`.

- A physics-free engine function that, given a periodic source s on `PeriodicGrid{2}` (and N = 1, where it must agree with the existing route), returns φ with the discrete Laplacian of φ equal to s − mean s, and the backward-difference gradient that the module turns into E. The engine names no equation (A9, A10); the module decides that s is ρ.
- Per-axis DFTs written as KernelAbstractions kernels with a fixed association order, the Yee-consistent symbol Σ_d (2 sin(k_d Δ_d/2)/Δ_d)², so that the backward-difference divergence of the resulting E equals ρ − mean ρ to roundoff. O(n) work per mode is acceptable at CPU test sizes; say in the report up to which size, and note what a production solver would need (it is only called once, at initialization).
- No new dependency: FFTW does not take `Dual`, and Krylov solvers give no fixed order, so neither is on the Reference path.
- Every kernel within §0.4, no float literals, `workgroupsize = nothing`.

## Acceptance

Tests T-310 to T-314, Float32, Float64 and `Dual{Float32,3}`:

- T-310 the discrete divergence of the solved E equals s − mean s at every node to C·eps(T) scaled by n·max|s| [A7, WP-021].
- T-311 a single oblique Fourier mode is reproduced against its closed form with the Yee-consistent symbol to C·eps(T) [A7, WP-021].
- T-312 bitwise across reruns, thread counts and two `workgroupsize` values [A4, WP-021].
- T-313 at N = 1 the result agrees with the `cumsum_periodic!` route to C·eps(T) [A7, WP-021].
- T-314 `Dual`: partials of a parameter in s propagate, primal bitwise equal to the Float32 run [A8, WP-021].
- `tolerances.toml` entries `status = "proposed"`; existing tests pass; `build.jl --check` passes; CPU CI green.

## Non-goals

No electromagnetic code, no change to `initialize!` (WP-023 wires it in), no GPU run, no `Project.toml` change.

## Depends on

WP-018. Runs in parallel with WP-020 and WP-022. Test ids T-310 to T-319.
