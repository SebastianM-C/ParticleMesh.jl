+++
id = "WP-022"
status = "done"
goals = ["A1", "A2", "A4", "A7", "A8", "A9", "A10"]
depends_on = ["WP-014", "WP-018"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A10 = "30906c6b96b63dab1b6d4eab77c5f25dcb8e1213"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-022: the Yee field solver in N dimensions, its Courant limit, and its verification against the discrete theory

The field half of `docs/plan/architecture-draft.md` §0.8 item 5. The 2D slice needs the fields of all six components advanced on the 2D Yee grid; written once for any N, the same code serves the 3D packages later. This package delivers that solver, the stability limit that comes with it, and tests that hold it to the exact discrete theory of the scheme rather than to loose agreement (A7). Particles, the 2D state and the diagnostics are WP-024; the current from particles is WP-020 and WP-023. A spike in the A0 sense: the choices below are starting points, and the code may show better ones.

## Scope

`lib/ParticleMeshElectromagnetic` (src and test), `scripts/` and `reports/wp-022-yee-nd.md` for the convergence measurement, §0.8 item 5 and the module's paragraph of §0.7 in `docs/plan/architecture-draft.md`, `questions/`.

- **The layout as a rule in N.** With ρ on the primal nodes, component c of E (and of J) sits at `Center()` along axis c and at `Node()` along every other resolved axis; component c of B sits at `Node()` along axis c and at `Center()` along the others; a component of a collapsed axis (c > N) has no axis of its own, so E_c and J_c sit at `Node()` and B_c at `Center()` on every axis. At N = 1 this is today's `FIELD_LAYOUT`; at N = 2 it is the layout listed in §0.8 item 5; at N = 3 it is the standard Yee cell. Check the N = 2 layout against the Smilei and PIConGPU sources and cite what you read (what you could not read is marked unverified). `em_fields(backend, T, grid)` allocates the six fields and three currents for any `PeriodicGrid{N}`.
- **`solve_fields!(::Yee, f, grid::PeriodicGrid{N}, dt)`** with the half-step structure of 1D (B half step, E full step with J, B half step), the curl written once over the N resolved axes: a derivative from `Node()` to `Center()` is the forward difference, from `Center()` to `Node()` the backward one, and a derivative along a collapsed axis is absent (dropped statically, not computed as a difference of equal values). Bx is a dynamic component for N ≥ 2, and every component that the solve changes has its guards filled. The code is generic in N where that costs nothing; N = 1 and N = 2 are claimed and tested (the engine's fields and guards stop at N = 2). The generic kernels replace the 1D ones only if they reproduce the 1D bits: record a digest of the 1D vacuum and edge-case runs (T-210, T-213) before the change and compare after; otherwise the 1D methods stay. The digests agree, so the generic kernels are the only ones.
- **The Courant limit belongs to the solver.** `courant_limit(::Yee, grid)` = 1/√(Σ_d 1/Δ_d²) (c = 1, A1), which is Δx at N = 1. `check_courant(solver, grid, dt)` throws an `ArgumentError` naming the limit for Δt ≥ `courant_limit(solver, grid)`; the state constructor calls it with `numerics.solver`, replacing the hard-coded Δt < Δx check (#202), and WP-024's N = 2 constructor calls the same helper. At N ≥ 2 the computed limit is rounded down to min_d Δ_d where rounding would put it above. Since the limit is at most min_d Δ_d and |v| < c, the same check keeps every particle within one cell per axis per step, which `deposit_flux!` needs; state that in the docstring. A later solver brings its own `courant_limit`. Case files use 0.95 of the limit; check that the committed 2D cases of WP-019 satisfy it, and if one does not, write a question rather than edit a case Smilei ran.
- Every kernel within §0.4, no float literals, `workgroupsize = nothing` on every launch.

## Acceptance

Tests T-320 to T-329, Float32, Float64 and `Dual{Float32,3}` where they apply, at N = 1 and N = 2 unless stated, on non-square grids with Δx ≠ Δy at N = 2. Each closed form is stated in the test, derived from the one-step map of the scheme.

- T-320 discrete dispersion [A2, A7, WP-022]: an oblique discrete eigenmode, in each of the two polarizations at N = 2 (Ex, Ey, Bz and Bx, By, Ez), evolves by the exact discrete relation sin(ωΔt/2)/Δt = √(Σ_d sin²(k_dΔ_d/2)/Δ_d²) to C·eps(T) scaled by the step count, phase and amplitude. This compares the code with its own scheme, so the tolerance is roundoff.
- T-321 convergence to Maxwell [A2, A7, WP-022]: at a fixed Courant number of 0.95, the phase-velocity error against c falls as Δ² over three resolutions, along an axis and along the diagonal; the observed order and its tolerance are measured and proposed. The anisotropy between the two directions is recorded in the report.
- T-322 divergence identities [A2, A7, WP-022]: the discrete ∇·B (forward differences, on its own stagger) is unchanged by any number of steps to C·eps(T) scaled by max|B|·Σ_d 2/Δ_d and the steps (a per-node identity, so no factor of n); with a prescribed current J, the change of the backward-difference ∇·E over one step equals −Δt·∇·J to C·eps(T). This is the identity WP-023's Gauss law with particles rests on.
- T-323 discrete energy [A7, WP-022]: in vacuum the scheme's conserved energy (E·E at integer times and the B·B product of the two neighbouring half steps, as T-210 uses at N = 1) is constant to C·eps(T) scaled by the steps; with a prescribed J, its change per step is the scheme's discrete Poynting term in J and E, to C·eps(T).
- T-324 reductions and symmetry [A7, WP-022]: a 2D state invariant along y evolves as the 1D solver to C·eps(T) (whether it is bitwise is recorded); exchanging the two axes (transposed arrays, components exchanged with the signs of a reflection, Δx and Δy swapped) maps a run onto the exchanged run to C·eps(T). Together they catch a transposed stagger, a wrong sign and a cell size taken from the wrong axis.
- T-325 the Courant limit is where the scheme becomes unstable [A2, A7, WP-022]: a vacuum run seeded at the highest mode of the grid (k = π/Δx at N = 1, the diagonal (π/Δx, π/Δy) at N = 2), advanced through `solve_fields!`, keeps its energy to C·eps(T) at 0.99 of `courant_limit` and grows at 1.01 of it by the closed-form amplification factor per step, (S + √(S² − 1))² with S the Courant-scaled symbol, to a measured tolerance. Using the 1D limit in 2D must fail this test.
- T-326 the setup rejects what violates the limit [A2, A7, WP-022]: the state constructor (N = 1) and `check_courant` (N = 2, until WP-024 brings the N = 2 state) accept Δt = 0.95 of the limit and throw for 1.0 and 1.01 of it, with the limit in the message.
- T-327 bitwise across reruns, 1 and 4 threads and two `workgroupsize` values, at N = 1 and N = 2 [A4, WP-022].
- T-328 `Dual` [A8, WP-022]: the solve is linear, so the partials of a parameter in the initial fields evolve as the solver applied to the partials, to C·eps(T); the primal agrees with the Float32 run within C·eps(T) (whether it is bitwise is recorded, not asserted, A4).
- T-329 guards and allocation [A7, WP-022]: after `solve_fields!` every component's guards are its periodic images, Bx included at N = 2; allocation per step does not grow with the grid size (the T-220 rule, at two sizes after warm-up).
- The PR states four mutations and the tests each one fails: one component on a transposed stagger, the sign of one curl term flipped, a full step in place of a B half step, and the 1D limit used in 2D.
- `tolerances.toml` entries `status = "proposed"`. T-210 to T-221, T-230 to T-237 and the harness tests pass unchanged. `build.jl --check` passes; CPU CI green.

## Non-goals

No particles, state or diagnostics at N = 2 (WP-024); no current deposition (WP-020, WP-023); no boundaries other than periodic and no absorbing layers; N = 3 not claimed; no GPU run; no `Project.toml` change; no engine change beyond a bug the work finds (stated in the PR).

## Depends on

WP-014 (the module) and WP-018 (the engine at N = 2). Runs in parallel with WP-020; WP-024 and WP-023 build on it. Test ids T-320 to T-329.
