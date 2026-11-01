+++
id = "WP-022"
status = "proposed"
goals = ["A1", "A2", "A7", "A8", "A9"]
depends_on = ["WP-014", "WP-015", "WP-018"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-022: the electromagnetic module at N = 2 without current: 2D Yee, the field layout, and the particle push in prescribed fields

The first half of `docs/plan/architecture-draft.md` §0.8 item 5, cut so that it can run while the 2D flux (WP-020) and the initial solve (WP-021) are built. Everything here is testable without a current: the vacuum field solver and particles pushed through given fields. WP-023 joins the three into the full step. A spike in the A0 sense.

## Scope

`lib/ParticleMeshElectromagnetic` (src and test), §0.8 item 5 in `docs/plan/architecture-draft.md`, `questions/`.

- N becomes a type parameter of the field layout, the state and the solver; the 1D methods stay, so T-210 to T-221 and the harness tests pass unchanged. Keep the `EMState` field names the harness reads, `step!(st, p, k)` and `initialize!`.
- The 2D Yee layout of §0.8 item 5 (ρ and Ez at (Node, Node); Ex, Jx at (Center, Node); Ey, Jy at (Node, Center); Bx at (Node, Center); By at (Center, Node); Bz at (Center, Center)). All six components exist and vary only in the plane, as in Smilei and PIConGPU. Check the layout against both codes' conventions and record what you read.
- `solve_fields!(::Yee, f, grid::PeriodicGrid{2}, dt)` with the same half-step structure as 1D.
- The Courant condition comes from the field solver: `courant_limit(::Yee, grid)` is Δx at N = 1 and 1/√(1/Δx² + 1/Δy²) at N = 2 (c = 1, A1). The simulation setup rejects every configuration that violates it: the state constructor throws an `ArgumentError` for Δt ≥ `courant_limit(solver, grid)`, at N = 1 in place of the hard-coded Δt < Δx check of #202. The limit is at most min_d Δx_d, so with |v| < c the same check keeps every particle within one cell per step, which `deposit_flux!` needs. The case files use 0.95 of the limit, the standard safety factor; a later solver brings its own `courant_limit`.
- The 2D push: gather all six components at their staggers (engine `gather!` at N = 2), any pusher of `pushers.jl` unchanged, positions advanced through the engine. The current deposition call is left for WP-023: this package's step may take `J = 0` or a prescribed J.
- Bx is a dynamic component at N = 2: the 2D Yee update advances it and fills its guards (the 1D `yee_b_half!` leaves Bx untouched and never fills its guards, harmless only while Bx stays zero).
- Diagnostics that take a wave vector: electrostatic energy Σ(Ex² + Ey²), magnetic and electric energies, mode amplitude as the projection on k. They run as a parallel reduction with a fixed association order, not as the 1D single-work-item kernel (`diagnostics.jl`), whose 3·n sin/cos per step would cost milliseconds on 2D grids of about 10⁶ cells (an estimate from the review of `b714331`).
- The kinetic energy is a deterministic per-particle reduction (for example per-cell sums over the sort's segment bounds, then over cells in order), not a deposition onto the grid: only its total is used, and the review of `b714331` measured the deposition at 17.3 ns per particle against 1.1 ns for a plain particle sum (CPU, 1 thread), about 11 % of the 1D Reference step (unverified: no run manifest or JSON of the review's timings is in the repository). The 1D method may switch as well; the PR states any bits of T-218 that change.

## Acceptance

Tests T-320 to T-329, Float32, Float64 and `Dual{Float32,3}` where they apply:

- T-320 vacuum plane wave at an oblique angle [A2, A7, WP-022]: the measured phase velocity matches the Yee numerical dispersion relation (closed form, stated in the test) to a tolerance set from a resolution pair, not from eps.
- T-321 ∇·B stays at its initial value to C·eps(T) scaled by n and steps; ∇·E is unchanged with J = 0 [A2, A7, WP-022].
- T-322 the discrete Yee energy (with the staggered-time E·E and B·B product of the scheme) is conserved in vacuum to C·eps(T) over the run [A7, WP-022].
- T-323 2D reduces to 1D: a y-invariant field state evolves as the 1D solver to C·eps(T) [A7, WP-022].
- T-324 single particle in prescribed fields [A2, WP-022]: gyration in uniform Bz and E×B drift with uniform Ex, Bz match the closed forms of the pusher used (as T-230 to T-237 do in 1D), and an oblique uniform E accelerates along E in both axes, so a transposed stagger fails.
- T-325 the gather picks each component at its own stagger: a field linear in x and y per component is reproduced at every particle to C·eps(T) [A7, WP-022].
- T-326 `Dual` through one vacuum step and one push, primal bitwise equal to the Float32 run [A8, WP-022].
- T-327 the 2D diagnostics and the kinetic-energy reduction are bitwise across reruns, thread counts and two `workgroupsize` values, and equal a serial sum to C·eps(T)·count [A4, A7, WP-022].
- T-328 `initialize!` and `step!` put the momenta at half steps (A2's half-step centring) [A2, WP-022]: a test particle started through `initialize!` and advanced by `step!` in prescribed uniform fields follows the closed-form orbit (E×B drift, gyration phase) with second-order phase convergence over a Δt pair, at N = 1 and N = 2. It must fail on a mutation that removes the initial half-step kick of `initialize!` (state the check in the PR).
- T-329 the Courant limit is where the scheme goes unstable, and the setup rejects what violates it [A2, A7, WP-022]: a vacuum run seeded at the grid's highest-frequency mode (k = π/Δx at N = 1; the diagonal mode (π/Δx, π/Δy) at N = 2), driven through `solve_fields!` directly at 0.99 and 1.01 of `courant_limit`, stays bounded to C·eps(T) at 0.99 and grows at 1.01 by the per-step amplification factor of the Yee dispersion relation (closed form stated in the test), at N = 1 and N = 2. Using the 1D limit in 2D must fail it. The state constructor accepts Δt = 0.95 of the limit and throws for 1.0 and 1.01 of it, at N = 1 and N = 2.
- T-210 to T-221, T-230 to T-237 and the harness tests pass unchanged. `build.jl --check` passes; CPU CI green.

## Non-goals

No current deposition and no Gauss-law test with particles (WP-023), no initial solve (WP-021), no change of `pushers.jl`, no boundaries other than periodic, no GPU run, no `Project.toml` change, no engine change beyond a bug the port finds (stated in the PR).

## Depends on

WP-014 (the module), WP-015 (the pushers), WP-018 (engine at N = 2). Runs in parallel with WP-020 and WP-021; WP-023 builds on it. Test ids T-320 to T-329.
