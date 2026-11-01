+++
id = "WP-024"
status = "proposed"
goals = ["A1", "A2", "A3", "A4", "A7", "A8", "A10"]
depends_on = ["WP-015", "WP-018", "WP-021", "WP-022"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A10 = "30906c6b96b63dab1b6d4eab77c5f25dcb8e1213"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
+++
# WP-024: the electromagnetic module's problem, state, push and diagnostics at N = 2, in prescribed fields

The particle half of `docs/plan/architecture-draft.md` §0.8 item 5, between WP-022's field solver and WP-023's full step. Everything here is testable without a current from particles: the 2D problem description and state, the initial field from WP-021's solve, particles pushed through given fields, and the diagnostics. A spike in the A0 sense.

## Scope

`lib/ParticleMeshElectromagnetic` (src and test), §0.8 item 5 and the module's paragraph of §0.7 in `docs/plan/architecture-draft.md`, `questions/`.

- **N becomes a type parameter of the problem and the state**, and the 1D methods stay, so T-210 to T-221 and WP-008's harness tests pass unchanged; keep the `EMState` field names the harness reads (`K` included), `step!(st, p, k)` and `initialize!`. `EMProblem` at N = 2 carries cells and box lengths per axis and an integer wave vector per species for the cosine profile, so the loader's weights are `density·(1 + amplitude·cos(k·x))/prod(ppc)` and parameters enter through the weights only (A3).
- **Uniform background fields in the problem description**: `E0` and `B0` (default zero, A1 units), added to the fields at t = 0 by `initialize!`. They are physics (a magnetized plasma needs B0) and are what the prescribed-field tests use; test particles are a species of zero density (zero weight, so no current and no charge) on the known quiet-start lattice.
- **`initialize!` at N = 2**: Ex and Ey from the deposited charge with WP-021's solve (`inverse_laplacian_periodic!` of ρ, then `gradient_periodic!`, E = G), plus `E0` and `B0`, then the half-step kick back, as in 1D.
- **The 2D push**: all six components gathered at their staggers (engine `gather!` at N = 2), any pusher of `pushers.jl` unchanged, positions advanced per axis through the engine; WP-022's Courant check guards the state.
- **No silent missing current.** Until WP-023 adds `deposit_flux!` at N = 2, the N = 2 step deposits no current, and it does so only when the numerics say so: `EMNumerics` at N = 2 takes an explicit current choice by type (`NoCurrent()` here; WP-023 adds the Esirkepov one), and the N = 2 state refuses numerics without one. A static configuration branch, no branch on device data (§0.4).
- **Diagnostics at N = 2** as parallel reductions with a fixed association order (not the 1D single-work-item kernel, whose 3·n sin and cos per step would cost milliseconds on grids of about 10⁶ cells, an estimate from the review of `b714331`): the six field energies; for a wave vector k, the Fourier amplitude of each of Ex, Ey and Bz on its own stagger and the longitudinal projection k̂·E. The phases come from per-axis tables written once in the state (O(Σ_d n_d) trigonometric evaluations, not O(prod n) per step). WP-023's oblique Landau case needs the Ex and Ey amplitudes separately.
- **The kinetic energy at N = 2** as a deterministic reduction over particle slots (fixed chunks of slots summed serially, then the chunks in order), which works in both deposition modes because the slot order does not depend on the threads; not a deposition onto the grid, since only its total is used (the review of `b714331` measured the deposition at 17.3 ns per particle against 1.1 ns for a plain particle sum, CPU, 1 thread, about 11 % of the 1D Reference step; unverified: no run manifest or JSON of those timings is in the repository). `K` stays the name of its buffer, at a fixed size. The 1D method stays as it is, so T-218's bits do not change.

## Acceptance

Tests T-350 to T-359, Float32, Float64 and `Dual{Float32,3}` where they apply:

- T-350 a single particle in prescribed uniform fields [A2, WP-024]: gyration in uniform Bz and the E×B drift in uniform Ex and Bz match the closed forms of the pusher used (as T-230 to T-237 do in 1D); an oblique uniform E accelerates along E in both axes, so exchanged components fail.
- T-351 the gather picks each component at its own stagger [A7, WP-024]: a field linear in x and y per component (as T-264 builds one) is reproduced at every particle to C·eps(T); a component on a transposed stagger fails.
- T-352 half-step centring [A2, WP-024]: a test particle started through `initialize!` and advanced by `step!` in prescribed uniform fields follows the closed-form orbit (E×B drift, gyration phase) with second-order phase convergence over a Δt pair, at N = 1 and N = 2; removing the initial half-step kick of `initialize!` fails it (the PR states the check).
- T-353 the initial field at N = 2 [A2, A7, WP-024]: after `initialize!` of a perturbed plasma, the backward-difference divergence of E equals ρ − mean ρ to C·eps(T) scaled as T-310, and `E0`, `B0` are added exactly.
- T-354 the diagnostics against closed forms [A7, WP-024]: the field energy of a prescribed mode, the Fourier amplitude and phase of an oblique cosine on each stagger, and the longitudinal projection, to C·eps(T); a y-invariant state gives the 1D values.
- T-355 the diagnostics and the kinetic-energy reduction are bitwise across reruns, 1 and 4 threads and two `workgroupsize` values, and equal a serial sum to C·eps(T)·count [A4, A7, WP-024].
- T-356 `Dual` through `initialize!`, one step and one push: the partials of a density parameter propagate, and the primal agrees with the Float32 run within C·eps(T) (whether it is bitwise is recorded, not asserted, A4) [A8, WP-024].
- T-357 allocation per step at N = 2 does not grow with the particle count or the grid size (the T-220 rule) [A7, WP-024].
- T-358 input checks [A7, WP-024]: an N = 2 state without a current choice, a wave vector that is not an integer mode of the box, and a problem whose dimension differs from the numerics' are refused with an `ArgumentError`.
- `tolerances.toml` entries `status = "proposed"`. T-210 to T-221, T-230 to T-237, T-320 to T-329 and the harness tests pass unchanged. `build.jl --check` passes; CPU CI green.

## Non-goals

No current deposition and no Gauss-law test with moving particles (WP-023); no change of `pushers.jl` or of the 1D bits; no boundaries other than periodic; no GPU run; no `Project.toml` change; no engine change beyond a bug the work finds (stated in the PR).

## Depends on

WP-015 (the pushers), WP-018 (the engine at N = 2), WP-021 (the initial solve), WP-022 (the field layout, the solver and the Courant check). WP-023 builds on it. Test ids T-350 to T-359.
