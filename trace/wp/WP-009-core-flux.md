+++
id = "WP-009"
status = "parked"
goals = ["A2", "A4", "A7", "A8"]
depends_on = ["WP-005", "WP-006"]
adds_dependencies = []

[checked]
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
+++
# WP-009: engine core, flux: charge-conserving deposition (Esirkepov)

Parked (A0: contracts are written from working code). The spike WP-010 builds a 1D electrostatic PIC first and tests the contracts below as proposals; this package is re-derived from the spike's write-up, kept or withdrawn, before it is dispatched.

The third particle→mesh transfer of `lib/ParticleMeshCore`, next to `gather!` and the scalar `deposit!`. Contracts: `docs/plan/architecture-draft.md` §0.7 items 7, 8, 14, 17 and 18, and §2 (Kernels). A2 chooses Esirkepov's scheme. Charge-conserving deposition belongs to every electromagnetic PIC, not to LWFA, so it lives in the engine. It knows no Maxwell equations: the electromagnetic module calls it for the current, and Gauss's law to roundoff (A7) is then that module's test, built on this package's continuity test.

## Scope

`deposit_flux!(mode, flux, species, shape, quantity, old, grid, gstate, ws)`: from each particle's old position (`old`, a module-declared extra holding cell and offset before the push) and its current position, deposit onto the faces staggered along each resolved axis the flux that satisfies the discrete continuity equation together with the scalar `deposit!` at both positions. For collapsed axes in 2D3V, the component along the collapsed axis comes from a module-supplied per-particle velocity along it, with Esirkepov's weights for that case. Both modes, `Reference()` and `Fast()`, with the accumulation trait of WP-006.

## Deliverables

1. **`deposit_flux!`** for N = 1, 2, 3, shapes of order 1 and 2 (cubic when WP-006 provides `Shape{3}`): implement the 3D formula and specialize the collapsed axes (§2). Displacements up to one cell per step on every axis; a larger displacement is outside the scheme's contract and is documented, not handled.
2. **Both modes**: `Fast()` with atomics on primitive types, `Reference()` with the sorted segmented reduction over the sort's segment offsets; `Sorted()`-trait types always sorted (§0.7 item 8). The flux shares WP-006's workspace and `DiffCache` scratch conventions (item 17) and is folded with `fold_guards!` like any deposit target.
3. **Test helper**: the staggered discrete divergence D that matches the flux's staggering, in the shared test module, for the continuity tests only. The engine itself has no divergence operator.
4. Tests in `lib/ParticleMeshCore/test/flux.jl`, its table in `tolerances.toml`.

## Non-goals

No Maxwell solve, Yee update, pusher or filter (the electromagnetic module, `lib/ParticleMeshElectromagnetic`). No Gauss's-law test, which needs a field solve and belongs to that module. No current smoothing. No Reactant (A9). Parameters never enter through positions (A3): the flux is linear in the weights.

## Acceptance

Test ids T-110 to T-129.

- T-110: discrete continuity, (ρ(xⁿ⁺¹) − ρ(xⁿ))/Δt + D·J = 0, with ρ from the scalar `deposit!` at both positions, to a proposed `C·eps(T)` bound per step, for N = 1, 2, 3, shapes of order 1 and 2 (cubic when available), displacements up to one cell including cell crossings, every `T` of the A8 matrix.
- T-111: over N steps of `drift!` the continuity residual does not grow (no drift), checked at two N.
- T-112: uniform drift of all particles in a periodic box: the total flux equals Σ wᵢqᵢvᵢ to roundoff, on every axis, including the collapsed axis in 2D3V.
- T-113: `Reference()` and `Fast()` agree per face to a proposed bound (Float32 and Float64); `Reference()` is bitwise identical across reruns and across `JULIA_NUM_THREADS` = 1 and many, for each `T` of the A8 matrix separately (A4: bitwise holds per element type and backend, not across them).
- T-114: with `Dual{Float32,3}` weights, the partials equal the derivative of the flux with respect to the weights, known analytically because the flux is linear in them (A3; the flux of the same particles with unit weight), to a proposed `C·eps(Float32)` bound, the base type of the duals (A7). The value part agrees with the Float32 flux of the duals' value parts, in `Reference()` mode, within a proposed bound, with the measured difference recorded; A4 does not promise bit-for-bit agreement across element types (the spike saw it hold for deposition and fail by 1 ulp in the loader), so the test never requires it. No finite-difference scan here: the derivative is analytic, and A7's measured-floor rule applies to the gradient through particle motion, which is WP-007's and the electromagnetic module's test.
- T-115: a particle at rest deposits zero flux exactly.
- T-116: allocation does not grow with particle count or grid size (§0.7 item 13).

## Depends on

WP-005 (species, `drift!`, sort offsets) and WP-006 (shapes, weights, scalar `deposit!`, `fold_guards!`, the accumulation trait). Runs in parallel with WP-007.

## Budgets

CPU time per particle for both modes recorded without a threshold (A7's throughput budget is on the GPU). Proposed constants for T-110, T-113 and T-114 in `tolerances.toml`, marked "proposed", with the measured maxima at two or more sizes. No gradient-check floor to measure in this package: it has no finite-difference check (A7).
