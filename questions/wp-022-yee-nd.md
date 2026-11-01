# Choices WP-022 took that its text did not settle

## 1. Where the Courant check lives before an N = 2 state exists

The package says the state constructor throws for Δt ≥ `courant_limit(numerics.solver, grid)` and T-326 tests the constructor, but the N = 2 state is WP-024's. **Assumption:** the check is a public helper, `check_courant(solver, grid, dt)`, which throws the `ArgumentError` naming the limit; the 1D `EMState` constructor calls it, WP-024's N = 2 constructor will call the same helper, and T-326 tests the constructor at N = 1 and the helper at N = 2. The work package text and WP-024's push bullet say so.

## 2. Rounding of the limit at N ≥ 2

`1/√(Σ_d 1/Δ_d²)` computed in floating point can exceed min_d Δ_d by an ulp when one cell size is much larger than the other, and the docstring promises that a Δt below the limit keeps every particle within one cell per axis. **Assumption:** at N ≥ 2 `courant_limit` returns `min(minimum(Δ), 1/√(Σ_d 1/Δ_d²))`, which changes the value only where rounding would break that promise; at N = 1 it returns the cell size itself (not `1/√(1/Δx²)`, which can differ by an ulp).

## 3. The scale of the ∇·B test (T-322)

The package says "C·eps(T) scaled by n and the steps". The identity holds per node, and the rounding of one node's divergence does not grow with the number of cells. **Assumption:** the scale is `eps(T)·max|B|·Σ_d 2/Δ_d·steps` (the size of a divergence's terms times the steps), with no factor of n; with a factor of n the test would pass a defect n times larger. The text of T-322 says so.

## 4. What "one component on a transposed stagger" means for the mutation

The solver has no stagger table: a component's stagger is encoded by which differences its update and its neighbours' updates take. **Assumption:** the mutation makes Ex's update take the forward difference of Bz along y, as if Ex sat at `Center()` rather than `Node()` along y (the y stagger of Ex transposed), and leaves everything else as it is. The other three mutations are those the package names: the sign of Ez's ∂_y Bx term flipped, a full step Δt in place of the first B half step, and `minimum(Δ)` (the 1D limit) as the limit at N = 2.

Affected: `lib/ParticleMeshElectromagnetic/src/yee.jl`, `lib/ParticleMeshElectromagnetic/test/yee_nd.jl`, `trace/wp/WP-022-yee-nd.md`, `trace/wp/WP-024-em-state-push-2d.md`.
