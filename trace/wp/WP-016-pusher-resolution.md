+++
id = "WP-016"
status = "proposed"
goals = ["A1", "A2", "A7", "A8"]
depends_on = ["WP-015"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
+++
# WP-016: the quadratic-unitary pusher and a resolution rule in a₀·ω₀Δt for electrons in the laser

How large is each pusher's error for electrons inside the laser at PIC resolution, and what Δt does a planned run need at its a₀?

Gordon and Hafizi push an electron through a plane wave at a₀ = 10 with constant proper-time steps, about 63 per laser period. They find Boris and Vay not viable and Higuera–Cary visibly off. Their quadratic-unitary pusher follows the exact orbit.

A fixed lab Δt resolves the figure-8 more finely on average, but not where it matters. At the zeros of the vector potential γ ≈ 1 and the field peaks, so the phase step is the full Δt and the per-step kick is about a₀·ω₀Δt, which is 2 at a₀ = 10 and ω₀Δt = 0.2. A quick, unreviewed check during planning found the Boris error in γ − u_z growing with a₀·ω₀Δt and reaching tens of percent at a₀ = 10 and ω₀Δt = 0.2.

This package does three things:

1. It adds the quadratic-unitary pusher to WP-015's set.
2. It scans a₀ and Δt for all five pushers.
3. It turns the result into a resolution rule that a planned run can be checked against, next to the CFL condition.

`docs/plan/pushers-literature.md` has the sources and the argument in full.

## Scope

**The pusher.** `QuadraticUnitary` is a fifth subtype of the module's pusher type, a device function with the same signature as the others, and an OrdinaryDiffEq algorithm through WP-015's extension. It implements Gordon and Hafizi's Eq. 18 on the four-velocity (γ, u).

- Their 2×2 spinor form can be expanded into real arithmetic, with the common real denominator factored out as they suggest.
- The proper-time step comes from the lab Δt through their second-order Eq. 22, the same inversion they use for their own lab-frame runs.
- The update needs no square root and no branch in the fields.

Its properties, from their Secs. 3 and 4:

- the Minkowski norm is preserved exactly;
- energy is preserved for E = 0;
- in a plane wave the update equals the exact one, so k·u is preserved.

Their exact-unitary and SO(3,1) pushers are not built; they report the quadratic truncation gives results very close to them.

**The scan.** The figure-8 orbit, with WP-015's conventions, runs for all five pushers:

- a₀ ∈ {1, 2, 5, 10};
- ω₀Δt from 0.4 down to 0.0125 at a fixed lab Δt;
- the reference is the closed form, with `Vern9` through the SciML interface where the closed form needs inverting.

The scan also runs once at Gordon and Hafizi's own resolution, a constant phase step Δη = 0.1 at a₀ = 10. This is a diagnostic outside any PIC CFL limit, so the comparison with their verdict is direct.

**Float32 rounding at weak rotation (#203).** Vay and the exact-rotation Boris drift systematically in |u| in Float32 when the rotation per step is small, t² = (qBΔt/2mγ)² ≲ eps(Float32), the weak-field, high-γ regime of LWFA. #203 gives the reproducer (|u| = 10⁴, B = 50, qm_h = −0.05, 10⁴ steps: Vay 3900 eps, exact rotation 1300, Boris and Higuera–Cary 18) and, for Vay, a rewrite of the last line as `u′ + s·((u′·t)t + u′×t − t²u′)` that cuts it to 51 eps; the exact-rotation cause is open. The package fixes both, or states why one cannot be fixed without changing the algorithm, and checks `QuadraticUnitary` in the same regime.

## Deliverables

1. **Source**: `QuadraticUnitary` in `pushers.jl` and in the extension, generic in `T` and within the §0.4 rules. Its docstring cites the paper and states the properties above.
2. **Tests**: T-240 to T-245.
3. **Scan**: `scripts/pushers/resolution/`.
   - Plots of four errors per pusher against a₀·ω₀Δt, and the same against lab Δt:
     - γ − u_z;
     - u_x + q A_x;
     - the drift velocity against its exact value a₀²/(4 + a₀²);
     - the orbit position against the reference after one period.
   - The Δη = 0.1 diagnostic at a₀ = 10.
   - CairoMakie PNGs with JSON sidecars and run manifests (F3).
4. **Report**: a resolution section in `reports/pushers.md`.
   - For each pusher, the largest a₀·ω₀Δt that keeps each figure-8 error below 10⁻³ and below 10⁻² relative.
   - That threshold applied to the CFL Δt of the planned LWFA runs at their a₀.
   - Whether Gordon and Hafizi's verdict holds at their resolution and at ours.
   - A recommendation for the default pusher, and for any Δt rule, stated as a proposal against A2.

## Non-goals

- No exact-unitary or SO(3,1) pusher.
- No focused or finite-duration pulse: the plane wave is the exact case, and a pulse is a later check against a high-order reference.
- No radiation reaction, no change of the default pusher, no GPU run, no new dependency.

## Acceptance

Test ids T-240 to T-249.

- **T-240 the slot** [A2, WP-016]: the short Landau run of T-230 completes with `QuadraticUnitary`.
- **T-241 invariants** [A2, A7, A8, WP-016]: for `QuadraticUnitary`, three quantities are preserved to C·eps(T)·steps:
  - γ² − |u|² = 1 in random uniform fields;
  - energy for E = 0;
  - γ − u_z in the plane wave.

  Its uniform-B gyrophase is checked to roundoff against the closed form the test header derives from Eq. 18 and Eq. 22 at E = 0.
- **T-242 crossed fields and uniform E** [A2, A7, WP-016]: in T-232's and T-233's setups, the u error and the drift error of `QuadraticUnitary` are recorded.
- **T-243 the scan** [A1, A2, A7, WP-016]: for all five pushers, a₀ ∈ {1, 2, 5, 10} and ω₀Δt ∈ {0.2, 0.1, 0.05, 0.025}:
  - every error decreases with an observed order of 2 within a measured tolerance where a₀·ω₀Δt ≤ 0.5;
  - every error is recorded against a₀·ω₀Δt, including the cases outside the asymptotic range.
- **T-244 element types** [A8, WP-016]: T-241 passes for Float32, Float64 and `Dual{Float32,3}`.
- **T-245 Float32 at weak rotation** [A2, A7, A8, WP-016]: for all five pushers in Float32, in uniform B with t² below eps(Float32) (#203's reproducer and |u| = 10 with B = 0.05), the drift of |u| over 10⁴ steps stays within C·eps(T)·√steps, the scale of unbiased rounding, with C measured and proposed; a linear drift fails it. T-230 to T-237 pass unchanged, and any change of their recorded values is stated in the PR.
- `build.jl --check` passes and CPU CI is green.

The implementing PR closes #203.

## Depends on

WP-015: the pusher set, the SciML extension, the figure-8 conventions and the report.

## Budgets

A7:

- The invariants of T-241 hold to roundoff, with C measured and proposed.
- Convergence orders lie within a measured tolerance of 2.
- The resolution thresholds are measured and proposed, not taken from the literature.

Dependencies (F2): none beyond WP-015's.
