+++
id = "WP-015"
status = "proposed"
goals = ["A1", "A2", "A7", "A8", "A10"]
depends_on = ["WP-014"]
adds_dependencies = ["OrdinaryDiffEq"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A10 = "66d01e20241b6a7137b53d5c2346b113ac0d3e71"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
+++
# WP-015: swappable relativistic pushers: Boris, Vay, Higuera–Cary, exact-rotation Boris and quadratic-unitary against exact orbits, with a resolution rule in a₀·ω₀Δt

A2 fixes Boris as the baseline push and asks that each choice be documented with its validation test. The explicit relativistic pushers differ in exactly the regime this project lives in, so the differences are measured here instead of assumed. `docs/plan/pushers-literature.md` collects the claims and their sources. In short:

- **Boris** preserves phase-space volume, but gets the E×B velocity wrong once the drift is relativistic.
- **Vay** fixes the drift but does not preserve volume, and Higuera and Cary show it develops energy errors near resonant orbits.
- **Higuera–Cary** does both, for one more square root.
- **Exact-rotation Boris** (Zenitani and Umeda) removes the magnetic gyrophase error, at the price of an O((ΩΔt)²) error in the E×B drift.
- **Quadratic-unitary** (Gordon and Hafizi) advances the four-velocity with a truncated Lorentz transformation. It keeps the Minkowski norm exactly, keeps energy for E = 0, and in a plane wave equals the exact update, so it preserves k·u. Gordon and Hafizi blame Boris's trouble in strong laser fields on splitting the electric boost from the magnetic rotation. An exact rotation does not remove that error; their pusher does.

The question this package must answer is how large the error of each pusher is for electrons inside the laser at PIC resolution. Gordon and Hafizi find Boris and Vay not viable at a₀ = 10 with constant proper-time steps, about 63 per laser period. A fixed lab Δt resolves the figure-8 more finely on average, but not where it matters. At the zeros of the vector potential γ ≈ 1 and the field peaks, so the phase step there is the full Δt and the per-step kick is about a₀·ω₀Δt. That kick is 2 at a₀ = 10 and ω₀Δt = 0.2. A quick, unreviewed check during planning found the Boris error in γ − u_z growing with a₀·ω₀Δt and reaching tens of percent at a₀ = 10 and ω₀Δt = 0.2.

This package therefore scans a₀ and Δt, and reports each pusher's error as a function of a₀·ω₀Δt. That result is a resolution rule a planned run can be checked against, next to the CFL condition.

No long-time energy bound is proved for any of these pushers in our fields. In a time-dependent laser field the exact energy is not conserved at all. So energy behaviour is measured in static fields, and the laser orbits are checked against their invariants.

A2's AD-smoothness rule decides ties. Switching the default away from Boris would be a proposal to the owner (an edit of A2), not part of this package.

TestParticle.jl is not used as a reference (checked against v0.26.0 on 2026-10-31). Its native Boris pushers, in the `BorisPushers` subpackage, are nonrelativistic and require KernelAbstractions 0.9. Its relativistic path is the Lorentz-force ODE in (x, γv) through OrdinaryDiffEq, which `scripts/pushers/` reproduces directly. Its test files are a useful list of orbit cases.

## Scope

`Vay`, `HigueraCary`, `BorisExactRotation` and `QuadraticUnitary` join WP-014's `Boris` as subtypes of the module's pusher type, in `lib/ParticleMeshElectromagnetic/src/pushers.jl`. Each is a device function with the same signature as `Boris`: it takes the momentum `u`, the gathered E and B, and q Δt/(2m) in A1's normalized units. The module step chooses the pusher by type only (A10), so a run swaps pushers by changing one field of its numerics and nothing in `step.jl`.

Implementation notes:

- **Deriving updates.** Derive each update from its defining equation, not from a printed matrix form; the sign and factor-of-2 conventions differ between papers. For Boris this is u⁺ − u⁻ = (u⁺ + u⁻) × t/γ, with t = qBΔt/(2m).
- **Higuera–Cary.** Its γ is their Eq. 20.
- **`BorisExactRotation`.**
  - Write cos θ and sin θ/|B| as functions of θ², with a series near zero. |B| = √(B·B) has an infinite derivative at B = 0, and a threshold on |B| is a branch, so neither is safe for duals.
  - Its docstring notes that exact-rotation schemes become unstable at ΩΔt/γ = kπ.
- **`QuadraticUnitary`.**
  - It implements Gordon and Hafizi's Eq. 18 on the four-velocity (γ, u). Their 2×2 spinor form can be expanded into real arithmetic, with the common real denominator factored out as they suggest.
  - The proper-time step comes from the lab Δt through their second-order Eq. 22, the inversion they use for their own lab-frame runs.
  - It needs no square root in the update and no branch in the fields.

The tests push single particles in prescribed fields on the CPU, with no mesh. The convergence study against a high-order ODE solution lives in `scripts/pushers/`.

The figure-8 orbit of an electron in a linearly polarized plane wave (Sarachik and Schappert, Phys. Rev. D 1, 2738 (1970)) is the main relativistic test. It has a closed form in the phase η = t − z for a wave travelling in +z with E_x = B_y.

- **Invariants.** Two quantities are conserved along the exact orbit: the light-front momentum γ − u_z, and the transverse canonical momentum u_x + q A_x(η), with q the signed charge in units of e and A in A1's units.
- **Closure.** In the frame of the average drift, the orbit closes after every laser period.
- **Initial condition.** The closed form assumes an electron at rest before an adiabatic switch-on. A test that starts at rest inside a constant-amplitude wave therefore starts at a zero of A_x.
- **Half steps.** u lives at integer steps and x at half steps. A is evaluated at z^n = (z^{n−½} + z^{n+½})/2; using z^{n+½} instead adds an O(Δt) offset that halves the observed order.
- **Expected behaviour.** ElectronDynamicsModels.jl's figure-8 test integrates the same orbit with a ninth-order solver at tolerance 1e-9. A second-order pusher will not close the orbit at a practical Δt, so the tests assert the convergence order and record the closure error, never closure itself.

The derivation, with these conventions, goes in the test header.

## Deliverables

1. **Source**: the four pushers in `pushers.jl`, generic in `T`, within the §0.4 rules and branch-free in the fields. Each docstring cites its paper and states which properties the pusher has (volume or Minkowski-norm preservation, the E×B velocity, energy for E = 0, gyrophase).
2. **Tests** T-230 to T-237 in the module's test suite. They use only its existing test dependencies (ForwardDiff, Test).
3. **Convergence study** in `scripts/pushers/`.
   - The reference integrates the same orbits with OrdinaryDiffEq (`Vern9`, absolute and relative tolerance near 1e-12, Float64).
   - Cases:
     - E parallel to B, accelerating to γ ~ 100;
     - crossed fields at |E|/|B| ∈ {0.1, 0.9, 0.99};
     - the figure-8 at a₀ ∈ {1, 2, 5, 10} and ω₀Δt from 0.4 down to 0.0125.
   - Plots:
     - the figure-8 errors (γ − u_z, u_x + q A_x, drift velocity, and orbit position against the reference) per pusher against a₀·ω₀Δt;
     - the same figure-8 errors against lab Δt;
     - the a₀ = 10 figure-8 at Gordon and Hafizi's resolution, a phase step Δη = 0.1. This is a diagnostic outside any PIC CFL limit, to compare with their verdict;
     - the long-run energy error per pusher, with classic RK4 as the baseline that does not preserve volume.
   - Each plot is a CairoMakie PNG with a JSON sidecar and a run manifest (F3).
   - OrdinaryDiffEq is added to the `scripts/` environment only, never to a `lib/` package or the test environment (F2).
4. **Report**: `reports/pushers.md`.
   - The measured orders, error constants, drift errors and energy behaviour per pusher.
   - The resolution rule: for each pusher, the largest a₀·ω₀Δt that keeps the figure-8 errors below 10⁻³ and below 10⁻² relative. This is applied to the CFL Δt of the planned LWFA runs at their a₀.
   - Whether Gordon and Hafizi's verdict holds at their resolution and at ours.
   - A recommendation, stated as a proposal against A2.

## Non-goals

- No implicit, adaptive, multi-evaluation or high-order pushers in the loop (`pushers-literature.md` says why). High-order frozen-field test particles stay on the SciML path.
- No exact-unitary or SO(3,1) pusher: Gordon and Hafizi report the quadratic truncation gives results very close to them.
- No radiation reaction, no change of the default pusher, no TestParticle.jl dependency, no GPU run, no field solve.

## Acceptance

Test ids T-230 to T-239.

- **T-230 the pusher slot** [A2, A10, WP-015]: a short Landau run of WP-014's step completes with each of the five pushers, changing only the pusher type. With `Boris` it is bitwise identical to WP-014's run.
- **T-231 uniform B** [A2, A7, A8, WP-015]: for all five pushers, |u| is conserved to C·eps(T)·steps. With u ⊥ B, the per-step gyrophase matches to roundoff:
  - Boris and Vay: 2·atan(θ/2);
  - `BorisExactRotation`: θ = ΩΔt/γ;
  - Higuera–Cary: 2·atan(β/γ_new), with γ_new from their Eq. 20;
  - `QuadraticUnitary`: its own closed form, which the test header derives from Eq. 18 and Eq. 22 at E = 0.

  C is measured per pusher.
- **T-232 uniform E, B = 0** [A1, A2, A7, WP-015]:
  - u(t) is exact to C·eps(T)·steps for Boris, Vay, Higuera–Cary and `BorisExactRotation`. For `QuadraticUnitary`, the u error is recorded.
  - For all five, the position error against the exact relativistic x(t) is recorded with its observed order. This is done both for a start with x at the half step taken from the analytic solution and for a start at t = 0.
- **T-233 crossed fields** [A2, A7, WP-015]: uniform E ⊥ B with |E|/|B| ∈ {0.1, 0.5, 0.9, 0.99}.
  - A particle started at E×B/B² stays there to C·eps(T)·steps for Vay and Higuera–Cary.
  - The same holds for a force-free particle with a velocity component along B.
  - For Boris, `BorisExactRotation` and `QuadraticUnitary`, the drift error is recorded. For `BorisExactRotation` it is expected to be O((ΩΔt)²) even at small |E|/|B|.
- **T-234 phase-space volume** [A2, A7, A8, WP-015]: the determinant of the one-step momentum map's Jacobian ∂u′/∂u is taken with ForwardDiff at fixed gathered E and B, in the variables u = γv, for random E, B and u with γ up to 10³. With the leapfrog position shears, this is the determinant of the full step.
  - Boris, Higuera–Cary and `BorisExactRotation`: 1 to C·eps(T), with C measured per γ.
  - Vay: equal to Higuera and Cary's ratio J(u_i)/J(u_f) (their Eqs. 40–42) to roundoff. The N-step product in constant fields is recorded.
  - `QuadraticUnitary`: its determinant is recorded. Gordon and Hafizi do not discuss volume.
- **T-235 figure-8** [A1, A2, A7, WP-015]: each pusher runs at a fixed lab Δt with a₀ ∈ {1, 2, 5, 10} and ω₀Δt ∈ {0.2, 0.1, 0.05, 0.025}, using the conventions of the Scope section. Three errors are measured: the error in γ − u_z, the error in u_x + q A_x(η), and the closure error per period in the drift frame.
  - Where a₀·ω₀Δt ≤ 0.5, each error decreases with an observed order of 2 within a measured tolerance.
  - All errors are recorded per pusher against a₀·ω₀Δt, including the cases outside the asymptotic range.
  - For `QuadraticUnitary`, γ − u_z is conserved to roundoff, because its update preserves k·u in a plane wave (Gordon and Hafizi, Sec. 4).
- **T-236 long-run energy in static fields** [A2, A7, WP-015]: the energy error over at least 10⁵ oscillation periods is recorded, with a fitted drift rate per pusher, in Float32 and Float64, so that rounding drift is told apart from scheme drift. Two cases:
  - Higuera and Cary's integrable system (their Eq. 48, H = 4, p_z = 0, p_y,0 ∈ {1.625, 1.775}, Δt = 1/10). The test header states the remaining initial conditions it assumed.
  - A static non-uniform B with a static non-quadratic potential.

  Only finiteness is asserted. A drift bound is proposed for A7 after measurement.
- **T-237 element types** [A8, WP-015]: T-231, T-234 and T-235 at a₀ = 2 pass for Float32, Float64 and `Dual{Float32,3}`, including duals through `BorisExactRotation` near B = 0.
- `build.jl --check` passes and CPU CI is green.

## Depends on

WP-014, which creates the module, the pusher type, `Boris`, and the step that dispatches on it.

## Budgets

A7:

- Conservation in T-231 to T-235 holds to roundoff where it is asserted, with C measured and proposed.
- Convergence orders lie within a measured tolerance of 2.
- Every error constant, drift error, energy-drift rate and resolution threshold is recorded as a proposal, with no thresholds taken from the literature.

Dependencies (F2): OrdinaryDiffEq goes into the `scripts/` environment only, with a compat entry. There is no new dependency in `lib/` or `test/`.
