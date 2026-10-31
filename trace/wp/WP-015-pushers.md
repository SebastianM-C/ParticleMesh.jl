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
# WP-015: swappable relativistic pushers: Boris, Vay, Higuera–Cary and exact-rotation Boris against exact orbits

A2 fixes Boris as the baseline push and asks that each choice be documented with its validation test. The explicit relativistic pushers differ in exactly the regime this project lives in, so the differences are measured here instead of assumed. `docs/plan/pushers-literature.md` collects the claims and their sources. In short:

- **Boris** preserves phase-space volume, but gets the E×B velocity wrong once the drift is relativistic.
- **Vay** fixes the drift but does not preserve volume, and Higuera and Cary show it develops energy errors near resonant orbits.
- **Higuera–Cary** does both, for one more square root.
- **The exact-rotation Boris** of Zenitani and Umeda removes the magnetic gyrophase error, at the price of an O((ΩΔt)²) error in the E×B drift. It does not address the error from splitting the electric boost and magnetic rotation, which Gordon and Hafizi blame for Boris's trouble in strong laser fields.

Among the papers read, the only comparison in a laser field is Gordon and Hafizi's: a plane wave, no wake. At a₀ = 10 and about 63 steps per laser period, they find Boris and Vay not viable and Higuera–Cary visibly off. A PIC run at a fixed lab Δt resolves the same orbit more than ten times more finely (about 820 steps per figure-8 period at a₀ = 10 and ω₀Δt = 0.2). This package therefore measures the pushers at PIC resolution, and separately at Gordon and Hafizi's resolution. Its result decides whether their quadratic-unitary pusher gets a package.

No long-time energy bound is proved for any of these pushers in our fields. In a time-dependent laser field the exact energy is not conserved at all, so energy behaviour is measured in static fields and the laser orbits are checked against their invariants.

A2's AD-smoothness rule decides ties. Switching the default away from Boris would be a proposal to the owner (an edit of A2), not part of this package.

TestParticle.jl is not used as a reference (checked against v0.26.0 on 2026-10-31). Its native Boris pushers, in the `BorisPushers` subpackage, are nonrelativistic and require KernelAbstractions 0.9. Its relativistic path is the Lorentz-force ODE in (x, γv) through OrdinaryDiffEq, which `scripts/pushers/` reproduces directly. Its test files are a useful list of orbit cases.

## Scope

`Vay`, `HigueraCary` and `BorisExactRotation` join WP-014's `Boris` as subtypes of the module's pusher type, in `lib/ParticleMeshElectromagnetic/src/pushers.jl`. Each is a device function with the same signature as `Boris`, taking the momentum `u`, the gathered E and B, and q Δt/(2m) in A1's normalized units. The module step chooses the pusher by type only (A10), so a run swaps pushers by changing one field of its numerics and nothing in `step.jl`.

Implementation notes:

- Derive each update from its defining equation; do not copy a printed matrix form, because the sign and factor-of-2 conventions differ between papers. For Boris that equation is u⁺ − u⁻ = (u⁺ + u⁻) × t/γ with t = qBΔt/(2m).
- Higuera–Cary's γ is their Eq. 20.
- `BorisExactRotation` writes cos θ and sin θ/|B| as functions of θ², with a series near zero. |B| = √(B·B) has an infinite derivative at B = 0, and a threshold on |B| is a branch, so neither is safe for duals.
- The docstring of `BorisExactRotation` notes that exact-rotation schemes become unstable at ΩΔt/γ = kπ.

The tests push single particles in prescribed fields on the CPU, with no mesh. The convergence study against a high-order ODE solution lives in `scripts/pushers/`.

The figure-8 orbit of an electron in a linearly polarized plane wave (Sarachik and Schappert, Phys. Rev. D 1, 2738 (1970)) is the main relativistic test. It has a closed form in the phase η = t − z for a wave travelling in +z with E_x = B_y.

- **Invariants.** Along the exact orbit two quantities are conserved: the light-front momentum γ − u_z, and the transverse canonical momentum u_x + q A_x(η), with q the signed charge in units of e and A in A1's units.
- **Closure.** In the frame of the average drift the orbit closes after every laser period.
- **Initial condition.** The closed form assumes an electron at rest before an adiabatic switch-on. A test that starts at rest inside a constant-amplitude wave therefore starts at a zero of A_x.
- **Half steps.** u lives at integer steps and x at half steps, so A is evaluated at z^n = (z^{n−½} + z^{n+½})/2. Using z^{n+½} directly would add an O(Δt) offset and halve the observed order.
- **Precedent.** ElectronDynamicsModels.jl's figure-8 test integrates the same orbit with a ninth-order solver at tolerance 1e-9. A second-order pusher will not close the orbit at a practical Δt, so the tests assert the convergence order and record the closure error, never closure itself.

The derivation, with these conventions, goes in the test header.

## Deliverables

1. **Source**: the three pushers in `pushers.jl`, generic in `T`, within the §0.4 rules and branch-free in the fields. Each docstring cites its paper and states which properties the pusher has: volume preservation, the E×B velocity, energy for E = 0, gyrophase.
2. **Tests** T-230 to T-237 in the module's test suite. They use only its existing test dependencies, ForwardDiff and Test.
3. **Convergence study** in `scripts/pushers/`.
   - Reference: the same orbits integrated with OrdinaryDiffEq (`Vern9`, absolute and relative tolerance near 1e-12, Float64).
   - Cases: E parallel to B accelerating to γ ~ 100; the figure-8 at a₀ ∈ {0.5, 2, 10}; crossed fields at |E|/|B| ∈ {0.1, 0.9, 0.99}.
   - Plots:
     - error against lab Δt per pusher;
     - the figure-8 at a₀ = 10 at Gordon and Hafizi's resolution, a phase step Δη = 0.1 (about 63 steps per period). This is a diagnostic outside any PIC CFL limit, to compare with their verdict.
     - long-run energy error per pusher, with classic RK4 as the baseline that does not preserve volume.
   - Each plot is a CairoMakie PNG with a JSON sidecar and a run manifest (F3).
   - OrdinaryDiffEq is added to the `scripts/` environment only, never to a `lib/` package or the test environment (F2).
4. **Report**: `reports/pushers.md`.
   - The measured orders, error constants, drift errors and energy behaviour per pusher.
   - Whether the pushers differ materially at PIC resolution, and whether Gordon and Hafizi's verdict holds at their own resolution.
   - A recommendation, stated as a proposal against A2.

## Non-goals

- No implicit, adaptive, multi-evaluation or high-order pushers in the loop (`pushers-literature.md` says why). High-order frozen-field test particles stay on the SciML path.
- No Gordon–Hafizi quadratic-unitary pusher here. It steps in proper time and needs their lab-time conversion. The report decides whether it gets a package.
- No radiation reaction, no change of the default pusher, no TestParticle.jl dependency, no GPU run, no field solve.

## Acceptance

Test ids T-230 to T-239.

- T-230 the pusher slot [A2, A10, WP-015]: a short Landau run of WP-014's step completes with each of the four pushers, changing only the pusher type. With `Boris` it is bitwise identical to WP-014's run.
- T-231 uniform B [A2, A7, A8, WP-015]: for all four pushers, |u| is conserved to C·eps(T)·steps. The per-step gyrophase, with u ⊥ B, matches to roundoff:
  - Boris and Vay: 2·atan(θ/2).
  - `BorisExactRotation`: θ = ΩΔt/γ.
  - Higuera–Cary: 2·atan(β/γ_new), with γ_new from their Eq. 20.

  C is measured per pusher. Ripperda et al. see larger γ errors for Vay and Higuera–Cary than for Boris.
- T-232 uniform E, B = 0 [A1, A2, A7, WP-015]: u(t) is exact to C·eps(T)·steps for all four pushers. The position error against the exact relativistic x(t) is recorded with its observed order, both for a start with x at the half step from the analytic solution and for a start at t = 0.
- T-233 crossed fields [A2, A7, WP-015]: uniform E ⊥ B with |E|/|B| ∈ {0.1, 0.5, 0.9, 0.99}. A particle started at E×B/B² stays there to C·eps(T)·steps for Vay and Higuera–Cary. The same holds for a force-free particle with a velocity component along B. For Boris and `BorisExactRotation`, the drift error is recorded. It is expected to be O((ΩΔt)²) for `BorisExactRotation` even at small |E|/|B|.
- T-234 volume preservation [A2, A7, A8, WP-015]: take with ForwardDiff the determinant of the one-step momentum map's Jacobian ∂u′/∂u at fixed gathered E and B, in the variables u = γv, for random E, B and u with γ up to 10³. With the leapfrog position shears, this is the determinant of the full step.
  - Boris, Higuera–Cary and `BorisExactRotation`: 1 to C·eps(T), with C measured per γ.
  - Vay: equal to Higuera and Cary's ratio J(u_i)/J(u_f) (their Eqs. 40–42) to roundoff. The N-step product in constant fields is recorded.
- T-235 figure-8 [A1, A2, A7, WP-015]: at a₀ ∈ {0.5, 2, 10}, each pusher runs at a fixed lab Δt with ω₀Δt ∈ {0.2, 0.1, 0.05, 0.025}, using the conventions of the Scope section. Three errors are measured:
  - the error in γ − u_z;
  - the error in u_x + q A_x(η);
  - the closure error per period in the drift frame.

  Each decreases with an observed order of 2 within a measured tolerance. The error constants are recorded per pusher and a₀.
- T-236 long-run energy in static fields [A2, A7, WP-015]: the energy error over at least 10⁵ oscillation periods is recorded, with a fitted drift rate per pusher, in Float32 and Float64, so that rounding drift is told apart from scheme drift. Two cases:
  - Higuera and Cary's integrable system (their Eq. 48, H = 4, p_z = 0, p_y,0 ∈ {1.625, 1.775}, Δt = 1/10; the test header states the remaining initial conditions it assumed);
  - a static non-uniform B with a static non-quadratic potential.

  Only finiteness is asserted. A drift bound is proposed for A7 after measurement.
- T-237 element types [A8, WP-015]: T-231, T-234 and T-235 at a₀ = 2 pass for Float32, Float64 and `Dual{Float32,3}`, including duals through `BorisExactRotation` near B = 0.
- `build.jl --check` passes and CPU CI is green.

## Depends on

WP-014, which creates the module, the pusher type, `Boris` and the step that dispatches on it.

## Budgets

A7:

- Conservation in T-231 to T-234 holds to roundoff, with C measured and proposed.
- Convergence orders lie within a measured tolerance of 2.
- Every error constant, drift error and energy-drift rate is recorded as a proposal, with no thresholds taken from the literature.

Dependencies (F2): OrdinaryDiffEq goes into the `scripts/` environment only, with a compat entry. There is no new dependency in `lib/` or `test/`.
