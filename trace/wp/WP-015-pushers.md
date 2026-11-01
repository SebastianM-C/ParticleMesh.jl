+++
id = "WP-015"
status = "active"
goals = ["A1", "A2", "A6", "A7", "A8", "A10"]
depends_on = ["WP-014"]
adds_dependencies = ["SciMLBase", "OrdinaryDiffEqCore", "OrdinaryDiffEqVerner"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A10 = "30906c6b96b63dab1b6d4eab77c5f25dcb8e1213"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
+++
# WP-015: swappable relativistic pushers (Boris, Vay, Higuera–Cary, exact-rotation Boris) as device functions and as OrdinaryDiffEq algorithms, against exact orbits

A2 fixes Boris as the baseline push and asks that each choice be documented with its validation test. The explicit relativistic pushers differ in exactly the regime this project lives in, so the differences are measured instead of assumed. `docs/plan/pushers-literature.md` collects the claims and their sources. In short:

- Boris preserves phase-space volume but gets the E×B velocity wrong once the drift is relativistic.
- Vay fixes the drift but does not preserve volume, and Higuera and Cary show it develops energy errors near resonant orbits.
- Higuera–Cary does both, for one more square root.
- The exact-rotation Boris of Zenitani and Umeda removes the magnetic gyrophase error, at the price of an O((ΩΔt)²) error in the E×B drift.

This package builds the four pushers and checks each against the exact orbits and structural properties the literature claims. WP-016 adds Gordon and Hafizi's quadratic-unitary pusher and the scan over a₀·ω₀Δt that turns the comparison into a resolution rule for laser runs.

The pushers are also built to be usable through SciML. Each pusher is one device function, which the PIC kernel calls. A package extension wraps the same functions as OrdinaryDiffEq algorithms for single particles in prescribed fields. Tests, scripts and later test-particle stages can then compare a pusher with another one, or with a high-order reference such as `Vern9`, by changing only the algorithm, and they get `saveat`, callbacks and ensembles for free. The PIC step itself does not go through this interface (`sciml-reuse.md` §1). This part is exploratory: the package builds it, and the report records what fits and what does not.

No long-time energy bound is proved for any of these pushers in non-uniform fields. Energy behaviour is therefore measured in static fields, not asserted. A2's AD-smoothness rule decides ties. Switching the default away from Boris would be a proposal to the owner (an edit of A2), not part of this package.

TestParticle.jl is not used as a reference (checked against v0.26.0 on 2026-10-31). Its native Boris pushers, in the `BorisPushers` subpackage, are nonrelativistic and require KernelAbstractions 0.9. That subpackage is prior art for the SciML wrapping: algorithms as `OrdinaryDiffEqAlgorithm` subtypes with their own `perform_step!`, half-step initialization and `saveat` handling. Its test files are a useful list of orbit cases.

## Scope

**Device functions.** `Vay`, `HigueraCary` and `BorisExactRotation` join WP-014's `Boris` as subtypes of the module's pusher type, in `lib/ParticleMeshElectromagnetic/src/pushers.jl`. Each is a device function with the same signature as `Boris`. It takes the momentum `u`, the gathered E and B, and q Δt/(2m) in A1's normalized units. The module step chooses the pusher by type only (A10), so a run swaps pushers by changing one field of its numerics and nothing in `step.jl`.

Implementation notes:

- Derive each update from its defining equation, not from a printed matrix form; the sign and factor-of-2 conventions differ between papers. For Boris that equation is u⁺ − u⁻ = (u⁺ + u⁻) × t/γ with t = qBΔt/(2m).
- Higuera–Cary's γ is their Eq. 20.
- `BorisExactRotation` writes cos θ and sin θ/|B| as functions of θ², with a series near zero. The norm |B| = √(B·B) has an infinite derivative at B = 0, and a threshold on |B| is a branch, so neither is safe for duals. Its docstring notes that exact-rotation schemes become unstable at ΩΔt/γ = kπ.

**SciML interface.** The extension `ParticleMeshElectromagneticOrdinaryDiffEqCoreExt` has weak dependencies OrdinaryDiffEqCore and SciMLBase. It makes each pusher an OrdinaryDiffEq algorithm for one particle with state (x, u), whose parameters carry E(x, t) and B(x, t) as callables. The algorithm's step calls the same device function as the PIC kernel, so each pusher has exactly one implementation. Questions to settle while building, with the answers recorded in the report:

- Whether the algorithm types wrap the pusher types, or whether the pusher types become the algorithm types. The second needs OrdinaryDiffEqCore as a hard dependency of the module, which this package does not add.
- How the half-step momentum stagger is started, and how saved states are synchronized to integer times.
- Which problem type fits: a plain `ODEProblem` with a placeholder right-hand side, as `BorisPushers` uses, or a `DynamicalODEProblem`.
- Whether DiffEqGPU's `EnsembleGPUKernel` can run these algorithms. This is checked on paper only, with no GPU run.

**Tests.** The tests push single particles in prescribed fields on the CPU, with no mesh. Where convenient they go through the SciML interface, with `Vern9` from OrdinaryDiffEqVerner as the reference for orbits without a closed form.

The figure-8 orbit of an electron in a linearly polarized plane wave (Sarachik and Schappert, Phys. Rev. D 1, 2738 (1970)) is the relativistic exact-orbit test. Its closed form is in the phase η = t − z, for a wave travelling in +z with E_x = B_y.

- Two quantities are conserved along the exact orbit: the light-front momentum γ − u_z, and the transverse canonical momentum u_x + q A_x(η). Here q is the signed charge in units of e and A is in A1's units.
- In the frame of the average drift, the orbit closes after every laser period.
- The closed form assumes an electron at rest before an adiabatic switch-on. A test that starts at rest in a constant-amplitude wave therefore starts at a zero of A_x.
- u lives at integer steps and x at half steps, so A is evaluated at z^n = (z^{n−½} + z^{n+½})/2. Using z^{n+½} adds an O(Δt) offset that halves the observed order.
- ElectronDynamicsModels.jl's figure-8 test integrates the same orbit with a ninth-order solver at tolerance 1e-9. A second-order pusher will not close the orbit at a practical Δt, so the tests assert the convergence order and record the closure error, never closure itself.

The derivation, with these conventions, goes in the test header.

## Deliverables

1. **Source**: the three pushers in `pushers.jl`, generic in `T`, within the §0.4 rules and branch-free in the fields. Each docstring cites its paper and states which of these properties the pusher has: volume preservation, the E×B velocity, energy for E = 0, and gyrophase.
2. **Extension**: `ext/ParticleMeshElectromagneticOrdinaryDiffEqCoreExt.jl`, with OrdinaryDiffEqCore and SciMLBase as weak dependencies and compat entries in `lib/ParticleMeshElectromagnetic/Project.toml`. OrdinaryDiffEqCore, SciMLBase and OrdinaryDiffEqVerner are test dependencies. The shared Manifest is re-resolved and the PR lists the change (F2).
3. **Tests**: T-230 to T-237.
4. **Scripts**: `scripts/pushers/`, which runs the long-run energy cases of T-236 beyond test length. It plots per pusher, with classic RK4 as the baseline that does not preserve volume. Each plot is a CairoMakie PNG with a JSON sidecar and a run manifest (F3).
5. **Report**: `reports/pushers.md`, with three parts:
   - the measured orders, error constants, drift errors and energy behaviour per pusher;
   - the answers to the SciML questions above;
   - a recommendation, stated as a proposal against A2. WP-016 adds the resolution rule to the same report.
6. **Tier 2** (F4): the edit of `lib/ParticleMeshElectromagnetic/Project.toml` is a code-owner path.

## Non-goals

- No quadratic-unitary pusher and no a₀ scan (WP-016).
- No implicit, adaptive, multi-evaluation or high-order pushers in the loop (`pushers-literature.md` says why).
- The PIC step does not go through OrdinaryDiffEq, and OrdinaryDiffEqCore is not a hard dependency of the module.
- No radiation reaction, no change of the default pusher, no TestParticle.jl dependency, no GPU run, no field solve.

## Acceptance

Test ids T-230 to T-237.

- **T-230 the pusher slot** [A2, A10, WP-015]: a short Landau run of WP-014's step completes with each of the four pushers, changing only the pusher type. With `Boris` it is bitwise identical to WP-014's run.
- **T-231 uniform B** [A2, A7, A8, WP-015]: for all four pushers, |u| is conserved to C·eps(T)·steps. With u ⊥ B, the per-step gyrophase matches its closed form to roundoff:
  - Boris and Vay: 2·atan(θ/2).
  - `BorisExactRotation`: θ = ΩΔt/γ.
  - Higuera–Cary: 2·atan(β/γ_new), with γ_new from their Eq. 20.

  C is measured per pusher.
- **T-232 uniform E, B = 0** [A1, A2, A7, WP-015]: u(t) is exact to C·eps(T)·steps for all four pushers. The position error against the exact relativistic x(t) is recorded with its observed order, both for x at the half step taken from the analytic solution and for a start at t = 0.
- **T-233 crossed fields** [A2, A7, WP-015]: uniform E ⊥ B with |E|/|B| ∈ {0.1, 0.5, 0.9, 0.99}.
  - A particle started at E×B/B² stays there to C·eps(T)·steps for Vay and Higuera–Cary.
  - The same holds for a force-free particle with a velocity component along B.
  - For Boris and `BorisExactRotation`, the drift error is recorded. It is expected to be O((ΩΔt)²) for `BorisExactRotation` even at small |E|/|B|.
- **T-234 phase-space volume** [A2, A7, A8, WP-015]: the determinant of the one-step momentum map's Jacobian ∂u′/∂u is taken with ForwardDiff at fixed gathered E and B, in the variables u = γv, for random E, B and u with γ up to 10³. With the leapfrog position shears, this is the determinant of the full step.
  - Boris, Higuera–Cary and `BorisExactRotation`: 1 to C·eps(T), with C measured per γ.
  - Vay: equal to Higuera and Cary's ratio J(u_i)/J(u_f) (their Eqs. 40–42) to roundoff. The N-step product in constant fields is recorded.
- **T-235 figure-8** [A1, A2, A7, WP-015]: at a₀ ∈ {0.5, 2}, each pusher runs at a fixed lab Δt with ω₀Δt ∈ {0.2, 0.1, 0.05, 0.025}. Three errors are measured: in γ − u_z, in u_x + q A_x(η), and the closure error per period in the drift frame. Each decreases with an observed order of 2 within a measured tolerance, and the error constants are recorded.
- **T-236 long-run energy in static fields** [A2, A7, WP-015]: over at least 10⁴ oscillation periods in the test, and longer in the script, the energy error is recorded with a fitted drift rate per pusher. Runs are in Float32 and Float64, so rounding drift is told apart from scheme drift. Two systems:
  - Higuera and Cary's integrable system: their Eq. 48, with H = 4, p_z = 0, p_y,0 ∈ {1.625, 1.775} and Δt = 1/10. The test header states the remaining initial conditions it assumed.
  - A static non-uniform B with a static non-quadratic potential.

  Only finiteness is asserted. A drift bound is proposed for A7 after measurement.
- **T-237 the SciML interface** [A7, A8, A10, WP-015]:
  - Through `solve`, each pusher reproduces the orbit of its device function, stepped by hand, bitwise for the same Δt and start.
  - The figure-8 at a₀ = 2, solved with `Vern9` through the same problem, agrees with the closed form within the solver tolerance.
  - T-231, T-234 and T-235 at a₀ = 2 also pass for Float32, Float64 and `Dual{Float32,3}`, including duals through `BorisExactRotation` near B = 0.
- `build.jl --check` passes and CPU CI is green. Tier 2 (F4).

## Depends on

WP-014, which creates the module, the pusher type, `Boris` and the step that dispatches on it.

## Budgets

A7:

- Conservation in T-231 to T-234 to roundoff, with C measured and proposed.
- Convergence orders within a measured tolerance of 2.
- Every error constant, drift error and energy-drift rate is recorded as a proposal, with no thresholds taken from the literature.

Dependencies (F2):

- SciMLBase and OrdinaryDiffEqCore as weak dependencies of ParticleMeshElectromagnetic, and as test dependencies with OrdinaryDiffEqVerner, all with compat entries.
- None of them is a hard dependency of `lib/` or of the engine.

Each worktree pays one extra precompilation for them.
