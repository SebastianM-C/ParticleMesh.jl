+++
id = "WP-014"
status = "done"
goals = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
depends_on = ["WP-011", "WP-009"]
adds_dependencies = ["ParticleMeshElectromagnetic"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A10 = "66d01e20241b6a7137b53d5c2346b113ac0d3e71"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A6 = "ea80cb6a8044e825be16a39e91d6c45b6dfdb6f9"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
A9 = "d30c76a6cdd40296c596e59d188ed4da9d2fb621"
+++
# WP-014: the 1D electromagnetic module: Yee and Boris, Landau and two-stream as edge cases, Weibel as the third A5 case

The proof of the engine/module split that A9 asks for, and the only route by which the Landau and two-stream cases survive, since A9 rules out an electrostatic module. Each new piece is first tested against an exact discrete or analytic identity, so the physics benchmarks that come last measure resolution and noise, not bugs. Because WP-009 has already proven the flux conservative, a Gauss failure here points to the module. Weibel gets a cold, closed-form definition, so it has the expected value that A5 requires and the plan lacks, without a Smilei run. The package comes after WP-011 and WP-009 because it uses interfaces they own, and it makes the slice's only tier-2 edit, which the owner then reviews against merged engine contracts. It also owns what WP-011 took out of the withdrawn WP-007: the step composition, which the module writes from engine pieces, and the separation of problem description and numerics of A10. Deleting the electrostatic reference goes to a follow-up, because `scripts/es1d/` and WP-012's ParticleMesh column still use it.

## Scope

Creates `lib/ParticleMeshElectromagnetic`, a new workspace member and the only physics module (A9), for N = 1 periodic. It contains the module-declared field layout (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz with their Yee staggers), the 1D Yee curl updates, and a relativistic Boris push with half-step momentum centring (A2) behind a pusher type that the step dispatches on, so that WP-015 can add other pushers without touching the step, with the momenta in `u` as WP-011's contract allows. Everything is in A1's normalized units (c = 1, time in 1/ω₀, length in c/ω₀, fields in m_e c ω₀/e, density in units of n_c).

The module composes its own step: store the old position as extras, push, sort, `deposit_flux!` with Jy and Jz from the quantity callable, fold, Yee half steps. The initial Ex comes from WP-011's `cumsum_periodic!` of the deposited charge, as Smilei solves at t = 0. Yee and Boris live in the module, as §0.1, WP-009's non-goals and `factory-and-observability.md` §3c say. The composition keeps the problem description (case file, field layout, species, parameters) apart from the numerical implementation (Yee in the mesh-solve slot, Boris, the step order), as A10 asks; the engine owns no `System`, `State` or `step!` (WP-011).

The package is validation-first: Yee against its exact discrete dispersion relation, Boris against exact gyration, the discrete Gauss law to roundoff from continuity, and the transverse fields staying exactly zero in the electrostatic edge cases. After those come Landau and two-stream against linear theory, with the difference to the electrostatic reference recorded without a threshold. The schemes differ by construction (Ampère with a relativistic push against Gauss with a nonrelativistic leapfrog), so no bit or roundoff agreement is expected.

Weibel is defined as cold beams counter-streaming along y with k along x, with a closed-form cold-fluid growth rate, and becomes A5's third 1D case. This package owns the transverse-drift extension of the case schema. Adding a workspace member edits `/Project.toml` and a new `/lib/*/Project.toml`, which are tier-2 paths (A6 layout: a `lib/` subpackage as workspace member), so an unattended session opens the PR as a draft for the owner.

## Deliverables

1. **Package**: `lib/ParticleMeshElectromagnetic` with a `Project.toml`: direct dependencies ParticleMeshCore through `[sources]`, KernelAbstractions and Adapt, each with a compat entry, and test dependencies ForwardDiff and Test. It is added to the root workspace, the shared Manifest is re-resolved, and the PR states the change (F2). `lib/README.md` and `architecture-draft.md` §0.1 say the module exists.
2. **Source**: `yee.jl` (1D staggered E and B half steps, periodic through `fill_guards!`), `pushers.jl` (an abstract pusher type with `Boris` as its first subtype, the relativistic Boris momentum update as a device function), `step.jl` (`EMState`; `initialize!` with Ex from `cumsum_periodic!`; `step!`; `run!`), and diagnostics (field energy per component, kinetic energy, mode amplitudes). All of it is KernelAbstractions kernels, generic in `T`, within the §0.4 rules, and engine launches pass WP-011's `workgroupsize` keyword through.
3. **Step composition and A10**: `step.jl`'s docstring and §0.7 state the module's step order and that the module owns it; `initialize!` takes a problem description built from the case file (layout, species, parameters in A1 units) that names no kernel, and the numerics (Yee, Boris, deposition mode, shape) are chosen by type, so the mesh-solve slot of A10 holds the Yee kernels.
4. **Weibel case**: `scripts/crosscode/cases/weibel.toml`: schema 1 extended with a per-species transverse drift. Two cold beams at ±v₀ ŷ, half the density each, immobile ions, k along x. The observable is the growth rate of the Bz field energy in a pre-registered window. The case file states that every quantity is in A1's normalized units (time in 1/ω₀, length in c/ω₀, density in units of n_c). The derivation of the cold-fluid transverse dispersion relation, including the relativistic γ₀ factors, goes in the header of the test that computes the theory value, as `theory.jl` does for T-141 and T-142, and that header states the same normalization. A reference is cited only if it is verified and is otherwise marked unverified.
5. **Drivers and plots**: edge-case drivers that read `landau.toml` and `two-stream.toml` unchanged, with no transverse momenta or fields. `scripts/em1d/` produces the CairoMakie plots of the three cases as PNG with a JSON sidecar, plus a run manifest per run (F3).
6. **Report**: `reports/em-module-1d.md`: the measured rates against theory, against the electrostatic reference and against the stage 1b numbers of WP-012 if present, plus Gauss and energy residuals and CPU timings, all as proposals.
7. **Follow-ups**: a `follow-up` issue (named in the PR description) for deleting `lib/ParticleMeshCore/test/es_reference.jl` and rewiring `scripts/es1d/` to the module, which the slice's other packages still use; and a note re-deriving WP-008 (the SciML harness, parked) now that a module `step!` exists. The PR description also lists A2's binomial current filter as deferred to the 2D LWFA module, because the periodic 1D set does not need it.
8. **Dispatch note** in the PR: if WP-013's PR is open when this one is, rebase and re-resolve the shared Manifest as F2 says (take the base branch's version, re-add the packages, run the tests). The two packages are ordered by the dispatcher, not by a dependency.

## Non-goals

- No binomial filter, no absorbing layers or PML, no laser antenna, no moving window, no bounded axes, no Wakefields.
- No N > 1, no GPU run, no Smilei Weibel run (cross-code stage 2), no SciML harness, no Reactant, no Double32.
- No change to engine interfaces (`cumsum_periodic!`, extras, the quantity and the launch keyword are owned by WP-011; `deposit_flux!` by WP-009). Anything the module needs from them goes to `questions/` and a follow-up issue, not into `lib/ParticleMeshCore/src`.
- No bitwise or roundoff agreement with the electrostatic reference, no threshold against other codes, and no deletion of the reference in this PR.

## Acceptance

Test ids T-210 to T-229.

- T-210 Yee in vacuum [A1, A2, A7, A8, WP-014]: a single Fourier mode advances its phase per step by the ω of the exact discrete relation sin(ωΔt/2)/Δt = sin(kΔx/2)/Δx (c = 1 in A1 units), to C·eps(T)·steps, and the staggered discrete field energy is conserved to C·eps(T)·steps.
- T-211 Boris [A2, A7, A8, WP-014]: in a uniform Bz, |u| is conserved to C·eps(T)·steps and the gyrophase advances per step by 2·atan(ΩΔt/(2γ)) to roundoff. In crossed uniform E and B, the gyration-averaged drift equals E×B/B² within a measured relative error, which is recorded.
- T-212 discrete Gauss law [A2, A7, A8, WP-014]: the backward difference of Ex minus (ρ − mean ρ) is zero to C·eps(T)·n·max|ρ|, with the step-count scaling stated. It holds after every step of the Landau and two-stream edge cases, starting from the `cumsum_periodic!` initial field, in both modes and for Float32, Float64 and `Dual{Float32,3}`.
- T-213 the edge cases stay electrostatic [A9, WP-014]: with zero transverse momenta and fields, Ey, Ez, By and Bz stay exactly zero (bitwise) through the whole Landau and two-stream runs.
- T-214 Landau and T-215 two-stream through the module against linear theory (two-stream against the γ₀³ value, since Boris is relativistic). The relative errors and the difference to the electrostatic reference are recorded as proposals in `tolerances.toml`, and only finiteness and sign are asserted [A5, A7, A9, WP-014].
- T-216 Weibel [A1, A5, A7, WP-014]: the measured growth rate of the Bz energy against the cold-fluid dispersion relation derived in the test header in A1 units, recorded as A5's proposed expected value. The energy transfer from beam kinetic to magnetic energy has the right sign.
- T-217 the total energy drift (fields plus kinetic) over each run is recorded as a proposed A7 bound per case [A7, WP-014].
- T-218 Reference mode is bitwise identical across reruns and at `JULIA_NUM_THREADS` = 1 and 4 for a short run of each case [A4, WP-014].
- T-219 a ForwardDiff derivative of the field energy at a fixed time, with respect to the density perturbation amplitude through the weights (A3), agrees with the `fd_scan` floor within a measured multiple. Float64 Reference, common seeds, on a run long enough for particles to cross cells and the seam [A3, A7, WP-014].
- T-220 allocation per step after warm-up does not grow with particle count or grid size [A7, WP-014].
- `build.jl --check` passes and CPU CI is green. The PR is opened as a draft when unattended (tier 2).

## Depends on

WP-011 (the engine contracts: `cumsum_periodic!`, extras, the deposited quantity, the launch keyword, `fd_scan`) and WP-009 (`deposit_flux!` with its continuity proof). Not on WP-013: the shared Manifest is the only overlap, and F2's rebase rule handles it (deliverable 8).

## Budgets

A7: Gauss's law to roundoff, C·eps(T) scaled by n·max|ρ| and the step count, with C measured and proposed. The energy-drift bound over N steps is measured and proposed per case. Gradient agreement is within the proposed multiple of the measured FD floor. Rates against linear theory are recorded as proposals, with no thresholds. Throughput: CPU ns per particle-step of the EM step, Fast and Reference, at 1 and 16 threads, reported with no threshold (the throughput budget is on the GPU). Dependencies (F2): the new workspace member ParticleMeshElectromagnetic, with direct dependencies ParticleMeshCore, KernelAbstractions and Adapt and test dependencies ForwardDiff and Test, all already in the shared Manifest. Tier 2 through the workspace and `Project.toml` edits: owner approval, and a draft if unattended.
