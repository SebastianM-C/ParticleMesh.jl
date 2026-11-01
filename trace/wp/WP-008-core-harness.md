+++
id = "WP-008"
status = "parked"
goals = ["A7", "A8", "A9"]
depends_on = ["WP-014"]
adds_dependencies = ["SciMLBase", "RecursiveArrayTools", "OrdinaryDiffEqFunctionMap"]

[checked]
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-008: engine core, harness: step! as a SciML DiscreteProblem

Parked. The spike's write-up (`reports/spike-electrostatic-1d.md`, section 4, item 6) keeps the harness as an adapter around a working simulation. The engine owns no `step!` (`docs/plan/architecture-draft.md` §0.7); the step the harness wraps is the electromagnetic module's (WP-014), and `run!` is the test loop over it. The deliverables and tests below still name an engine `System`, `State` and `step!` and a physics-free mock module, which do not exist, so this package is re-derived against the module's step, kept or withdrawn, before it is dispatched.

The SciML entry point (A9: SciML is the main integration goal). Contracts: `docs/plan/architecture-draft.md` §0.6, §0.7; `docs/plan/sciml-reuse.md` §1.

## Scope

A package extension of `lib/ParticleMeshCore`, triggered by SciMLBase and RecursiveArrayTools, that wraps `step!` in a `DiscreteProblem` with `FunctionMap`, with `remake`, saved states and parameters following the SciML conventions.

## Deliverables

1. **Check first** that `FunctionMap` (OrdinaryDiffEqFunctionMap; SimpleDiffEq's `SimpleFunctionMap` as the fallback) accepts a state with mixed element types (`Int32` and `T`) through an `ArrayPartition`. Record the result, with versions, in the PR. If it does not, choose a custom `AbstractVector` wrapper and say why.
2. **`State` ↔ `ArrayPartition`**: a zero-copy view and back, plus `copyto!`, in the extension.
3. **The problem constructor**: `DiscreteProblem(system, state, params, tspan)`, whose in-place map copies the previous state into the next and calls `step!`; `remake` with new parameters; `saveat` on whole steps.
4. Weak dependencies SciMLBase and RecursiveArrayTools with an `[extensions]` entry and compat entries; OrdinaryDiffEqFunctionMap (or SimpleDiffEq) as a test dependency. Tests in `lib/ParticleMeshCore/test/harness.jl`, loaded only when the extension's triggers are present.

## Non-goals

No ODE integrator inside the step (§2, time loop). No callbacks that branch on device data. No Reactant path (A9 parks it; OrdinaryDiffEq PR #4436 stays a note in `sciml-reuse.md`). No tangent-bundle layout.

## Acceptance

Test ids T-100 to T-109.

- T-100: `solve` of the `DiscreteProblem` with a physics-free mock module gives a final state bitwise equal to `run!` over the same steps in `Reference()` mode.
- T-101: `remake(prob; p = p′, u0 = load(p′))` gives the same result, bitwise in `Reference()` mode, as a problem built from scratch for p′. Density parameters enter through the weights at load time (A3), so a `remake` that changes only `p` does not reload them; the docstring says so.
- T-102: saved states at `saveat` match `run!` snapshots bitwise.
- T-103: harness overhead per step measured against `run!` and reported (no threshold).
- T-104: the memory the harness holds, measured with `save_everystep = false` as a multiple of the `State`, reported with the cache's fields (A7's memory budget assumes a factor of two; the `FunctionMap` cache may hold more copies, which is unverified).
- T-105: the spike's gradient check (WP-010 deliverable 5, T-143) repeated through `solve`: the ForwardDiff derivative of the same diagnostic with respect to the same weight parameter, the problem built with `remake`, against a Float64 finite-difference scan in `Reference()` mode with common random numbers, following A7's rule: a scan of at least five step sizes, the floor the spread of the three neighbouring difference quotients around the minimum, and the AD derivative asserted within a proposed multiple of that floor, the multiple recorded in `lib/ParticleMeshCore/test/tolerances.toml` marked "proposed" as the spike's T-143 does, and the floor, the multiple and the agreement reported. The case is the electromagnetic module's edge case with no transverse fields (A9).

## Depends on

WP-014 (the module `step!` the harness wraps).

## Budgets

A7 memory budget: the measured factor of T-104, stated in bytes for a reference size, proposed as the budget's multiplier. Harness overhead per step recorded without a threshold. The multiple of the measured floor for T-105 in `tolerances.toml`, marked "proposed". Tier 2 (F4): edits `lib/ParticleMeshCore/Project.toml`.
