+++
id = "WP-008"
status = "parked"
goals = ["A7", "A8", "A9"]
depends_on = ["WP-007"]
adds_dependencies = ["SciMLBase", "RecursiveArrayTools", "OrdinaryDiffEqFunctionMap"]

[checked]
A7 = "b4cade448159d5276774626df7f94299f8621395"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
A9 = "c4a9112c185bc4f4af5c969b185524cc74321372"
+++
# WP-008: engine core, harness: step! as a SciML DiscreteProblem

Parked (A0: contracts are written from working code). The spike WP-010 builds a 1D electrostatic PIC first and tests the contracts below as proposals; this package is re-derived from the spike's write-up, kept or withdrawn, before it is dispatched.

The SciML entry point (A9: SciML is the main integration goal). Contracts: `docs/plan/architecture-draft.md` §0.6, §0.7 item 11; `docs/plan/sciml-reuse.md` §1.

## Scope

A package extension of `lib/ParticleMeshCore`, triggered by SciMLBase and RecursiveArrayTools, that wraps `step!` in a `DiscreteProblem` with `FunctionMap`, with `remake`, saved states and parameters following the SciML conventions.

## Deliverables

1. **Check first** that `FunctionMap` (OrdinaryDiffEqFunctionMap; SimpleDiffEq's `SimpleFunctionMap` as the fallback) accepts a state with mixed element types (`Int32` and `T`) through an `ArrayPartition`. Record the result, with versions, in the PR. If it does not, choose a custom `AbstractVector` wrapper and say why.
2. **`State` ↔ `ArrayPartition`**: a zero-copy view and back, plus `copyto!`, in the extension.
3. **The problem constructor**: `DiscreteProblem(system, state, params, tspan)` (or the name B10 settled in WP-007), whose in-place map copies the previous state into the next and calls `step!`; `remake` with new parameters; `saveat` on whole steps.
4. Weak dependencies SciMLBase and RecursiveArrayTools with an `[extensions]` entry and compat entries; OrdinaryDiffEqFunctionMap (or SimpleDiffEq) as a test dependency. Tests in `lib/ParticleMeshCore/test/harness.jl`, loaded only when the extension's triggers are present.

## Non-goals

No ODE integrator inside the step (§0.7 item 11). No callbacks that branch on device data. No Reactant path (A9 parks it; OrdinaryDiffEq PR #4436 stays a note in `sciml-reuse.md`). No tangent-bundle layout.

## Acceptance

Test ids T-100 to T-109.

- T-100: `solve` of the `DiscreteProblem` with the WP-007 mock module gives a final state bitwise equal to `run!` over the same steps in `Reference()` mode.
- T-101: `remake(prob; p = p′, u0 = load(p′))` gives the same result, bitwise in `Reference()` mode, as a problem built from scratch for p′. Density parameters enter through the weights at load time (A3), so a `remake` that changes only `p` does not reload them; the docstring says so.
- T-102: saved states at `saveat` match `run!` snapshots bitwise.
- T-103: harness overhead per step measured against `run!` and reported (no threshold).
- T-104: the memory the harness holds, measured with `save_everystep = false` as a multiple of the `State`, reported with the cache's fields (A7's memory budget assumes a factor of two; the `FunctionMap` cache may hold more copies, which is unverified).

## Depends on

WP-007.

## Budgets

A7 memory budget: the measured factor of T-104, stated in bytes for a reference size, proposed as the budget's multiplier. Harness overhead per step recorded without a threshold. Tier 2: edits `lib/ParticleMeshCore/Project.toml`, so the owner approves it, and an unattended session opens it as a draft.
