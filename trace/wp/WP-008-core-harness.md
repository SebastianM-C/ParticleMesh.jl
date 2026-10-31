+++
id = "WP-008"
status = "proposed"
goals = ["A7", "A8", "A9"]
depends_on = ["WP-007"]
adds_dependencies = ["SciMLBase", "RecursiveArrayTools", "OrdinaryDiffEqFunctionMap"]

[checked]
A7 = "b4cade448159d5276774626df7f94299f8621395"
A8 = "c60f1646a0139b96db9e52e98cbd6408a1a9272a"
A9 = "89bb5c44cea879c80e8ec750e17539f6de4ee053"
+++
# WP-008: engine core, harness: step! as a SciML DiscreteProblem

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
- T-101: `remake` with new parameters gives the same result as a fresh `System` with those parameters.
- T-102: saved states at `saveat` match `run!` snapshots bitwise.
- T-103: harness overhead per step measured against `run!` and reported (no threshold).
- T-104: the memory of the harness state is twice the `State`, as A7's memory budget assumes; measured and reported.

## Depends on

WP-007.

## Budgets

A7 memory budget: the measured doubling of T-104, stated in bytes for a reference size. Harness overhead per step recorded without a threshold. Tier 2: edits `lib/ParticleMeshCore/Project.toml`, so the owner approves it, and an unattended session opens it as a draft.
