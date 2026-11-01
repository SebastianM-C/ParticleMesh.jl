+++
id = "WP-008"
status = "done"
goals = ["A3", "A7", "A8", "A9"]
depends_on = ["WP-014"]
adds_dependencies = ["SciMLBase", "RecursiveArrayTools", "OrdinaryDiffEqFunctionMap"]

[checked]
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-008: the SciML harness: the electromagnetic module's step as a `DiscreteProblem`

SciML is the main integration goal (A9). Wrapping the module's step in SciML is the entry point for the rest of that integration: callbacks, ensembles, `remake` over parameters, and later the reduced models and sensitivities (`docs/plan/sciml-reuse.md` §1). The engine owns no step (`architecture-draft.md` §0.7). The step to wrap is the electromagnetic module's (WP-014), which takes the form `step!(state::EMState, params::EMParams, k)`.

In that form, the parameters of A3 enter through the problem description (`EMProblem`): `initialize!` turns them into particle weights. `EMParams` carries only the time step and the diagnostic wavenumber. The harness is an adapter around this working simulation. It must give results bitwise equal to the module's own `run!` loop, so it adds an interface without adding a second implementation.

## Scope

A package extension of `lib/ParticleMeshElectromagnetic` triggered by SciMLBase and RecursiveArrayTools. It builds a `DiscreteProblem` whose map advances the module by one step, and it is solved with `FunctionMap` (OrdinaryDiffEqFunctionMap). SimpleDiffEq's `SimpleFunctionMap` is the fallback if the first check below fails.

**What the SciML state holds.** It holds the particles and the fields, as an `ArrayPartition` over their arrays. The workspaces and the diagnostic histories live in a cache that the problem owns, built from an `EMState`:
- **Why the histories stay out:** `EMState` keeps one history entry per step for the whole run. The `FunctionMap` cache copies the previous state into the next on every step, so carrying the histories in the state would cost a full copy of them per step.
- **The step index:** the map recovers `k` from `t` and `dt`, as an integer on the host.
- **Element types:** the cache is allocated for the state's element type, so a problem rebuilt with duals (T-105) gets a dual cache. The extension says how it does this: rebuild in `remake`, or keep a cache keyed by element type.

**Parameters.** They follow the module's split:
- `p` is the problem's parameter set: the quantities `initialize!` reads, such as the perturbation amplitude and the densities.
- `u0` is the loaded state for that `p`.

`remake(prob; p = p′)` therefore has to reload the state. The extension provides a constructor that does both, and its docstring states that a `remake` changing only `p` does not reload the weights (A3).

`saveat` works on whole steps. Saved states are copies of the particle and field arrays. The diagnostic histories stay readable from the cache.

## Deliverables

1. **Check first:** whether `FunctionMap` accepts an `ArrayPartition` with mixed element types, `Int32` cells and status next to `T` offsets, momenta, weights and fields. Record the result, with package versions, in the PR. If it does not, use a custom `AbstractVector` wrapper or `SimpleFunctionMap`, and say why.
2. **State mapping:** a zero-copy `ArrayPartition` over an `EMState`'s particle and field arrays, and `copyto!` back.
3. **Problem constructor:**
   - `DiscreteProblem` from an `EMProblem` and `EMNumerics`. It calls `initialize!` for `u0`, and its in-place map calls the module's `step!`.
   - a `remake` path for new parameters that reloads the state;
   - `saveat` on whole steps.
4. **Dependencies:**
   - SciMLBase and RecursiveArrayTools as weak dependencies of ParticleMeshElectromagnetic, with an `[extensions]` entry and compat entries;
   - OrdinaryDiffEqFunctionMap as a test dependency.

   WP-015 adds a second extension to the same package (pushers as OrdinaryDiffEq algorithms, on OrdinaryDiffEqCore). The two share SciMLBase, and whichever lands second rebases its `Project.toml` and manifest as F2 says.
5. **Tests:** in `lib/ParticleMeshElectromagnetic/test/harness.jl`.

## Non-goals

- No ODE integrator inside the step (`sciml-reuse.md` §1), and no callbacks that branch on device data.
- No Reactant path (A9).
- No tangent-bundle layout.
- No change to the module's `step!`, `initialize!` or `run!`.
- No deletion of the electrostatic reference (issue #97, part 1).

## Acceptance

Test ids T-100 to T-109.

- T-100 harness equals the loop [A9, WP-008]: `solve` of the `DiscreteProblem` over n steps of a Landau edge case gives a final state bitwise equal to `initialize!` plus `run!` over the same steps in `Reference()` mode. The diagnostic histories in the cache are also bitwise equal.
- T-101 `remake` [A3, A9, WP-008]: the reloading `remake` path with p′ gives the same result, bitwise in `Reference()` mode, as a problem built from scratch for p′.
- T-102 saved states [A9, WP-008]: the states saved at `saveat` match snapshots of `run!` bitwise.
- T-103 overhead [A7, WP-008]: the harness overhead per step, measured against `run!` and reported. There is no threshold.
- T-104 memory [A7, WP-008]: the memory the harness holds with `save_everystep = false`, as a multiple of the state's particle and field arrays. It is reported together with the cache's fields. A7's memory budget assumes a factor of two; whether the `FunctionMap` cache holds more copies is unverified, and this test measures it.
- T-105 gradient through `solve` [A3, A7, A8, WP-008]: the ForwardDiff derivative of the field energy at a fixed time with respect to the perturbation amplitude, through `solve` with the problem built by the reloading `remake`. It is compared with WP-014's T-219 value and with the `fd_scan` floor: Float64, `Reference()` mode, common seeds, asserted within a proposed multiple of the floor. The multiple goes in `tolerances.toml` marked "proposed".
- T-106 element types [A8, WP-008]: T-100 passes for Float32, Float64 and `Dual{Float32,3}`.
- The trace check passes and CPU CI is green.

## Depends on

WP-014: the module's `EMState`, `EMProblem`, `EMParams`, `initialize!`, `step!` and `run!`.

## Budgets

A7:
- **Memory:** the factor measured by T-104, in bytes for a reference size, proposed as the budget's multiplier.
- **Overhead:** the harness overhead per step, recorded with no threshold.
- **T-105:** its multiple of the floor, in `tolerances.toml` marked "proposed".

Tier 2 (F4): the package edits `lib/ParticleMeshElectromagnetic/Project.toml`. Dependencies (F2): SciMLBase and RecursiveArrayTools as weak dependencies, and OrdinaryDiffEqFunctionMap for tests. Each worktree pays one extra precompilation.
