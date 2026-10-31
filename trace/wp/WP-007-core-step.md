+++
id = "WP-007"
status = "proposed"
goals = ["A3", "A4", "A7", "A8", "A9", "A10"]
depends_on = ["WP-004", "WP-005", "WP-006"]
adds_dependencies = []

[checked]
A10 = "66d01e20241b6a7137b53d5c2346b113ac0d3e71"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "c85034796ed0c2c17c6629196b93ee07c4d8b2f2"
A7 = "b4cade448159d5276774626df7f94299f8621395"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
A9 = "c4a9112c185bc4f4af5c969b185524cc74321372"
+++
# WP-007: engine core, step: System, State, the module interface and step!

Where the engine and the physics modules meet. Contracts: `docs/plan/architecture-draft.md` §0.1 (interface sketch), §0.4 rules 2 and 6, §0.7 items 2, 4, 7, 11, 13, 14 and 17; open item B10.

## Scope

`System` (static: grid, module, deposition mode, shape, layouts, time step, `DiffCache` chunk size) and `State` (fields, species, `GridState`, device-side counters), with workspaces separate; the module interface: how a module declares its field layout, particle extras and species constants, and the hooks `push_particles!(module, ...)`, `boundary!(module, species, grid)`, the mesh-solve hook and the deposited quantity; `step!(system, state, params)` as a pure function composing the engine's pieces; `run!` as a plain loop; a physics-free mock module that proves the split.

## Deliverables

1. **B10 first.** Half a day reading Molly's `System`, interaction and simulator types and `src/interactions/ewald.jl` to settle the names and signatures of the container and interaction interfaces; record the outcome in the PR description. No code dependency on Molly.
2. **`System`, `State`, workspaces** and their constructors from a module.
3. **The module interface** as functions a module implements, documented: layout declaration, extras and constants, which fields are deposit targets, `push_particles!`, `boundary!`, the mesh-solve hook, and `deposit_sources!(module, targets, species, ...)`, whose default calls the engine's `deposit!` with the module's quantity and which a module overrides with its own deposition (the LWFA module's Esirkepov current, §0.7 item 14). No kernel is named `push!`.
4. **`step!`**: zero the deposit targets the module declares (only those), `deposit_sources!` for every species (sorting first in `Reference()` mode or for a `Sorted()` type), `fold_guards!` on the deposit targets only (folding fields whose guards hold filled copies would double edge values), the module's mesh solve, `fill_guards!`, gather, the module's push, `exchange_particles!`, `boundary!`, advance the step counter in `GridState`. Static dispatch only; no host-side branching on device data, no globals.
5. **Mock module** in the test suite, using the default `deposit_sources!`: a 1D periodic "mesh solve" with an exact discrete answer (for example a linear map of the deposited density) and an analytic push, so the composition has a closed-form result. No physics names.
6. Tests in `lib/ParticleMeshCore/test/step.jl`.

## Non-goals

No electrostatic or electromagnetic module (their own packages in `lib/`). No SciML harness (WP-008). No tangent-bundle layout or `jvp` interface. No Reactant (A9). This package touches no `Project.toml`.

## Acceptance

Test ids T-080 to T-099.

- T-080: `step!` with the mock module matches the closed-form result over N steps to a proposed `C·eps(T)` bound, for every `T` of the A8 matrix.
- T-081: `Reference()` mode is bitwise reproducible over N steps across reruns and across `JULIA_NUM_THREADS` = 1 and many (A4).
- T-082: `Reference()` and `Fast()` agree over N steps to a proposed bound (Float32 and Float64).
- T-083: a `ForwardDiff.gradient` of a scalar of the final state with respect to a density parameter matches a Float64 finite-difference scan (`fd_scan`), with the scan's error floor reported; both sides run in `Reference()` mode with the same seed (common random numbers), and the parameter enters through the weights only (A3, A7).
- T-084: allocation of `step!` after warm-up does not grow between two particle counts and two grid sizes (§0.7 item 13).
- T-085: the engine source contains no physics names (a test over `lib/ParticleMeshCore/src`), and the mock module is defined only in the tests.
- T-086: the step counter in `GridState` advances and t = n·dt; changing it changes no type.

## Depends on

WP-004, WP-005, WP-006.

## Budgets

CPU time per particle-step of the mock step recorded without a threshold. Proposed constants for T-080 and T-082 in `tolerances.toml`, marked "proposed". The FD check runs in Float64 only (A7).
