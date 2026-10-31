+++
id = "WP-004"
status = "parked"
goals = ["A1", "A6", "A7", "A8", "A9"]
depends_on = []
adds_dependencies = ["KernelAbstractions", "Atomix", "Adapt", "AcceleratedKernels", "PreallocationTools", "ForwardDiff", "DoubleFloats"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A6 = "344832ac0e32f874b8cbaa674a2282ee40a05d9c"
A7 = "b4cade448159d5276774626df7f94299f8621395"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
A9 = "c4a9112c185bc4f4af5c969b185524cc74321372"
+++
# WP-004: engine core, mesh: bootstrap, grid, field sets, guard fill

Parked (A0: contracts are written from working code). The spike WP-010 builds a 1D electrostatic PIC first and tests the contracts below as proposals; this package is re-derived from the spike's write-up, kept or withdrawn, before it is dispatched.

The first engine package. It creates `lib/ParticleMeshCore` with its whole dependency set and the test scaffolding every later core package uses, and implements the mesh side of the engine. The contracts are `docs/plan/architecture-draft.md` §0.7 items 1 to 4, 9, 10, 13, 16 to 19; read §0.4 and §0.7 in full before starting.

## Scope

`Grid{N}` for N = 1, 2, 3 with its static part (extents, spacing as `T`, one symmetric guard width as a type parameter, per-axis `Periodic()` or `Bounded()`) and the dynamic `GridState` (per-axis global index of the first local cell, the step counter) as a small device-side integer array; the index helper `idx(grid, gstate, I)`; `StaggeredField{T,N}` and field sets allocated from a module-declared layout (a `NamedTuple` of staggers, with one `T` or a `NamedTuple` of element types); `fill_guards!(fields, grid, gstate)` on periodic axes as a KernelAbstractions kernel. No physics names anywhere.

## Deliverables

1. **Bootstrap.** `lib/ParticleMeshCore` with `Project.toml`, added to the root workspace `projects`; the umbrella `ParticleMesh` reaches it through `[sources]` and re-exports it. Direct dependencies, resolved together and with compat entries: KernelAbstractions `"0.10"` (kernels written against the 0.10 API), Atomix, Adapt, AcceleratedKernels (0.5, which admits KA 0.10), PreallocationTools. The PR description lists the resolved versions, including the CUDA.jl version the workspace would take later, and says that no package depends on Reactant (A9).
2. **Grid and `GridState`**, with the conventions of item 10 written into docstrings: 1-based local cells, 0-based global index, in-cell offset measured from the lower unstaggered node, face k = 2(axis − 1) + 1 (low) or + 2 (high). `Bounded()` exists as a type and is rejected with a clear error wherever this package does not handle it yet.
3. **Field sets.** `StaggeredField`, allocation from a layout through `similar` or `KernelAbstractions.allocate`, and `idx` for every access. The guard width comes in as a parameter; the transfers package later supplies `guard_width(shape)`.
4. **`fill_guards!`** for periodic axes, all N, all staggers, written as a KA kernel within the §0.4 rules (index arithmetic, loads and stores only).
5. **Test scaffolding** for all core packages: `lib/ParticleMeshCore/test/runtests.jl` with an include list (one file per work package: `mesh.jl` here); a helper module with `roundoff_bound(T; n, steps, C)`, the A8 type-matrix loop (Float32, Float64, Double32, `Dual{Float32,3}`), `fd_scan(f, x; hs)`; `lib/ParticleMeshCore/test/tolerances.toml` with one table per package; `lib/ParticleMeshCore/test/Project.toml` as a workspace member, with ParticleMeshCore through `[sources]` plus ForwardDiff and DoubleFloats as test dependencies, so `julia +1.13 --project=lib/ParticleMeshCore -e 'using Pkg; Pkg.test()'` works as `CLAUDE.md` describes; and the root `test/runtests.jl` includes the core suite, with the root `test/Project.toml` listing the same dependencies, so the existing CI job runs it with no workflow change.

## Non-goals

No particles, shapes, deposition, `fold_guards!` (transfers package) or `step!`. No Reactant or CUDA.jl dependency (A9 parks Reactant). No moving window: only the hooks of item 10. No unit conversion (A1: user layer only). No ForwardDiff or DoubleFloats in `lib/ParticleMeshCore`'s own dependencies; they are test dependencies only (§0.7 item 8).

## Acceptance

Test ids T-020 to T-039, each named with its id and what it verifies, for example `@testset "T-020 fill_guards! reproduces a periodic function exactly [A7, WP-004]"`.

- T-020 to T-022: `fill_guards!` reproduces a periodic function exactly (bitwise) on every stagger, for N = 1, 2, 3 and every `T` of the A8 matrix.
- T-023: `idx` with a nonzero global offset in `GridState` addresses the same storage as with zero offset shifted by hand; changing `GridState` changes no type (check with `typeof`).
- T-024: layouts with a per-component element type allocate the declared types; a 2D3V layout is `Grid{2}` with three-component vector fields and nothing stored for the collapsed axis.
- T-025: `Bounded()` axes raise the documented error in `fill_guards!`.
- T-026: allocation of `fill_guards!` after warm-up does not grow between two grid sizes (§0.7 item 13).
- T-027: the type matrix runs with Float32, Float64 and `Dual{Float32,3}` passing; Double32 passes or is `@test_broken` with the missing DoubleFloats function named (A8).
- `julia +1.13 --project=scripts scripts/trace/build.jl --check` passes; the CI test job is green.

## Depends on

Nothing. Later core packages depend on it.

## Budgets

A7's throughput budget is defined on the GPU; this package records CPU timings of `fill_guards!` without a threshold and proposes a CI-time budget for the core suite, marked unverified. Tolerances: everything here is exact (bitwise), so `tolerances.toml` gets the table with no proposed constants yet. Tier 2: the PR edits `/Project.toml` and `/lib/*/Project.toml` (`.github/CODEOWNERS`), so it needs the owner's approval, and an unattended session opens it as a draft.
