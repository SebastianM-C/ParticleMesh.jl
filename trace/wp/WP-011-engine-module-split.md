+++
id = "WP-011"
status = "done"
goals = ["A0", "A2", "A3", "A4", "A7", "A8", "A9"]
depends_on = ["WP-010"]
adds_dependencies = []

[checked]
A0 = "a8c098e1e3e8427f42c9d233ce646b15b1b66088"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-011: engine/module split: the electrostatic step leaves the engine, §0.7 written as contracts from the spike's code

The package that turns the spike WP-010 into an engine with contracts (A0: contracts are written from working code). It carries the highest-risk interface work of the slice, and every other package of the slice depends on it. It closes the gaps the spike left open: where the electrostatic step lives under A9, who owns the periodic solve and its association order, how module extras survive the sort, how a module deposits its own quantity, how a launch shape is chosen, and how the finite-difference floor is defined. Each one becomes an engine contract written from working code and proven by a test, and the refactor itself is proven by bitwise equality with the spike's recorded results. Keeping the electrostatic reference in test support avoids a tier-2 edit and keeps `lib/README.md` consistent with A9.

## Scope

`lib/ParticleMeshCore` and the plan text it settles. This is a refactor, proven by the spike's own tests passing unchanged and by bitwise-identical results, plus the physics-free engine interfaces the later packages need, each with an exact identity test.

The `Electrostatic` submodule (`lib/ParticleMeshCore/src/Electrostatic/`: `ESConfig`, `ESState`, `ESParams`, `initialize!`, `step!`, `run!`, `CosineDensity`, `landau_case`, `two_stream_case`, the history and fit helpers) moves out of `src` into test-support code, `lib/ParticleMeshCore/test/es_reference.jl`. The tests and `scripts/es1d/` include that file, as they already include `cases.jl`. No `lib/` package, no new workspace member, no tier-2 path.

The engine keeps the names the spike settled: `Species`, `PeriodicGrid`, `Workspace`, `sort_by_cell!`, `deposit!`, `gather!`, `fill_guards!`/`fold_guards!`, `load_quiet!`, `Shape`/`Stagger`, `Reference`/`Fast`, `Atomic`/`Sorted` and `philox4x32`. This package is the single owner of five engine interfaces:

- (a) `cumsum_periodic!`: a physics-free, deterministic periodic prefix sum with zero mean, taken out of the spike's `gauss_kernel!`. Its association order is part of the contract and is fixed by the operator, not by the launch configuration or the backend: tiles of a fixed constant size, a serial left-to-right sweep inside each tile, the tile totals combined by a serial exclusive scan in tile order, and each tile's offset added once to its elements (none to the first tile); the mean is the sum over the same tiles in the same order. The one-work-item CPU path implements exactly that order, so the CPU test (T-172) and the CUDA test (WP-013, T-192) check one association, and CPU and CUDA results differ only by the backends' own arithmetic (measured in WP-013, T-195). The tile size is a package constant (for example 1024) at least as large as every grid of the bitwise check T-170, so those grids are a single tile and keep the spike's serial bits. The reference's field solve calls it. The names Gauss and Poisson stay in the modules.
- (b) Module-declared per-particle extras: `Species` carries a `NamedTuple` of capacity-length arrays that `sort_by_cell!` permutes together with the core attributes. `u::NTuple{3}` holds whatever the module says it holds: velocities in the reference, momenta in the electromagnetic module.
- (c) A module-supplied deposited quantity in `deposit!`: a device callable `quantity(s, i)` that defaults to q·w. The rest of the spike's signature stays as it is.
- (d) The launch keyword: every engine launch (`sort_by_cell!`, `deposit!`, `gather!`, `fill_guards!`, `fold_guards!`, `load_quiet!`, `cumsum_periodic!`, and every later engine transfer such as WP-009's `deposit_flux!`) takes an optional `workgroupsize = nothing`, where `nothing` means the backend's KernelAbstractions default. It selects only the launch shape; in `Reference()` mode on the CPU backend it must not change the bits (T-177). A4 includes launch shapes in its definition of bitwise; WP-013 tests it on CUDA. On the CPU `cumsum_periodic!` accepts and ignores it (one work item).
- (e) The shared test helper `fd_scan`, which holds A7's operational definition of the finite-difference floor together with A7's conditions on the check.

The §0.7 contract also covers the following points. `advance`'s periodic wrap is the single-device implementation of the particle exchange on periodic axes, and `exchange_particles!` is the later multi-device entry point that replaces it. §1 and §2 list kernels of the whole stack, not of the engine: the engine holds only the physics-free transfers, sort, guards, loading and mesh operators, and Yee and Boris belong to the electromagnetic module. The engine owns no `System`, `State` or `step!`: a module composes its own step from engine pieces, and that composition, with the separation of problem description and numerics of A10, is owned by the electromagnetic module (WP-014). The required element-type matrix is Float32, Float64 and `Dual{Float32,3}` (A8's required-set sentence), and Double32 is a later trial package. The electrostatic reference serves as the reference for the Landau and two-stream edge cases until the electromagnetic module reproduces them, and is deleted after that.

## Deliverables

1. **The reference leaves the engine.** `lib/ParticleMeshCore/test/es_reference.jl`: the former `Electrostatic` submodule as test-support code, included by `runtests.jl`, `physics.jl`, `gradient.jl`, `determinism_worker.jl` and `scripts/es1d/common.jl`. `lib/ParticleMeshCore/src` no longer defines or includes it, and the module docstring no longer mentions it. The file's docstring says that it is the edge-case reference until the electromagnetic module reproduces the cases, and that it is deleted after that (A9: there is no electrostatic module).
2. **`cumsum_periodic!(out, a, grid; workgroupsize = nothing)`** in a new `lib/ParticleMeshCore/src/mesh_ops.jl`: a zero-mean periodic prefix sum on the interior, with its guards filled. The docstring gives the exact discrete identity it satisfies and the tiled association order of item (a), with the tile size as a named constant, so that a later backend can stay bitwise across launch shapes. It runs as one work item on every backend for now; the CUDA scan belongs to WP-013 and implements the same order. The reference's `gauss_kernel!` is rewritten to call it.
3. **Species extras**: an `extras::NamedTuple` field, empty by default, set by the constructor keyword `extras = (old_cell = Int32, old_offset = T, ...)`, allocated at capacity on the species' backend and permuted by `sort_by_cell!` through the existing permutation buffers. An empty tuple costs no extra scratch.
4. **The deposited quantity**: `deposit!` gains `quantity`, a device callable `(s, i) -> value` with the default `ChargeWeight()` (q·w), in Reference and Fast mode alike. The Sorted path's node-wise reduction keeps its order, and the default path stays bitwise identical to the spike's.
5. **The launch keyword** of item (d) on every engine launch, with the default leaving every existing call and its bits unchanged.
6. **`fd_scan(f, x0, hs; seed)`** lifted from `lib/ParticleMeshCore/test/gradient.jl` into `lib/ParticleMeshCore/test/helpers.jl`. It evaluates `f(x, Reference(), seed)` at every step of the scan, so every evaluation runs in the deterministic reference mode with one seed (common random numbers), and it throws unless `x0` and the values `f` returns are Float64. It returns the difference quotients, the spread of three neighbouring quotients, the argmin and the relative floor. Its docstring holds A7's operational definition (the spread of three neighbouring difference quotients, minimised over a scan of at least five steps, relative to the AD value) and A7's three conditions: Float64 even where production is Float32, the Reference mode on both sides, and common random numbers. The caller computes the AD value with the same `f`, mode and seed, and the docstring states that as the caller's obligation. T-143 is rewritten on top of it, with the same assertion.
7. **`tolerances.toml`**: a proposed entry for T-143's multiple (the measured ratio rel/floor, asserted at 1), plus measured "proposed" entries for the new tests.
8. **Plan text**: `docs/plan/architecture-draft.md` §0.7 rewritten from proposals into contracts: the signatures, what each one guarantees and which test proves it, plus the guard convention "deposit into guards, then fold", the accumulation trait with Fast falling back to Reference for `Dual`, the scan's association order, the launch keyword, the wrap/exchange sentence, the stack-versus-engine sentence for §1 and §2, the sentence that the engine owns no `System`, `State` or `step!` and that WP-014 owns the module's step composition and A10, the required element-type matrix, Double32 as a later trial, and the FD-floor definition with A7's three conditions as `fd_scan` states them.
9. **`lib/README.md`**: drop `ParticleMeshElectrostatic` (issue #74) and say that the electrostatic cases are edge cases of ParticleMeshElectromagnetic (A9). `open-items.md` B10: "reference material for a later pairwise/P³M slot, no prerequisite", with the dangling §0.7 item 11 reference removed (issue #47).
10. **Trace**: WP-008 stays "parked", with one line saying that it is re-derived after the electromagnetic module's `step!` exists (WP-014). WP-009 is not touched here (it is re-derived on its own). `build.jl --check` passes.
11. **Scripts**: `scripts/es1d/` rewired to the moved file. Its outputs are regenerated and compared bit for bit with the spike's recorded values, and `scripts/es1d/timing.jl` is rerun before and after the move.

## Non-goals

- No physics change and no new physics: same step order, same kernels, same results bit for bit.
- No new `lib/` package and no new workspace member. No edit of `/Project.toml`, `/lib/*/Project.toml` or `docs/goals/` (no tier-2 path), and no new dependency.
- No GPU code, no CUDA dependency, no AcceleratedKernels scan (WP-013).
- No `deposit_flux!` and no Esirkepov (WP-009). No Yee, no Boris, no electromagnetic code, no module step composition (WP-014).
- No N > 1 or `CartesianIndex{N}` kernels, no bounded axes, no `exchange_particles!` implementation.
- No velocity quiet start and no change to `scripts/crosscode/` case files (WP-012 owns the `seeds` field of the case schema).
- No Molly study, no SciML harness (WP-008 stays parked), no Double32, no Reactant dependency (A9).
- Does not set WP-003's status; the cut does that.

## Acceptance

Test ids T-170 to T-179.

- T-130 to T-145 pass unchanged in name and assertion against the moved code, on 1 and 4 threads, except T-144's two checks in Reference mode at more than one thread: AcceleratedKernels' threaded CPU sample sort allocates about 8 bytes per particle there (#82), so they are `@test_broken` naming #82. The GPU path is the one that matters (`questions/wp-011-t144-threads.md`).
- T-170 the split preserves the bits [A0, A4, WP-011]: the Reference Landau and two-stream rates equal the spike's recorded values bit for bit (γ = −0.1492419736219506 and 0.35456900794051743, the two-stream value with `linear_fit`'s fixed summation order, one ulp above the report's 0.3545690079405174, whose `sum` rounding depended on the CPU's vector width; and the T-136 digest of the test configuration recorded before the move).
- T-171 the engine holds no physics [A9, WP-011]: `!isdefined(ParticleMeshCore, :Electrostatic)`, and no exported engine name contains Gauss, Poisson or Electrostatic. The check runs on `names(ParticleMeshCore)`, not on source text.
- T-172 `cumsum_periodic!` identities [A7, A8, WP-011]: the mean of the output is 0 to C·eps(T)·n·max|a|·dx; the periodic backward difference of the output recovers a − mean(a) to C·eps(T)·n·max|a|; the output is bitwise identical across reruns and thread counts, and bitwise equal to a plain serial implementation of the documented tiled order, on a grid of one tile and on a grid of several tiles. Checked for Float32, Float64 and `Dual{Float32,3}`, with the partials passing the same identities.
- T-173 extras follow the sort [A4, WP-011]: two extras of different element types (the pre-sort slot as `Int32`, and a T-valued copy of the offset), at capacity with dead slots present, still match the same particle after `sort_by_cell!`. The comparison is bitwise against the permuted core attributes, at 1 and 4 threads.
- T-174 `deposit!` quantity [A4, A7, A8, WP-011]: with the default quantity, the result is bitwise equal to the spike's q·w deposition in both modes. With quantity = q·w·u_y, the deposited total equals Σ q·w·u_y to C·eps(T)·count·max|q·w·u_y|, and Reference and Fast agree per node within T-135's proposed bound.
- T-175 `deposit!` with a quantity is the adjoint of `gather!` with the same weighting, including fill and fold, to roundoff. `Dual` weights give exact partials [A3, A7, A8, WP-011].
- T-176 `fd_scan` recovers the derivative of an analytic function with a known roundoff level and reports the floor by the documented definition; it throws for a Float32 `x0` or a Float32 return value. T-143 reads its multiple from `tolerances.toml` [A7, WP-011].
- T-177 the launch keyword [A4, WP-011]: every engine launch accepts `workgroupsize`; in `Reference()` mode, `sort_by_cell!`, `deposit!` and `cumsum_periodic!` give bitwise identical results for the default and two explicit sizes, at 1 and 4 threads, per element type.
- T-144 allocation non-growth still holds with an empty and with a non-empty extras tuple.
- The trace check passes. CPU CI is green. The PR states that §0.7 changed (numerics lens).

## Depends on

WP-010 (the spike, done): its code, tests and recorded values are the baseline the bitwise checks compare against. WP-009, WP-012, WP-013 and WP-014 build on this package.

## Budgets

A7 exact identities: every new test asserts C·eps(T) scaled by the problem size, with C measured and recorded as "proposed" in `tolerances.toml`. The T-143 gradient multiple is recorded as proposed. No throughput threshold: CPU ns per particle-step of the electrostatic step, Fast and Reference, at 1 and 16 threads and the spike's two particle counts, recorded before and after the move with sidecars. Scratch per particle is recorded with and without a two-array extras tuple (the spike measured 33 B for Float32 and 37 B for Float64), and any growth is reported. Allocation per step stays flat across sizes (T-144). Tier 1.
