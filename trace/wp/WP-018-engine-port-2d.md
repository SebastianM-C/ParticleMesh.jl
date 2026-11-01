+++
id = "WP-018"
status = "done"
goals = ["A0", "A2", "A3", "A4", "A6", "A7", "A8", "A9"]
depends_on = ["WP-009", "WP-011", "WP-014"]
adds_dependencies = []

[checked]
A0 = "a8c098e1e3e8427f42c9d233ce646b15b1b66088"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-018: the engine at N = 2 on the CPU backend: grid, guards, shapes, gather, deposit, sort, advance and loading, with the tensor-product Reference deposition measured first and N = 1 kept bit for bit

This is item 1 of the 2D3V slice (`docs/plan/architecture-draft.md` §0.8) and the highest-risk package of it. It retires the §0.7 thin spot that the tensor-product Reference deposition is the unmeasured main cost of N > 1 with an undecided kernel mechanism, by measuring before porting. It then fixes every N = 2 engine signature in code, so that the later packages (the 2D flux deposition, the periodic mesh solve, the module's 2D step) are cut against signatures that exist, not guessed ones (A0: contracts from working code). Validation is by exact identities held to roundoff and by N = 1 digests recorded before the port, which is how WP-011 proved the split.

## Scope

`lib/ParticleMeshCore` (src and test), a new `scripts/engine2d/` for the measurements, a new report `reports/wp-018-engine-2d.md` with its figure directory `reports/wp-018-engine-2d/`, the §0.7 contract text in `docs/plan/architecture-draft.md` that the port settles, and `questions/` files. This package owns every N = 2 signature in the engine's grid, guards, shapes, gather, deposit, sort, position advance and loading. It implements one interface family, §0.7 contracts 1 to 10 and 12 generalized to `N`, and nothing else.

**Order of work.** The package runs in three phases, in this order, and the PR names the commit that ends each:

1. Pre-port digests (deliverable 0) and the mechanism measurement on prototype kernels inside `scripts/engine2d/` (deliverable 1), with no change to `lib/ParticleMeshCore/src`. The §0.7 proposal names the tensor-product Reference deposition as the main cost of N > 1 and leaves the kernel mechanism open (`CartesianIndex{N}`, `Val{N}`-unrolled tuples, explicit per-axis loops); the mechanism is chosen from this measurement.
2. The port (deliverables 2 to 4), with the chosen mechanism. The N = 1 methods may stay as they are if the generic mechanism would change their bits.
3. The measurements that need the ported code (deliverable 5: the engine-only proxy step at the CI test size and the scratch bytes), then the report, the budget question and the plan text.

A larger-size proxy-step study is out of this package; the PR opens it as a `follow-up` issue.

**The starting choices of §0.8 item 1, implemented as stated there:**

- `PeriodicGrid{N,T}` stays a plain immutable struct with `n::NTuple{N,Int}`, `dx::NTuple{N,T}` and one guard width `g::Int` for every axis. The later flux package checks its per-axis flux guard width against it. One shape order applies to every axis, so a per-axis guard tuple buys nothing now and would change the N = 1 struct. `PeriodicGrid(n, dx, g)` and the N = 1 accessors (`ncells`, `cellsize`, `guards`, `length`, `interior`, `interior_view`) keep their signatures and return types. A 2D constructor takes tuples and checks `n_d ≥ 2g` per axis. `field(backend, T, grid)` allocates `n .+ 2g`, column-major.
- A stagger for N = 2 is an `NTuple{2,Stagger}`; N = 1 keeps accepting a bare `Stagger`. One `Shape{P}` applies to every axis. The weights are the tensor product of `shape_weights` per axis, with `stencil_reach` per axis.
- `Species(...; N = 2)` with `cell::NTuple{2}` and `offset::NTuple{2}` arrays (the constructor already takes `N`). `advance` applies per axis, and the periodic wrap stays the single-device particle exchange.
- Cells linearize column-major (axis 1 fastest). The sort key is `linear cell · capacity + slot` (Int64, overflow checked). Segment bounds are per linear cell, free slots come after every cell, and extras follow the one permutation as in contract 9. `Workspace(species, grid)` holds the per-linear-cell bounds.
- `Reference()` deposition: a node adds the contributions of its stencil's cells in increasing linear cell index, within a cell in slot order; that order is the A4 bitwise contract for N = 2. `Fast()` adds with atomics. The accumulation trait and `effective_mode` are unchanged, so `Dual` takes the sorted path with no type-specific code (A8). Guard convention: deposit into guards, then fold.
- `fill_guards!(a, grid)` and `fold_guards!(a, grid)` keep contract 5's signature, run axis by axis with the corners included, and stay adjoint.
- `load_quiet!` for N = 2 takes `ppc::NTuple{2,Int}` on the per-axis product lattice `δ_d = (k_d − 1/2)/ppc_d`. The in-cell index `k` is linear with axis 1 fastest (`prod(ppc) < 2^16`, checked), the Philox counter is `(cell₁, cell₂, 0, k << 16)`, and `density(x::NTuple{2}, p)` gives weights `density/prod(ppc)`, so parameters enter through the weights only (A3). N = 1 keeps its counter `(cell, 0, 0, k << 16)` and its scalar-position `density(x, p)` call, so neither the electrostatic reference nor the module's `CosineProfile` changes, nor do their bits.
- Every new launch takes `workgroupsize = nothing` (contract 12).
- Every new N = 2 kernel stays within §0.4 rules 1 to 6 (static shapes, no scalar indexing outside kernels, allocation through `similar` or `KernelAbstractions.allocate`, no host-side branching on device data) and holds no float literal (`zero(T)`, `one(T)`, `T(1)/2`) (A8, A9).

`deposit_flux!` and `cumsum_periodic!` stay N = 1 and unchanged (§0.8 items 2 and 3). The slice is single-device and periodic.

**Collision rules with the packages running in parallel.** WP-008 wraps `EMState` internals and `step!(state, params, k)`. WP-008 and WP-015 both edit `lib/ParticleMeshElectromagnetic/Project.toml` and the shared Manifest, and WP-015 edits `pushers.jl`. This package adds no dependency, edits no `Project.toml` and no Manifest, never touches `pushers.jl`, and touches `lib/ParticleMeshElectromagnetic` only where keeping its 1D tests green requires. The module's N = 1 bits are checked by a before/after script rather than a new permanent test in the module, so WP-015's pusher refactor cannot be broken by a test this package adds. In `architecture-draft.md` it edits only §0.7, §0.8 item 1, and the port's own sentence in §0.8's ordering text, which stands on its own line, separated from WP-019's sentence by a line neither package edits. WP-019 deletes §0.8 item 4 and its own ordering sentence, different hunks.

## Deliverables

0. **Pre-port digests**, committed before any source change. A new digest file next to the existing ones in `lib/ParticleMeshCore/test/` holds, per element type (Float32, Float64, `Dual{Float32,3}`), the N = 1 outputs of `load_quiet!`, `sort_by_cell!` (permutation, segments, count), `deposit!` (both modes, orders 1 and 2, both staggers), `gather!`, `fill_guards!`/`fold_guards!` and `deposit_flux!` (Reference), together with the T-136 digest and the T-170 values. `scripts/engine2d/n1_bits.jl` records the electromagnetic module's T-218 state digests of each case and element type, and the 1D timings of `scripts/es1d/timing.jl`. It is run before and after the port, and the PR lists both runs.
1. **Mechanism measurement before the mechanism is fixed** (phase 1). `scripts/engine2d/timing.jl` (scripts environment, as `scripts/es1d/timing.jl`, no new dependency) times prototype 2D Reference deposition kernels written inside the script, one per candidate mechanism (`CartesianIndex{N}` stencil loop, `Val{N}`-unrolled per-axis tuples, explicit nested per-axis loops). Each stays within §0.4 rule 5 (index arithmetic, loads, stores, scalar math; no shared memory or warp intrinsics). They are timed against a prototype 2D Fast deposition and the existing 1D Reference deposition, per particle: orders 1 and 2, natural and non-natural staggers, Float32 and Float64, at least two small 2D sizes, at 1 thread and at the machine's full thread count (stated). Output: CPU ns per particle per mechanism and mode, CairoMakie PNGs with JSON sidecars and a run manifest (F3), the CPU named by model. This phase changes no file under `lib/`.
2. **Source** in `lib/ParticleMeshCore/src` (`grid.jl`, `shapes.jl`, `species.jl`, `guards.jl`, `transfers.jl`, `sort.jl`, `loading.jl`, `launch.jl` where needed): the N = 2 methods of every choice in scope, with the mechanism deliverable 1 picked. Generic in `T`, no float literals in kernels, every new kernel within §0.4, every launch taking `workgroupsize = nothing`, every N = 1 signature and return type unchanged.
3. **Tests** T-260 to T-269 in a new `lib/ParticleMeshCore/test/engine2d.jl` with its determinism worker cases. They run on non-square grids with different `n`, `dx` and `ppc` per axis, so that an axis transposition fails, over Float32, Float64 and `Dual{Float32,3}`, at small sizes (of the order of 16 × 12 cells and 2 × 3 particles per cell) stated in the test file.
4. **`lib/ParticleMeshCore/test/tolerances.toml`**: the measured `C` multiple of every new identity test as a `status = "proposed"` entry, scaled as each test states.
5. **Measurements on the ported code** (phase 3), by a second part of `scripts/engine2d/timing.jl`:
   - an engine-only proxy step (advance, sort, order-2 deposit, order-2 gather) at the CI test size, CPU time per particle-step at 1 thread and at the machine's full thread count;
   - the ported 2D Reference and Fast deposition at the sizes of deliverable 1, so the report shows that the chosen mechanism kept its prototype cost;
   - scratch bytes per particle and per cell of the 2D `Workspace`, with and without a two-array extras tuple (1D: 33 B per particle for Float32, `reports/spike-electrostatic-1d/`).

   Same output rules as deliverable 1 (F3).
6. **Report** `reports/wp-018-engine-2d.md` with figures in `reports/wp-018-engine-2d/`, following `reports/README.md`: the mechanism comparison and why the chosen one won; the 2D Reference cost relative to 2D Fast and to 1D Reference; the scratch per particle and per cell; the CPU time per particle-step of the proxy step; the added `ParticleMeshCore` CPU test wall time; and the N = 1 before/after digests and timings.
7. **`questions/wp-018-2d-cpu-budget.md`**: the measured CPU time per particle-step and the added test wall time, proposed as the A7 bound for the CI-sized 2D tests until overview item 5 settles test budgets (A7: measure, then propose), with the assumption the package proceeded on.
8. **Plan text** in `docs/plan/architecture-draft.md`:
   - §0.7 contracts 1 to 10 and 12 restated for `N` from the code, each naming its test;
   - the §0.7 proposal sentence on the tensor-product deposition and `CartesianIndex{N}` replaced by the measured choice;
   - §0.8 item 1 deleted as done (current state only), and the port's sentence deleted from §0.8's ordering text, with what they settled carried in the PR description.

   No edit of other §0.8 items or of WP-019's ordering sentence.
9. **Trace**: `julia +1.13 --project=scripts scripts/trace/build.jl --check` passes; the PR lists the plan sections changed, the follow-up issue for the larger-size proxy study, and a per-kernel checklist of the new N = 2 kernels against §0.4 rules 1 to 6 and the no-float-literal rule.

## Non-goals

- No 2D `deposit_flux!`, no Esirkepov in 2D, no per-axis old-position extras, no per-axis flux guard width (§0.8 item 2).
- No 2D periodic mesh solve and no 2D `cumsum_periodic!` (§0.8 item 3).
- No electromagnetic code: no 2D Yee, no change to `EMState`, `EMProblem`, `step!(st, p, k)`, `initialize!` or the field layout (§0.8 item 5). Never `pushers.jl` (WP-015). No edit of `lib/ParticleMeshElectromagnetic` beyond what keeping its 1D tests green requires, and no new test in it.
- No new dependency and no edit of any `Project.toml` or `Manifest.toml` (WP-008 and WP-015 edit those).
- The slice is single-device and periodic: no bounded axes, no dynamic grid state of B1, no `gstate` argument, no `exchange_particles!`, no multi-device signatures of §3. §0.7 contract 5's signature `fill_guards!(a, grid)` stands and the grid stays a plain struct, so a later window field changes no kernel signature.
- No per-axis guard widths, no per-array accumulator element types (B5), no cubic shapes.
- No GPU code, no CUDA path, no GPU run or test (WP-013 owns the CUDA determinism work); no Reactant dependency (A9).
- No N = 3 tests: code may be generic in `N`, but only N = 1 and N = 2 are claimed and tested.
- No proxy-step study beyond the CI test size (a follow-up issue).
- No velocity quiet start; no edit of `scripts/crosscode/` or of any case file (WP-019, WP-012, WP-017); no edit of `docs/goals/`.
- No change of any N = 1 signature or of any N = 1 bit; no change of `deposit_flux!` or `cumsum_periodic!`.
- No accepted tolerance or budget and no throughput threshold asserted in a test: every `C` and the CPU-time figures are proposals for the owner.

## Acceptance

Test ids T-260 to T-269. Every 2D test runs Float32, Float64 and `Dual{Float32,3}` where it applies.

- T-110 to T-177 and the electromagnetic module's T-210 to T-221 pass unchanged in name and assertion (T-144's existing `@test_broken` naming #82 stays as it is), at 1 and 4 threads. T-171 (the engine holds no physics) passes with the new exports.
- T-260 the port keeps N = 1 bit for bit [A0, A4, WP-018]: the pre-port digests of deliverable 0 (the new 1D engine digest, T-136 and T-170) are reproduced exactly after the port, per element type, at 1 and 4 threads. `n1_bits.jl` shows the module's T-218 digests equal before and after, and the 1D timings unchanged within the run-to-run spread, both runs in the PR.
- T-261 the 2D grid, linearization and advance [A7, A8, WP-018]: field size `n .+ 2g`, the interior index mapping, the column-major linear cell index and its inverse, and the per-axis `n_d ≥ 2g` check. Free streaming at dyadic displacements is exact per axis, through each seam and diagonally through the corner; offsets stay in [0, 1); a `Dual` offset carries the partials of the continuous trajectory with integer cells.
- T-262 2D shape weights [A2, A7, A8, WP-018]: the tensor-product weights sum to one to C·eps(T) for orders 1 and 2 and every stagger pair, mixed pairs such as (Center, Node) included. A single particle's 2D weights equal the outer product of the 1D weights bitwise. The weight function is `@inferred` and returns element type `T` for Float32 and `Dual{Float32,3}`, which a Float64 literal would break.
- T-263 2D fill and fold [A7, A8, WP-018]: `fill_guards!` copies periodic images bitwise, corners included; ⟨fill(a), b⟩ = ⟨a, fold(b)⟩ over the full arrays to C·eps(T)·size, for unequal cell counts per axis.
- T-264 2D gather [A7, A8, WP-018]: constant fields are reproduced, and a + b·x + c·y + d·x·y away from the seam, at every alive particle to C·eps(T) times the field scale, guards filled, for orders 1 and 2 and every stagger pair. The per-particle gather is `@inferred` with element type `T` for Float32 and `Dual{Float32,3}`.
- T-265 2D deposition [A3, A7, A8, WP-018]: the deposited total equals Σ quantity to C·eps(T)·count·max|quantity| for the default `ChargeWeight()` and a module-supplied quantity. Deposition with fill and fold is the adjoint of `gather!` with the same weighting to roundoff. `Dual` weights give exact partials with primal values bitwise equal to the Float32 run. Reference and Fast agree per node within C·eps(T)·Σ|contributions|.
- T-266 2D sort and Reference determinism [A4, WP-018]: the permutation sorts by column-major linear cell then slot, free slots last; keys are unique; per-linear-cell segment bounds and the alive count are correct with dead slots present; two extras of different element types still match their particle bitwise after the sort. Reference deposition is bitwise identical across reruns, at 1 and 4 threads and for the default and two explicit `workgroupsize` values, per element type, with particles crossing both seams and the corner.
- T-267 2D loading [A3, A4, A8, WP-018]: the product lattice with unequal per-axis `ppc` places `prod(ppc)` particles per cell at the documented offsets. The Philox counter is `(cell₁, cell₂, 0, k << 16)`, checked against values the test computes from the known-answer `philox4x32`; the N = 1 counter is unchanged. The weights equal `density(x, p)/prod(ppc)` at the lattice points, and the total weight equals the lattice sum of an oblique cosine density to roundoff. Positions are bitwise independent of `p`, so a `Dual` parameter appears only in the weights. The same seed gives the same bits at 1 and 4 threads, and different streams differ.
- T-268 allocation does not grow in 2D [A7, A9, WP-018]: sort, deposit (both modes), gather, fill, fold and load allocate no more per call at two particle counts and two grid sizes after warm-up, with and without extras, measured as T-144 does (the static-shape side of §0.4).
- T-269 2D reduces to 1D [A7, WP-018]: a 2D load constant along y, deposited in Reference mode and summed over y, equals the 1D deposit of the same particles to C·eps(T)·size; the gather of a y-constant field equals the 1D gather to C·eps(T).
- Every new N = 2 kernel is within §0.4 rules 1 to 6 and has no float literal: the PR description carries the per-kernel checklist of deliverable 9, and T-262, T-264, T-268 and T-171 cover the checkable parts [A8, A9].
- The mechanism timing of deliverable 1 is committed before the first commit that changes `lib/ParticleMeshCore/src` (both named in the PR). Its PNGs, JSON sidecars and manifest, and those of deliverable 5, are committed with the report, and the kernel mechanism in the code is the one the report chooses.
- The report, sidecars, manifests and PR name the CPU by model and contain no hostname, ssh alias, machine user name or path outside `~`; the runs use Julia 1.13 (A6).
- The new tests' measured `C` multiples are in `tolerances.toml` as proposed. The scratch bytes per particle and per cell and the added CPU test wall time are in the report and the PR. `questions/wp-018-2d-cpu-budget.md` exists and the PR points to it.
- No `Project.toml` or `Manifest.toml` changed and `pushers.jl` untouched (checked on the diff). `build.jl --check` passes and CPU CI is green. Tier 1.

## Depends on

- WP-011: the engine contracts of §0.7 this package generalizes, the extras, the quantity callable and the launch keyword, and the T-136 and T-170 baselines.
- WP-009: `deposit_flux!`, whose N = 1 bits the pre-port digest records and the port must keep.
- WP-014: the electromagnetic module, whose 1D tests and T-218 digests the port must keep.

WP-018 and WP-019 run in parallel with each other and with WP-008 and WP-015. The 2D flux deposition, the periodic mesh solve and the module's 2D step (§0.8 items 2, 3 and 5) build on this package. Test ids T-260 to T-269; WP-019 takes T-270 onwards.

## Budgets

A7 exact identities: every new test asserts C·eps(T) scaled by the problem size as the test states, with C measured and recorded as "proposed" in `lib/ParticleMeshCore/test/tolerances.toml`. Allocation per call is flat across particle counts and grid sizes (T-268).

Measured, not thresholded:

- CPU ns per particle of the 2D deposition per mechanism and mode, and the 2D/1D Reference cost ratio;
- the CPU time per particle-step of the engine-only proxy step at the CI test size, at 1 thread and the machine's full thread count;
- scratch bytes per particle and per cell of the 2D `Workspace` (1D: 33 B per particle for Float32);
- the CPU named by model.

The CPU time per particle-step and the added test wall time are proposed in `questions/wp-018-2d-cpu-budget.md` as the bound on CI-sized 2D tests (overview item 5 is open; the throughput budget of A7 is on the GPU, later). CPU test time: 2D test sizes of the order of 16 × 12 cells and 2 × 3 particles per cell, with the added `ParticleMeshCore` test wall time measured and stated in the PR. No GPU. Dependencies: none added (F2). Tier 1.
