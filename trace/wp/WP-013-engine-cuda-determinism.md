+++
id = "WP-013"
status = "proposed"
goals = ["A4", "A6", "A7", "A8", "A9"]
depends_on = ["WP-011", "WP-014"]
adds_dependencies = ["CUDA"]

[checked]
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-013: the engine and the electromagnetic step on the RTX 4080 SUPER: CUDA identities, a deterministic scan, Reference mode across launch shapes, Fast against Reference throughput

The GPU is the project's target, and nothing has run on it yet. The two riskiest claims, a scan that is bitwise across launch shapes and a sorted deposition that does not depend on the workgroup size, were flagged by the spike, and the earlier they fail, the cheaper the fix. Doing this before the 2D module builds on the engine keeps any fix inside the engine and the 1D module. Ownership is split cleanly: WP-011 owns the scan's signature and association order and the launch keyword; this package owns the scan's CUDA implementation and the CUDA runs that vary the keyword. Workgroup-size independence of the sort and the sorted deposition is a measured result here and a proposal to the owner, not a contract, because A4 defines bitwise reproducibility over reruns and thread counts only.

## Scope

The backend contract of the engine. The same KernelAbstractions kernels, selected by array type, run on CUDA. The engine and the electromagnetic module's step (WP-014: the electrostatic cases as its edge cases with no transverse fields, and Weibel) run on the RTX 4080 SUPER through the machine's GPU runner, under `PM_TEST_GPU=1` and `scripts/gpu-lock.sh`, with no hostname in anything committed (A6: public repository, GPU tests on the local GPU runner). The engine and the module gain no dependency.

The package is validation-first: the CPU identity tests run unchanged on `CUDABackend`. It measures, per element type, whether Reference mode on CUDA is bitwise across reruns (A4's definition) and across two KernelAbstractions workgroup sizes for the sort, the sorted deposition and the scan. Agreement across backends is measured, not assumed. For `cumsum_periodic!`, independence of the workgroup size follows from the association order that WP-011 fixed (tiles of a constant size, serial within a tile, tile totals combined serially in tile order), which the CUDA implementation must reproduce exactly; for the sort and the sorted deposition it is a measured result. The package writes the GPU reading of A4 (bitwise across workgroup sizes as well) as a question for the owner and keeps §0.7 to what T-191 and T-192 prove until the owner accepts it in A4.

It uses the launch keyword `workgroupsize` that WP-011 defined for every engine launch, which the module's step passes through (`EMNumerics`), and varies it on every engine kernel, `deposit_flux!` (WP-009) included. Finally, it measures Fast against Reference throughput and memory as the first numbers of A7's throughput budget.

## Deliverables

1. **Dependency**: CUDA.jl as a test dependency of `lib/ParticleMeshCore/test`, `lib/ParticleMeshElectromagnetic/test` and `scripts/` only, with compat entries. There is no engine or module dependency: KernelAbstractions selects the backend. The shared root Manifest and `scripts/Manifest.toml` are re-resolved, and the PR states how much they grow (F2).
2. **The CUDA scan**: the CUDA path of `cumsum_periodic!` as a deterministic multi-work-item scan that implements WP-011's tiled association order exactly, with no signature change: AcceleratedKernels' scan only if its association is that order and independent of the launch, otherwise a fixed-tile KernelAbstractions scan. The CPU keeps the one-work-item path.

   Note on upstream options, from reading the sources (AcceleratedKernels 0.5.11; cuTile.jl's documentation), not measured:
   - **Default: a small fixed-tile KernelAbstractions kernel** that keeps WP-011's order exactly. One work item per tile does the serial sweep, then a serial scan of the tile totals and one offset per tile. For 1D grids of a few thousand cells the scan is a negligible part of the step.
   - **The alternative to measure, if the scan ever shows in a profile: AcceleratedKernels' `ScanPrefixes`.** It is already an engine dependency. It is reduce-then-scan with no block waiting for another, so with `block_size` and `items_per_thread` pinned its order is fixed by construction. It is not WP-011's order, for three reasons. Inside a tile it associates as a parallel tree, not as a serial sweep. Its CPU path (scan, then add the offsets) and its GPU path (reduce, then scan from the offset) associate differently, so the CPU would have to run the GPU-style kernel on the KA CPU backend to keep one association across backends. Its defaults come from each device's tuning, so both parameters must always be pinned. Adopting it means changing WP-011's documented order, which this package's non-goals rule out, so it would be a contract change in a PR of its own with T-172's serial reference rewritten to match.
   - **Not for the Reference path: cuTile.jl** (JuliaGPU's binding to NVIDIA's CUDA Tile model; it supports Ada and Julia 1.13). It is NVIDIA-only and not KernelAbstractions, which breaks §0.4. Its compiler decides how a tile is reduced, so it cannot give a documented association order. At most it could back an optimized native Fast-mode variant behind the same interface (§0.4 rule 5), more likely for tiled deposition than for the scan.
   - Tullio, TiledIteration and ThreadedScans are CPU loop tiling or CPU threads and do not apply. KernelIntrinsics.jl is registered but was not examined.
3. **GPU tests**: `lib/ParticleMeshCore/test/gpu.jl` for the engine and `lib/ParticleMeshElectromagnetic/test/gpu.jl` for the module's step, each included only when `PM_TEST_GPU=1` and run through `scripts/gpu-lock.sh`, with `CUDA.allowscalar(false)` for the whole file. Every identity test runs on `CUDABackend` for Float32, Float64 and `Dual{Float32,3}`, and the launch-shape tests pass explicit values of WP-011's `workgroupsize` keyword. The module's state is allocated on the device through `EMState(problem, numerics; backend)`; the host reads its histories and digests after a run, never inside a step.
4. **Timing**: `scripts/em1d/timing.jl` gains a backend option. It reports Fast and Reference ns per particle-step of the module's step for the whole step and for the sort, the current deposition (`deposit_flux!`) and the Yee solve separately, at 2^16, 2^20 and 2^24 particles, for Float32 and Float64, with the GPU synchronized around the timed region. It also records device memory per particle. Output: the standard plot, JSON sidecars and a run manifest with the commit and input hashes (F3).
5. **Report**: `reports/gpu-em-1d.md`: what ran, with the GPU model named and no hostname (A6); the bitwise results; throughput and memory as A7 proposals; the measured CPU–CUDA differences; and the measured launch-shape results with the proposed GPU reading of A4.
6. **Question**: `questions/wp-013-a4-gpu-reading.md` asks the owner whether A4 should also require bitwise reproducibility across workgroup sizes on one backend, with the T-192 results as the evidence and the assumption proceeded on (measured and reported, not a contract). §0.7 gains only the measured statement "on CUDA, Reference mode was bitwise across reruns (T-191) and across two workgroup sizes (T-192)" and points to the question; it does not state the stronger guarantee as a contract.
7. **Runner request**: the exact request in the PR, a Julia 1.13 command family with the GPU (`gpu: yes`) that runs the GPU tests and the timing script through `gpu-lock.sh`. If the runner lacks it, `questions/wp-013-gpu-runner.md` is written, the CPU-side scaffolding is merged, and the runs are marked absent in the report, as WP-003 did.

## Non-goals

- No CUDA or GPU dependency of the engine package itself, and no CUDA-specific code in `lib/ParticleMeshCore/src` beyond what KernelAbstractions dispatch selects. No ROCm or AMD run, no rented GPU.
- No change to the signatures of WP-011's contracts, its scan order or its launch keyword. No change of the module's step order or physics: a module kernel changes only where the CUDA run shows a defect, with its CPU tests unchanged. No N > 1, no Double32.
- No edit of `docs/goals/`: the GPU reading of A4 goes to the owner as a question.
- No warp intrinsics or shared-memory tricks in the Reference kernels (§0.4 rule 5). An optimized Fast variant may exist behind the same interface only if it is measured and documented.
- No throughput threshold asserted, because A7 asks for a factor of Smilei on the same card, and the Smilei GPU reference is cross-code stage 2.
- No Reactant (A9).

## Acceptance

Test ids T-190 to T-199.

- T-190 the CPU identity tests T-130 to T-135, T-138, T-140, T-110, T-112, T-113, T-172, T-174 and T-175 of the engine, and T-210 to T-212 of the module, pass on `CUDABackend` for Float32, Float64 and `Dual{Float32,3}` with the CPU's proposed C. A larger measured ratio on the GPU is recorded as its own proposed entry, never as a silent widening [A7, A8, WP-013].
- T-191 Reference mode on CUDA is bitwise identical across three reruns for one step of the module and for the short runs of T-218 (every case at 64 particles per cell over one time unit; the digest of T-218's `state_digest`: the particles, the fields and the histories), per element type [A4, WP-013].
- T-192 `sort_by_cell!`, Reference `deposit!` and `cumsum_periodic!` on CUDA are bitwise identical across two workgroup sizes (for example 64 and 256), per element type; `cumsum_periodic!` on the device is bitwise equal to the serial implementation of WP-011's tiled order run on the device, and satisfies T-172's identities; the module's Gauss law (T-212's residual) holds after every step. For the sort and the deposition the result is reported as a measurement for the A4 question [A4, A7, WP-013].
- T-193 the Philox integer outputs and `uniform_open` on CUDA are bitwise equal to the CPU's. The Box–Muller velocities agree within a measured number of ulp (CUDA's `log` and `sincos` may differ from Base), recorded as a proposal [A4, WP-013].
- T-194 Fast and Reference on CUDA agree per node within a measured bound (atomic order is not deterministic), recorded as a proposal next to T-135's [A4, A7, WP-013].
- T-195 CPU and CUDA Reference runs of the module on the Landau, two-stream and Weibel cases differ in rates and field-energy histories by a measured relative amount, recorded as A4's cross-backend tolerance proposal, not bitwise; the CPU–CUDA difference of `cumsum_periodic!` alone is recorded too, since both run one association order [A4, A7, WP-013].
- T-196 allocation per step of the module after warm-up on CUDA, and per call of every engine operation (T-144's set), does not grow with particle count [A7, WP-013].
- Without `PM_TEST_GPU`, the suite is unchanged and CPU CI stays green. With it, the runner's log ends with `# gpu-run finished:` and the tests pass, or the runs are marked absent and the question file exists. `questions/wp-013-a4-gpu-reading.md` exists. `build.jl --check` passes.

## Depends on

WP-011 (the contracts this package runs on CUDA: `cumsum_periodic!` with its association order, extras, the deposited quantity and the launch keyword) and WP-014 (the module's step, which composes them with `deposit_flux!`, the Boris push and the Yee solve, and is the integration run on the device).

## Budgets

A7 throughput, first measurement: ns per particle-step on the RTX 4080 SUPER for Fast and Reference, Float32 and Float64, at three particle counts, proposed and not asserted. Memory: bytes per particle of the species plus the `Workspace` on the device, and of the module's whole state, next to the CPU figures (33 B Float32, 37 B Float64), and the largest particle count that fits in 16 GB at Float32. A4: bitwise per element type per backend across reruns, as A4 defines it; bitwise across workgroup sizes is measured and proposed through the question file; the CPU–CUDA agreement is a measured tolerance. Tier 1: the test `Project.toml` files are not code-owner paths.
