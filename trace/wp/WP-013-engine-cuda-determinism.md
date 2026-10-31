+++
id = "WP-013"
status = "proposed"
goals = ["A4", "A6", "A7", "A8", "A9"]
depends_on = ["WP-011"]
adds_dependencies = ["CUDA"]

[checked]
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A6 = "344832ac0e32f874b8cbaa674a2282ee40a05d9c"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
A9 = "d30c76a6cdd40296c596e59d188ed4da9d2fb621"
+++
# WP-013: the engine on the RTX 4080 SUPER: CUDA identities, a deterministic scan, Reference mode across launch shapes, Fast against Reference throughput

The GPU is the project's target, and nothing has run on it yet. The two riskiest claims, a scan that is bitwise across launch shapes and a sorted deposition that does not depend on the workgroup size, were flagged by the spike, and the earlier they fail, the cheaper the fix. Doing this right after the split, before a second module depends on the engine, keeps any fix inside one package. Ownership is split cleanly: WP-011 owns the scan's signature and association order and the launch keyword; this package owns the scan's CUDA implementation and the CUDA runs that vary the keyword. Workgroup-size independence of the sort and the sorted deposition is a measured result here and a proposal to the owner, not a contract, because A4 defines bitwise reproducibility over reruns and thread counts only. This is the first of the two packages that add Julia dependencies.

## Scope

The backend contract of the engine. The same KernelAbstractions kernels, selected by array type, run on CUDA. The engine and the electrostatic reference run on the RTX 4080 SUPER through the machine's GPU runner, under `PM_TEST_GPU=1` and `scripts/gpu-lock.sh`, with no hostname in anything committed (A6: public repository, GPU tests on the local GPU runner). The engine gains no dependency.

The package is validation-first: the CPU identity tests run unchanged on `CUDABackend`. It measures, per element type, whether Reference mode on CUDA is bitwise across reruns (A4's definition) and across two KernelAbstractions workgroup sizes for the sort, the sorted deposition and the scan. Agreement across backends is measured, not assumed. For `cumsum_periodic!`, independence of the workgroup size follows from the association order that WP-011 fixed (tiles of a constant size, serial within a tile, tile totals combined serially in tile order), which the CUDA implementation must reproduce exactly; for the sort and the sorted deposition it is a measured result. The package writes the GPU reading of A4 (bitwise across workgroup sizes as well) as a question for the owner and keeps §0.7 to what T-191 and T-192 prove until the owner accepts it in A4.

It uses the launch keyword `workgroupsize` that WP-011 defined for every engine launch and varies it on the kernels that exist when it runs. `deposit_flux!` (WP-009) is in that set if WP-009 has merged before this package's runs; otherwise this package opens a `follow-up` issue, named in the PR, for the CUDA run of `deposit_flux!` with the same tests (T-190 to T-192 applied to T-110, T-112 and T-113). Finally, it measures Fast against Reference throughput and memory as the first numbers of A7's throughput budget.

## Deliverables

1. **Dependency**: CUDA.jl as a test dependency of `lib/ParticleMeshCore/test` and of `scripts/` only, with compat entries. There is no engine dependency: KernelAbstractions selects the backend. The shared root Manifest and `scripts/Manifest.toml` are re-resolved, and the PR states how much they grow (F2).
2. **The CUDA scan**: the CUDA path of `cumsum_periodic!` as a deterministic multi-work-item scan that implements WP-011's tiled association order exactly, with no signature change: AcceleratedKernels' scan only if its association is that order and independent of the launch, otherwise a fixed-tile KernelAbstractions scan. The CPU keeps the one-work-item path.
3. **GPU tests**: `lib/ParticleMeshCore/test/gpu.jl`, included only when `PM_TEST_GPU=1` and run through `scripts/gpu-lock.sh`, with `CUDA.allowscalar(false)` for the whole file. Every identity test runs on `CUDABackend` for Float32, Float64 and `Dual{Float32,3}`, and the launch-shape tests pass explicit values of WP-011's `workgroupsize` keyword.
4. **Timing**: `scripts/es1d/timing.jl` gains a backend option. It reports Fast and Reference ns per particle-step for the whole step and for the sort, the deposition and the solve separately, at 2^16, 2^20 and 2^24 particles, for Float32 and Float64, with the GPU synchronized around the timed region. It also records device memory per particle. Output: the standard plot, JSON sidecars and a run manifest with the commit and input hashes (F3).
5. **Report**: `reports/gpu-electrostatic-1d.md`: what ran, with the GPU model named and no hostname (A6); the bitwise results; throughput and memory as A7 proposals; the measured CPU–CUDA differences; and the measured launch-shape results with the proposed GPU reading of A4.
6. **Question**: `questions/wp-013-a4-gpu-reading.md` asks the owner whether A4 should also require bitwise reproducibility across workgroup sizes on one backend, with the T-192 results as the evidence and the assumption proceeded on (measured and reported, not a contract). §0.7 gains only the measured statement "on CUDA, Reference mode was bitwise across reruns (T-191) and across two workgroup sizes (T-192)" and points to the question; it does not state the stronger guarantee as a contract.
7. **Runner request**: the exact request in the PR, a Julia 1.13 command family with the GPU (`gpu: yes`) that runs the GPU tests and the timing script through `gpu-lock.sh`. If the runner lacks it, `questions/wp-013-gpu-runner.md` is written, the CPU-side scaffolding is merged, and the runs are marked absent in the report, as WP-003 did.

## Non-goals

- No CUDA or GPU dependency of the engine package itself, and no CUDA-specific code in `lib/ParticleMeshCore/src` beyond what KernelAbstractions dispatch selects. No ROCm or AMD run, no rented GPU.
- No change to the signatures of WP-011's contracts, its scan order or its launch keyword. No `deposit_flux!` implementation or electromagnetic step on the GPU (the flux's CUDA run is the follow-up named above when WP-009 is not merged yet), no N > 1, no Double32.
- No edit of `docs/goals/`: the GPU reading of A4 goes to the owner as a question.
- No warp intrinsics or shared-memory tricks in the Reference kernels (§0.4 rule 5). An optimized Fast variant may exist behind the same interface only if it is measured and documented.
- No throughput threshold asserted, because A7 asks for a factor of Smilei on the same card, and the Smilei GPU reference is cross-code stage 2.
- No Reactant (A9). No GPU run outside the runner, no two GPU jobs at once, and no CUDA attempt inside a sandboxed session.

## Acceptance

Test ids T-190 to T-199.

- T-190 the CPU identity tests T-130 to T-135, T-137, T-138, T-140, T-172, T-174 and T-175 pass on `CUDABackend` for Float32, Float64 and `Dual{Float32,3}` with the CPU's proposed C. A larger measured ratio on the GPU is recorded as its own proposed entry, never as a silent widening [A7, A8, WP-013].
- T-191 Reference mode on CUDA is bitwise identical across three reruns for one step and for the short Landau run of T-141 (a digest of the final fields and particles), per element type [A4, WP-013].
- T-192 `sort_by_cell!`, Reference `deposit!` and `cumsum_periodic!` on CUDA are bitwise identical across two workgroup sizes (for example 64 and 256), per element type; `cumsum_periodic!` on the device is bitwise equal to the serial implementation of WP-011's tiled order run on the device, and satisfies T-172's identities; the reference's Gauss law holds after every step. For the sort and the deposition the result is reported as a measurement for the A4 question [A4, A7, WP-013].
- T-193 the Philox integer outputs and `uniform_open` on CUDA are bitwise equal to the CPU's. The Box–Muller velocities agree within a measured number of ulp (CUDA's `log` and `sincos` may differ from Base), recorded as a proposal [A4, WP-013].
- T-194 Fast and Reference on CUDA agree per node within a measured bound (atomic order is not deterministic), recorded as a proposal next to T-135's [A4, A7, WP-013].
- T-195 CPU and CUDA Reference runs of the Landau and two-stream cases differ in rates and field-energy histories by a measured relative amount, recorded as A4's cross-backend tolerance proposal, not bitwise; the CPU–CUDA difference of `cumsum_periodic!` alone is recorded too, since both run one association order [A4, A7, WP-013].
- T-196 allocation per step after warm-up on CUDA does not grow with particle count [A7, WP-013].
- Without `PM_TEST_GPU`, the suite is unchanged and CPU CI stays green. With it, the runner's log ends with `# gpu-run finished:` and the tests pass, or the runs are marked absent and the question file exists. `questions/wp-013-a4-gpu-reading.md` exists. `build.jl --check` passes.

## Depends on

WP-011 (the contracts this package runs on CUDA: `cumsum_periodic!` with its association order, extras, the deposited quantity and the launch keyword). It does not depend on WP-009; the CUDA run of `deposit_flux!` is handled as stated in the scope.

## Budgets

A7 throughput, first measurement: ns per particle-step on the RTX 4080 SUPER for Fast and Reference, Float32 and Float64, at three particle counts, proposed and not asserted. Memory: bytes per particle of the species plus the `Workspace` on the device, next to the CPU figures (33 B Float32, 37 B Float64), and the largest particle count that fits in 16 GB at Float32. A4: bitwise per element type per backend across reruns, as A4 defines it; bitwise across workgroup sizes is measured and proposed through the question file; the CPU–CUDA agreement is a measured tolerance. Tier 1: `lib/ParticleMeshCore/test/Project.toml` is not a code-owner path.
