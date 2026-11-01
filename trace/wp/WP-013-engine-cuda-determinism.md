+++
id = "WP-013"
status = "proposed"
goals = ["A2", "A4", "A5", "A6", "A7", "A8", "A9"]
depends_on = ["WP-008", "WP-009", "WP-011", "WP-014", "WP-018"]
adds_dependencies = ["CUDA"]

[checked]
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-013: the engine and the electromagnetic module on the RTX 4080 SUPER: the CPU identities on CUDA, Reference mode bitwise across reruns and launch shapes, first throughput and memory

The GPU is the project's target, and nothing has run on it yet. Every kernel of the engine and of the electromagnetic module is a KernelAbstractions kernel selected by the array type, so running on CUDA should need no new package and no CUDA code in `src`; this package proves that and fixes what the GPU finds. It comes now because two modules' worth of code (WP-014's step, WP-008's harness) and the 2D engine (WP-018) already sit on the engine, and every later package (the 2D flux, the module's 2D step, LWFA) would otherwise inherit untested GPU assumptions. This is the package that adds CUDA.jl, to tests and scripts only.

## Scope

The CUDA backend of what exists: the engine at N = 1 and N = 2, including `deposit_flux!` (WP-009), and the electromagnetic module's 1D step and its SciML harness (WP-014, WP-008). The same kernels run on `CUDABackend` on the RTX 4080 SUPER, with no hostname in anything committed (A6). Neither `lib/ParticleMeshCore` nor `lib/ParticleMeshElectromagnetic` gains a dependency.

Validation-first. The CPU identity tests run on CUDA from the same test bodies, for Float32, Float64 and `Dual{Float32,3}`. Reference mode on CUDA is tested against A4 as it now reads: bitwise per element type across reruns and launch shapes. For the engine's own kernels this follows from their construction (one work item per output, a fixed summation order; unique sort keys, so the permutation does not depend on the algorithm; `cumsum_periodic!` runs as one work item in its documented order), so the tests confirm the contract rather than discover it. Agreement across backends is measured, not assumed (A4).

The GPU tests live in the packages' own test folders and run automatically wherever a functional GPU is present, so the GPU runner, the owner's machine and any later GPU CI job run them with a plain `Pkg.test()`, and CPU CI skips them and says so.

The electrostatic reference (`lib/ParticleMeshCore/test/es_reference.jl`, `scripts/es1d/`) is not run on CUDA: it is test-support code on its way out (§0.7; its deletion is follow-up #97), and the module reproduces the Landau and two-stream cases (WP-014).

## Deliverables

1. **Dependency**: CUDA.jl as a test dependency of `lib/ParticleMeshCore/test` and `lib/ParticleMeshElectromagnetic/test`, and of `scripts/`, with compat entries; no package dependency. The shared root Manifest and `scripts/Manifest.toml` are re-resolved together, and the PR states how much they grow and what CUDA.jl adds to a CPU-only test run: its precompile and load time, and whether `CUDA.functional()` downloads an artifact on a machine without a driver (F2). If the added CPU time is large, the PR says so and writes it as a question; it does not move the GPU tests out of the test folders.
2. **The GPU switch**, in each package's `test/setup.jl`, read once: `PM_TEST_GPU` unset runs the GPU testsets if and only if `CUDA.functional()`; `PM_TEST_GPU=1` requires them (no functional GPU is a test failure, so the runner can never skip silently); `PM_TEST_GPU=0` turns them off. When they do not run, the suite prints one line saying so and why. The GPU testsets run under `CUDA.allowscalar(false)`.
3. **Shared test bodies**: the CPU tests named in the acceptance are factored, where they are not already (`engine_cases.jl`, `flux_cases.jl`, `engine2d_cases.jl` and `determinism_cases.jl` are partly there), into functions of the backend, called once with `CPU()` under their existing ids and once with `CUDABackend()` under this package's ids from `test/gpu.jl`, which each `runtests.jl` includes. Device results are copied to the host with `Array` before they are compared. The CPU tests keep their ids, tolerances and bits (T-170, T-260 and T-213 still pass unchanged).
4. **Fixes the GPU finds**, in `src` and in the shared test bodies: host-side scalar indexing, allocations that grow with particle count, types that do not cross to the device (`Adapt`), launches that assume the CPU. Each fix keeps the CPU bits (the CPU suite unchanged) and is named in the PR. A suspected bug in CUDA.jl, KernelAbstractions, AcceleratedKernels or ForwardDiff goes to the `upstream-reporter` with a workaround here (CLAUDE.md, "Issues").
5. **Timing**: `scripts/em1d/timing.jl` and the `ported` part of `scripts/engine2d/timing.jl` gain a backend option. On CUDA they report Fast and Reference ns per particle-step for the module's three cases (Landau, two-stream, Weibel) and for the 2D engine proxy step, at 2^16, 2^20 and 2^24 particles, for Float32 and Float64, with the GPU synchronized around every timed region, and the share of the step spent in the sort, the deposition (`deposit!`, `deposit_flux!`), the field update and `cumsum_periodic!` where they can be timed separately. They record device memory per particle (species, `Workspace`, module state, and the state the `FunctionMap` harness doubles, A7) and the largest particle count that fits in 16 GB at Float32. Output: the standard plots, JSON sidecars and a run manifest with the commit, the input hashes and the GPU model (F3).
6. **Report**: `reports/gpu-cuda.md`: what ran on which GPU model, with no hostname (A6); the bitwise results; the measured Fast–Reference and CPU–CUDA differences; throughput and memory as A7 proposals, with the RTX 4080 SUPER's low Float64 rate stated next to the Float64 numbers; and every fix of deliverable 4.
7. **Session rules**: `CLAUDE.md` ("Working here", Tests) says that the GPU tests run wherever a functional GPU is present, how `PM_TEST_GPU` forces them on or off, that a sandboxed session requests them from the machine's GPU runner, and that runs outside the runner still go through `scripts/gpu-lock.sh`; the usage line of `scripts/gpu-lock.sh`, `scripts/README.md` and `docs/plan/factory-and-observability.md` §4 match. `CLAUDE.md` and `factory-and-observability.md` are tier 2.
8. **Runner request**: the exact request in the PR (one `julia +1.13 --project=lib/<Name> -e 'using Pkg; Pkg.test()'` per package and one per timing script, `network: no` unless CUDA.jl must fetch its artifacts on the first run). If no runner is listed, `questions/wp-013-gpu-runner.md` is written, the CPU-side work is merged, and the runs are marked absent in the report, as WP-003 did.

## Non-goals

- No CUDA code and no GPU dependency in `lib/*/src` beyond what KernelAbstractions dispatch selects. No ROCm or AMD run, no rented GPU.
- No parallel scan: `cumsum_periodic!` runs on CUDA as its existing one-work-item kernel. The module calls it once, for the initial field at t = 0, so its cost is outside the step; the timing records it, and a parallel kernel of the same order is a later package if a profile ever shows it.
- No change to the signatures of the engine contracts, the scan order or the launch keyword (§0.7).
- No electrostatic reference on CUDA and no edit of `scripts/es1d/` (#97 owns its retirement).
- No module step at N = 2 (it does not exist yet), no N = 3, no Double32.
- No warp intrinsics or shared-memory tricks in the Reference kernels (§0.4 rule 5). An optimized Fast variant may exist behind the same interface only if it is measured and documented.
- No throughput threshold asserted: A7 asks for a factor of Smilei on the same card, and the Smilei GPU reference is cross-code stage 2.
- No Reactant (A9).

## Acceptance

Test ids T-190 to T-199. On CUDA, "with the CPU's C" means the tolerance constant the CPU test uses (`tolerances.toml`); a larger measured ratio on the GPU is recorded as its own proposed entry, never as a silent widening.

- T-190 the engine's N = 1 identities on CUDA, for Float32, Float64 and `Dual{Float32,3}`, with the CPU's C: T-130 to T-135, T-139, T-140, T-172 to T-175, and the flux identities T-110, T-112, T-114 and T-115 [A7, A8, WP-013].
- T-191 the engine's N = 2 identities on CUDA, for the same element types, with the CPU's C: T-261 to T-265, T-267 and T-269 [A7, A8, WP-013].
- T-192 Reference mode of the engine on CUDA is bitwise identical across three reruns and three launch shapes (`workgroupsize = nothing`, 64 and 256), per element type, for `sort_by_cell!`, `deposit!`, `deposit_flux!`, `gather!`, `fill_guards!`, `fold_guards!` and `cumsum_periodic!`, at N = 1 and N = 2 with particles crossing every seam (the CUDA counterpart of T-136, T-177, T-113 and T-266) [A4, WP-013].
- T-193 the Philox integer outputs and `uniform_open` on CUDA are bitwise equal to the CPU's, and so is a quiet-start load; the Box–Muller velocities agree within a measured number of ulp (CUDA's `log` and `sincos` may differ from Base), recorded as a proposal [A4, WP-013].
- T-194 Fast and Reference on CUDA agree per node and per face within the CPU's C for `deposit!` and `deposit_flux!` at N = 1 and `deposit!` at N = 2 (atomic order is not deterministic), the measured ratios recorded next to T-135's, T-113's and T-265's [A4, A7, WP-013].
- T-195 the module on CUDA passes its CPU tests with the CPU's tolerances: T-210, T-211, T-212 (Gauss's law after every step), T-214 to T-217, and T-219 in Float64 [A2, A5, A7, A8, WP-013].
- T-196 the module's Reference step on CUDA is bitwise identical across three reruns and two launch shapes for the Landau, two-stream and Weibel cases (a digest of the final fields and particles), per element type, and the harness's `solve` equals `initialize!` plus `run!` bit for bit on CUDA (the CUDA counterpart of T-218 and T-100) [A4, A9, WP-013].
- T-197 CPU and CUDA Reference runs of the module's three cases differ in their rates and field-energy histories by a measured relative amount, recorded as A4's cross-backend tolerance proposal, not bitwise; the CPU–CUDA difference of `cumsum_periodic!` alone is recorded too, since both backends run one association order [A4, A7, WP-013].
- T-198 host and device allocation per step after warm-up on CUDA does not grow with particle count or grid size, for the engine at N = 1 and N = 2 and for the module's step (the CUDA counterpart of T-117, T-268 and T-220) [A7, WP-013].
- T-199 the GPU switch, tested on the CPU in a subprocess: without a functional GPU and without `PM_TEST_GPU` the GPU testsets do not run and the suite prints the one line; with `PM_TEST_GPU=0` they do not run; with `PM_TEST_GPU=1` and no functional GPU the suite fails [A6, WP-013].
- On CPU CI the suites pass unchanged apart from T-199 and the one line. Through the runner, the log ends with `# gpu-run finished: exit 0` and T-190 to T-198 pass, or the runs are marked absent and the question file exists. `build.jl --check` passes.

## Depends on

WP-011 (the engine contracts, the launch keyword), WP-009 (`deposit_flux!`), WP-018 (the engine at N = 2), WP-014 (the module's step and its tests) and WP-008 (the harness); all are done.

## Budgets

A7 throughput, first measurement: ns per particle-step on the RTX 4080 SUPER for Fast and Reference, Float32 and Float64, at three particle counts, for the module's 1D step and the 2D engine proxy step, proposed and not asserted. Memory: bytes per particle of the species, the `Workspace` and the module state on the device, with and without the harness, next to the CPU figures, and the largest particle count that fits in 16 GB at Float32. A4: bitwise per element type on CUDA across reruns and launch shapes, tested; the CPU–CUDA agreement is a measured tolerance. Tier 2: the `CLAUDE.md` and `factory-and-observability.md` edits of deliverable 7; the `lib/*/test/Project.toml` files are not code-owner paths.
