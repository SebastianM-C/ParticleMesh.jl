# Spike: Enzyme reverse mode through the 2D step on the GPU

Enzyme differentiates steps of the 2D electromagnetic module on CUDA (RTX 5090). The case is Fast mode with one species. Its gradient with respect to the whole state matches ForwardDiff by the chain rule to 5e-15 (relative) in Float64 over 20 steps and 3e-13 over 100. A reverse pass is linear in the step count at CUDA's default 8 MB device heap. It takes 0.74 to 0.96 ms per step up to 262 144 particles, where the plain run is bound by kernel launches, and 10.4 ms per step at 4 194 304 particles. That is 9 to 15 times the plain run, and 0.3 to 3.9 times a Dual pass with 4 partials. Enzyme's first call compiles for 104 to 111 s.

Two engine changes make the cost linear. The gather and the 2D atomic block add are written out with `ntuple`, so Enzyme keeps their per-term values in its fixed-size tape. The 2D atomic adds go through the element's pointer (`atomic_add!` in `lib/ParticleMeshCore/src/flux2d.jl`). Without the changes, Enzyme allocated device memory once per work item and step, and those blocks stayed alive until the reverse sweep. CUDA's device malloc slows as live blocks accumulate, so a reverse pass grew quadratically in the steps. It overflowed the default heap past 16 steps at 720 particles, and failed past 4 steps at 262 144 particles even with a 1 GiB heap.

Not working:
- the CPU, because KernelAbstractions' CPU backend is POCL (#509);
- Reference mode without the hand-written rules of WP-077 (#527);
- two species (#384);
- Active scalar kernel arguments. These need a patch of KernelAbstractions' Enzyme extension, which also drops their adjoints.

| | |
|---|---|
| Work package | WP-050 (goals A0, A3, A7, A8) |
| Commit of the runs | `54cecc391b5ad2aa06fa387831934678771039a9` (clean tree) |
| Runs | the `enzyme-*` runs in `wp-050-enzyme-spike/`, each a `result.toml` and a `manifest.toml` |
| Input hashes | SHA-256 of `scripts/enzyme/{chain_rule,cost,common}.jl`, `lib/ParticleMeshElectromagnetic/test/em2d_cases.jl`, the test environment's `Project.toml` and the root `Manifest.toml`, in each manifest |
| Environment | Julia 1.13.4; Enzyme 0.13.221, KernelAbstractions 0.10.8, ForwardDiff 1.4.6 (the root manifest); NVIDIA GeForce RTX 5090, driver 615.71.09, CUDA backend; AMD Ryzen 9 9950X host |
| Status | measured, not independently reviewed; the cost timings are the median of three runs in one process, the chain-rule timings single runs |

## Setup

- **Case.** WP-023's `plasma_step_2d`: Esirkepov current, Quadratic shape, Boris pusher, Yee solver, electrons only, Fast mode, `observer = ()`. The default grid is 10 × 8 cells with 3 × 3 particles per cell. The parameter α is the density perturbation amplitude, which enters through the weights (A3). The functional f is the field energy after n steps.
- **Check** (`scripts/enzyme/chain_rule.jl`). Enzyme's gradient ∇f with respect to every floating-point array of the initialized state, dotted with ForwardDiff's tangent dx/dα, is compared against ForwardDiff's df/dα of the whole run. A second check uses a random tangent on every array. Both must agree within 64·eps(T)·n·c, where c is the cancellation of the per-array contributions. Both compare two AD modes of the same code, so a physics defect would not show; the module's T-tests cover the physics.
- **Cost** (`scripts/enzyme/cost.jl`). For each step count k, it times k plain steps, the same k steps with α as a `Dual` with 1 and with 4 partials, and Enzyme's reverse pass of the k steps. Each is the median of three runs after compilation. `scripts/enzyme/run_gpu.sh` runs both scripts, and `scripts/enzyme/table.jl` prints the tables below from the runs' `result.toml`.

KernelAbstractions 0.10's Enzyme extension rejects Active kernel arguments, the scalars such as `dt`, cell sizes and charges; the draft JuliaGPU/KernelAbstractions.jl#979 adds them. The runs load `scripts/enzyme/spike/active_args.jl` (`PM_ENZYME_PATCH`), which passes them as Const. The gradient with respect to the state is unaffected, but a parameter that reaches a kernel only as a scalar argument gets no gradient.

## Results

| Mode | Steps | T | Seed | α direction, relative | Random direction, relative | c | Verdict | Reverse (s) | Primal (s) | First call (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 20 | Float64 | 7 | 5.4e-15 | 2.1e-16 | 4.36 | pass | 0.319 | 0.0020 | 110 |
| fast | 20 | Float32 | 7 | 3.0e-06 | 2.1e-07 | 4.36 | pass | 0.312 | 0.0013 | 111 |
| fast | 100 | Float64 | 3 | 2.7e-13 | 1.1e-13 | 21.43 | pass | 0.373 | 0.0084 | 110 |

| T | Cells | ppc | Particles | Steps | Primal (ms) | Dual{1} (ms) | Dual{4} (ms) | Reverse (ms) | Reverse / primal | Reverse / Dual{4} | Reverse per step (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Float32 | 10×8 | 3×3 | 720 | 1 | 0.07 | 0.10 | 7.03 | 1.26 | 19.3 | 0.18 | 1.256 |
| Float32 | 10×8 | 3×3 | 720 | 64 | 3.14 | 5.49 | 12.28 | 47.49 | 15.1 | 3.87 | 0.742 |
| Float64 | 10×8 | 3×3 | 720 | 1 | 0.09 | 0.26 | 0.66 | 1.23 | 13.5 | 1.86 | 1.230 |
| Float64 | 10×8 | 3×3 | 720 | 64 | 5.24 | 18.74 | 52.68 | 47.79 | 9.1 | 0.91 | 0.747 |
| Float32 | 64×64 | 3×3 | 36864 | 1 | 0.06 | 0.13 | 0.43 | 1.22 | 20.1 | 2.82 | 1.223 |
| Float32 | 64×64 | 3×3 | 36864 | 32 | 1.63 | 4.36 | 15.47 | 24.67 | 15.1 | 1.59 | 0.771 |
| Float32 | 256×256 | 2×2 | 262144 | 1 | 0.09 | 0.25 | 1.79 | 1.34 | 15.2 | 0.75 | 1.343 |
| Float32 | 256×256 | 2×2 | 262144 | 16 | 1.53 | 3.71 | 34.94 | 15.33 | 10.0 | 0.44 | 0.958 |
| Float32 | 1024×1024 | 2×2 | 4194304 | 1 | 0.88 | 3.35 | 30.15 | 9.93 | 11.3 | 0.33 | 9.933 |
| Float32 | 1024×1024 | 2×2 | 4194304 | 8 | 7.83 | 26.30 | 290.99 | 83.26 | 10.6 | 0.29 | 10.407 |

First reverse call (compiles), s: Float32 10×8 110; Float64 10×8 111; Float32 64×64 110; Float32 256×256 110; Float32 1024×1024 104

At the small sizes the plain run is bound by kernel launches, so its time barely changes from 720 to 262 144 particles. The ratios at those sizes say little about production runs; the 4 194 304-particle case (1024 × 1024 cells, 2 × 2 per cell) is the closest to one.

The Float32 10×8 Dual{4} time at one step (7.03 ms) is an outlier of the run, against 0.38 ms at two steps. The earlier run at commit `21d1cc5`, with the unrolled stencils but Atomix's adds, is in this report's commit message. It shows the quadratic growth.

## Why the reverse pass allocated

Enzyme's augmented forward pass of a kernel saves, per work item, the values its reverse needs. Values saved inside a loop go to `malloc`, which on CUDA is the device heap, even when the trip count is a compile-time constant (a CPU probe with a fixed 256-iteration loop still allocated). Values from straight-line code go to the fixed-size tape that KernelAbstractions' extension allocates once per launch. The two changes remove each source of malloc:
- The gather's loop over the 3 × 3 stencil, and the flux kernel's loops over its blocks of atomic adds, became `ntuple`s.
- `Atomix.modify!` wraps the add in `GC.@preserve` of the array. On the device the array is a by-value struct, and the preserve spills it to a stack slot that Enzyme moves to the heap (`enzyme_fromstack`). The engine now calls `UnsafeAtomics.modify!` on `pointer(J, i)`. That is the same operation, with the same ordering and scope, without the preserve.

With both changes, the 18 kernels of a reverse step contain no device malloc (counted in their LLVM IR on the RTX 5090). On the plain run the same changes made Fast mode 4 to 10 % faster in Float32 on the RTX 5090. Reference mode and the duals were within 2 % (per-change benchmark, single runs, not committed).

## Not shown, and what blocks

- **CPU.** Enzyme cannot compile POCL's kernels (#509). The owner set the GPU first.
- **Reference mode.** The sort hangs in `copyto!` from one `CuArray` to another under Enzyme, and the tuple recursion of `permute_all!` crashes Enzyme at compile time. Neither is reduced yet; they need a GPU (#385). WP-077 (#527) replaces the sort and the transfer kernels with hand-written Enzyme rules, and with those Reference mode differentiates on CUDA (its own measurements, not repeated here).
- **Two species.** The loops over the species tuple fail as #384 says.
- **Determinism.** The adjoint of the gather is a scatter that Enzyme accumulates with atomics, so the reverse pass is not bitwise reproducible: two calls differed in every chain-rule run (`repeat_bitwise_equal = false`; #386).
- **Not tried.** The moving window, the antenna, the open boundaries, the PML, the filter, 3D, and checkpointing for long runs. The tape grows linearly with the steps.

## Where the data is

`wp-050-enzyme-spike/` holds the result and manifest of each run. The logs and the device-malloc probes stayed on the machine that ran them. The CPU runs of 2026-11-09 (side environment, KernelAbstractions 0.9) are in git history.

## Open issues

- #385: the Enzyme check in the test suite, which waits for Active kernel arguments in a KernelAbstractions release (JuliaGPU/KernelAbstractions.jl#979); Reference mode; the abort when the state is built inside the gradient.
- #384 (`upstream`): "Illegal address space propagation" on a loop over a homogeneous tuple of structs.
- #386: the reverse pass is not bitwise reproducible.
- #526 (`upstream`): Enzyme heap-allocates a copy of an isbits argument under `GC.@preserve` (reduced on the CPU), the mechanism behind the pointer atomics.
- #509: Enzyme on POCL.
- EnzymeAD/Enzyme.jl#2324: an illegal memory access in a GPU reverse pass at larger sizes, the symptom these runs had before the changes; its cause is not identified there.
