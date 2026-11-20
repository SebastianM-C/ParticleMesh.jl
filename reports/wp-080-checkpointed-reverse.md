# WP-080: checkpointed reverse mode through the 2D and 3D step

**Summary.**
- Enzyme's reverse pass runs over full-length 2D and 3D runs on CUDA through Checkpointing.jl's `Revolve`, with `checkpoint_storage()` holding only the evolved arrays: 168 MiB per checkpoint at 1024² (the state is 633 MiB), 313 MiB at 96³ (963 MiB).
- The checkpointed gradient equals the plain reverse pass's, per array, to the spread between two plain passes: at most 1.4·10⁻¹⁴ in Float64 and 6.6·10⁻⁶ in Float32, up to 400 steps in 2D and 200 in 3D. It agrees with ForwardDiff by the chain rule, in the α direction and a random one, in all six checks.
- 2D, 1024², 4.2M particles, 128 steps (RTX 4080 SUPER): one whole gradient costs 4.27 s (Fast) and 4.47 s (Reference). The primal takes 0.49 s and 0.64 s, Dual{1} 0.97 s and 1.03 s, and Dual{4} 9.14 s and 9.74 s. So the gradient costs 4.4 Dual{1} runs and 0.46 to 0.47 of a Dual{4} run.
- 3D, 96³, 7.1M particles, 32 steps (RTX 5090): the gradient costs 2.41 s (Fast) and 1.63 s (Reference), Dual{1} 0.70 s and Dual{4} 3.39 to 3.41 s: 3.5 and 2.3 Dual{1} runs, 0.71 and 0.48 of a Dual{4} run. At 128 steps it is 5.2 (Fast) and 9.7 (Reference) times the primal.
- The checkpointing itself costs 1.26 to 1.35 times the plain reverse pass at 2 steps, the longest plain reverse pass measured (a few more steps of tape would not fit the card). Checkpoints in host memory are 1.4 to 3.0 times slower than on the device.
- Dual{4} costs 9.5 times Dual{1} in 2D and 4.9 times in 3D, against an expected 4 or less (#549). Counted in Dual{1} runs, one per parameter, reverse mode breaks even at 4 to 5 parameters in 2D, and wins from 4 (Fast) and 3 (Reference) parameters in 3D.
- The device memory does not grow with the step count: 10.4 to 10.9 GiB at 512 steps in 2D against 9.9 to 11.0 GiB at 32. The plain reverse pass of one step alone peaks at 2.7 GiB (2D) and 7.4 GiB (3D).

| | |
|---|---|
| Work package | WP-080 (`trace/wp/WP-080-checkpointed-reverse.md`); goals A0, A4, A7, A8 |
| Commits | checks and 3D cost runs: `231f9f9551a86d34bbab740261ffdb64b9ec05b6`; 2D cost runs: `e79544c2f0d4947c1ee5e279c990d8dc7862a209`, whose engine differs from 231f9f9 only in how `checkpoint_storage` finds the states of the loop body (the same arrays for this body); `dirty = false` in every manifest |
| Runs | the ten `enzyme-checkpointed-*` directories in `wp-080-checkpointed-reverse/`, each a `result.toml`, a `manifest.toml` and a `published.toml`; data on branch `runs/wp080-gpu-20261120T2239` |
| Input hashes | SHA-256 of `scripts/enzyme/{checkpointed,common}.jl`, the 2D and 3D case files, the test environment's `Project.toml` and the root `Manifest.toml`, in each manifest |
| Environment | Julia 1.13.4; Enzyme 0.13.221, Checkpointing 0.13.0, KernelAbstractions 0.10.8, ForwardDiff 1.4.6; NVIDIA GeForce RTX 4080 SUPER (16 GB, driver 620.70.06, Intel Core i9-14900K host) and NVIDIA GeForce RTX 5090 (32 GB, driver 615.71.09, AMD Ryzen 9 9950X host); CUDA backend; the spike patch `scripts/enzyme/spike/active_args.jl` as in WP-050 |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## Setup

- **Case.** WP-023's `plasma_step_2d` and WP-042's `plasma_step_3d`, electrons only, Esirkepov current, quadratic shape, Boris pusher, Yee solver, `observer = ()`. The parameter α is the density perturbation amplitude, which enters through the weights (A3). The functional is the field energy after n steps.
- **Checkpointing.** `@ad_checkpoint Revolve(c; storage = checkpoint_storage()) for k in 1:n step!(st, p, k) end` under `Enzyme.autodiff(set_runtime_activity(Reverse), ...)`. Revolve reverses one step per Enzyme call, so the tape spans one step. `checkpoint_storage(host = true)` keeps the checkpoints in host memory instead.
- **Check** (`scripts/enzyme/checkpointed.jl check`). It computes the checkpointed gradient with respect to every floating-point array of the initialized state, then the plain reverse pass of all n steps, a second plain pass and a pass with host checkpoints. The checkpointed gradient must match the plain one, per array, within 8 times the larger of the two spreads (plain against plain, host against device). The chain rule compares ∇f·t against ForwardDiff's derivative along t, for t = dx/dα and a random t on every array. Its tolerance, 64·eps(T)·n·c, uses the direction's own condition number c = Σ|∇fᵢtᵢ|/|∇f·t|. That bound is loose (up to 1.9·10⁻² in Float32), so the measured differences are the evidence, not the bound.
- **Cost** (`scripts/enzyme/checkpointed.jl cost`). For each step count it takes the median of three timings, each on a fresh input made outside the timing. It times the primal, Dual{1}, Dual{4}, the plain reverse pass (up to `plain` steps, where it fits), and the checkpointed pass with device and with host checkpoints. The peak is the device pool's high-water mark beyond the input, sampled during the call. It includes blocks that have been freed but not yet collected, so it is an upper bound. That is why it does not rise steadily with the steps. Dual states take the sorted Reference deposition in both modes (`effective_mode`, the engine has no atomics for Duals), so their two modes time the same.
- `scripts/enzyme/run_checkpointed.sh` runs the list, and `scripts/enzyme/table_checkpointed.jl` prints the tables below from the run directories.

## Results

The output of `julia +1.13 --project=scripts scripts/enzyme/table_checkpointed.jl reports/wp-080-checkpointed-reverse/enzyme-*`, unchanged:

| GPU | Commit | Dim | T | Mode | Particles | Steps | c | Checkpointed vs plain reverse (tolerance) | Host vs device slots | Plain rerun | Chain rule, α (tolerance) | Chain rule, random (tolerance) | Verdict | First call, checkpointed (s) | First call, plain (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 4080 SUPER | `231f9f9` | 2 | Float64 | Reference | 720 | 32 | 4 | 6.8e-15 (5.5e-14) | 6.8e-15 | 6.2e-15 | 1.8e-15 (3.3e-11) | 5.6e-16 (1.0e-11) | pass | 262 | 107 |
| RTX 4080 SUPER | `231f9f9` | 2 | Float32 | Fast | 720 | 32 | 4 | 5.2e-06 (4.9e-05) | 5.2e-06 | 6.1e-06 | 1.6e-06 (1.8e-02) | 7.7e-07 (1.9e-02) | pass | 244 | 83 |
| RTX 4080 SUPER | `231f9f9` | 2 | Float64 | Reference | 720 | 400 | 10 | 1.4e-14 (8.9e-14) | 7.5e-15 | 1.1e-14 | 4.5e-15 (7.6e-11) | 5.4e-14 (4.0e-10) | pass | 298 | 118 |
| RTX 4080 SUPER | `231f9f9` | 3 | Float64 | Reference | 2688 | 16 | 3 | 5.8e-15 (5.6e-14) | 6.2e-15 | 7.0e-15 | 8.9e-16 (3.1e-11) | 6.3e-16 (1.5e-11) | pass | 285 | 126 |
| RTX 4080 SUPER | `231f9f9` | 3 | Float32 | Fast | 2688 | 16 | 3 | 6.6e-06 (4.2e-05) | 5.2e-06 | 4.9e-06 | 5.4e-07 (1.7e-02) | 6.9e-07 (1.9e-02) | pass | 297 | 97 |
| RTX 4080 SUPER | `231f9f9` | 3 | Float64 | Reference | 2688 | 200 | 8 | 7.2e-15 (5.5e-14) | 6.9e-15 | 6.1e-15 | 8.5e-13 (2.0e-10) | 1.1e-11 (1.7e-09) | pass | 323 | 125 |

| GPU | Commit | Dim | T | Mode | Cells | Particles | c | Steps | Primal (ms) | Dual{1} (ms) | Dual{4} (ms) | Plain reverse (ms) | Checkpointed (ms) | Host slots (ms) | Checkpointed / Dual{4} | Checkpointed / plain reverse | Checkpointed per step (ms) | Peak: plain reverse, checkpointed (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Fast | 1024×1024 | 4194304 | 16 | 2 | 4.95 | 15.21 | 119.20 | 38.93 | 49.13 | 73.23 | 0.41 | 1.26 | 24.56 | 3041, 4090 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Fast | 1024×1024 | 4194304 | 16 | 8 | 19.94 | 59.65 | 510.35 | – | 196.76 | 592.01 | 0.39 | – | 24.59 | –, 5077 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Fast | 1024×1024 | 4194304 | 16 | 32 | 96.45 | 238.89 | 2181.73 | – | 899.58 | 1954.87 | 0.41 | – | 28.11 | –, 10156 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Fast | 1024×1024 | 4194304 | 16 | 128 | 485.60 | 965.42 | 9136.58 | – | 4265.15 | 6944.36 | 0.47 | – | 33.32 | –, 7780 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Fast | 1024×1024 | 4194304 | 16 | 512 | 2217.32 | – | – | – | 20059.34 | 28024.33 | – | – | 39.18 | –, 10672 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Reference | 1024×1024 | 4194304 | 16 | 2 | 9.78 | 15.49 | 121.10 | 46.71 | 62.90 | 95.00 | 0.52 | 1.35 | 31.45 | 3105, 4154 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Reference | 1024×1024 | 4194304 | 16 | 8 | 38.39 | 64.07 | 578.78 | – | 246.33 | 599.49 | 0.43 | – | 30.79 | –, 5125 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Reference | 1024×1024 | 4194304 | 16 | 32 | 160.65 | 254.60 | 2381.47 | – | 1055.63 | 1988.47 | 0.44 | – | 32.99 | –, 11308 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Reference | 1024×1024 | 4194304 | 16 | 128 | 638.02 | 1027.91 | 9738.30 | – | 4471.64 | 7717.90 | 0.46 | – | 34.93 | –, 11308 |
| RTX 4080 SUPER | `e79544c` | 2 | Float32 | Reference | 1024×1024 | 4194304 | 16 | 512 | 2690.39 | – | – | – | 19987.77 | 34504.32 | – | – | 39.04 | –, 11192 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Fast | 96×96×96 | 7077888 | 8 | 2 | 15.56 | 42.76 | 208.03 | 79.04 | 103.13 | 173.92 | 0.50 | 1.30 | 51.56 | 11161, 7875 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Fast | 96×96×96 | 7077888 | 8 | 8 | 70.59 | 171.64 | 839.75 | – | 417.54 | 1126.17 | 0.50 | – | 52.19 | –, 9754 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Fast | 96×96×96 | 7077888 | 8 | 32 | 446.26 | 695.98 | 3412.69 | – | 2409.21 | 6016.68 | 0.71 | – | 75.29 | –, 20208 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Fast | 96×96×96 | 7077888 | 8 | 128 | 2767.62 | – | – | – | 14412.49 | 22120.25 | – | – | 112.60 | –, 10067 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Reference | 96×96×96 | 7077888 | 8 | 2 | 11.30 | 42.75 | 208.11 | 75.88 | 95.67 | 154.72 | 0.46 | 1.26 | 47.83 | 11269, 12573 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Reference | 96×96×96 | 7077888 | 8 | 8 | 44.92 | 171.94 | 839.96 | – | 372.51 | 1083.91 | 0.44 | – | 46.56 | –, 14452 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Reference | 96×96×96 | 7077888 | 8 | 32 | 180.64 | 695.74 | 3393.05 | – | 1629.39 | 2931.13 | 0.48 | – | 50.92 | –, 25014 |
| RTX 5090 | `231f9f9` | 3 | Float32 | Reference | 96×96×96 | 7077888 | 8 | 128 | 728.36 | – | – | – | 7091.94 | 11016.64 | – | – | 55.41 | –, 14765 |

| Dim | T | Mode | Cells | Particles | State (MiB) | Dual{1} state (MiB) | Dual{4} state (MiB) | One checkpoint (MiB) | Plain reverse of one step, peak (MiB) | First call, checkpointed (s) | First call, plain (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Float32 | Fast | 1024×1024 | 4194304 | 633 | – | – | 168 | 2734 | 14 | 0.0 |
| 2 | Float32 | Reference | 1024×1024 | 4194304 | 633 | – | – | 168 | 2766 | 14 | 0.0 |
| 3 | Float32 | Fast | 96×96×96 | 7077888 | 963 | 1927 | 4816 | 313 | 7561 | 18 | 0.1 |
| 3 | Float32 | Reference | 96×96×96 | 7077888 | 963 | 1927 | 4816 | 313 | 7615 | 17 | 0.1 |

The first calls compile. In the checks, the checkpointed pass takes 244 to 323 s and the plain pass after it 83 to 126 s. In the cost runs, the first checkpointed call took 14 to 18 s. Why it took so much less there is not established.

## Not shown

- The 3D cost at 64³ on the RTX 4080 SUPER did not run: the card was in use by other software at the time. 96³ with 2×2×2 per cell needs 7.4 GiB of tape for one step, so on a 16 GB card it does not fit with checkpoints; it ran on the RTX 5090.
- Fast mode is 1.4 to 3.8 times slower than Reference in the 3D primal at 96³, and the gap grows with the steps (#550; cause not checked).
- `Periodic` and Checkpointing.jl's other schemes: `Periodic` keeps a period's inner steps as whole-body copies in its own `ArrayStorage` (`checkpoint_storage`'s docstring), so only `Revolve` is measured.
- Two species, ionization, the moving window and the PML in a reverse pass on the GPU. T-1112 runs Revolve's forward schedule on the 2D window with a back-edge PML and on the 3D window, and a breaker replayed the reverse schedule with ionization and two species on the CPU. WP-078 differentiates the trapped-beam case with the window.
- The CPU: Enzyme cannot differentiate the step's kernels on KernelAbstractions' CPU backend (#509). The CPU tests T-1110 to T-1114 cover the rule and the storage there.

## Data

`scripts/rundata/run-data.sh fetch reports/wp-080-checkpointed-reverse/<run>/published.toml` brings back each run's files from `runs/wp080-gpu-20261120T2239` and checks their hashes. The runs' logs stay off the branch, because they hold local paths.
