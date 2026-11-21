# WP-084: Enzyme's reverse mode through the 3D trapped-beam case

**Summary.**
- Enzyme's checkpointed reverse pass runs through the whole 3D trapped-beam case on CUDA. The case has the antenna, the moving window, PMLs on the open faces and on the window's back edge, field ionization and the compensated gather. The reverse sweep goes through Checkpointing.jl's `Revolve` with `checkpoint_storage` (WP-080), one step per Enzyme call, and needs three script-only patches: WP-078's two (#546; #551, #545) and a new one that unrolls the species loops of `current!`. Without the new patch Enzyme's type analysis of the 3D step fails (#384).
- At λ₀/16 (224 × 128 × 128 cells, 985 steps, 9.2M plasma and 2.0M born-electron slots, Float32), the reverse gradient of the four free parameters agrees with Dual{1} runs, one partial per parameter, to 1.3·10⁻⁵ (duration), 3.4·10⁻⁵ (waist), 2.1·10⁻⁶ (focus) and 2.5·10⁻⁵ (density), relative. On the warm-up box (112 × 64 × 64, 697 steps), five runs agree with Dual{4} to at most 2.7·10⁻⁶, and to 9.1·10⁻⁶ with one plasma particle per cell and a fifth of the born-electron capacity.
- The cost at λ₀/16 on the RTX 5090, with the 12 checkpoints in host memory: 778 s for the reverse pass, against 27.2 s for the plain run, 68 to 74 s for a Dual{1} run and 227 s for Dual{4}. So the gradient costs 28.6 plain runs, 11.4 Dual{1} runs and 3.4 Dual{4} runs. Its device memory peaks at 26.7 GiB, against 11.2 GiB for the plain run, 10.3 GiB for Dual{1} and 24.5 GiB for Dual{4}.
- Three full-size reverse passes did not fit in the card's 32 GB:
  - λ₀/16 with 6 checkpoints on the device;
  - λ₀/32 (448 × 128 × 128, 1706 steps) with host checkpoints, 3.4 GiB short;
  - λ₀/32 with one plasma particle per cell and a fifth of the born capacity, with host checkpoints, 210 MiB short.
  At λ₀/32 the plain run alone peaks at 23.5 GiB and Dual{1} at 21.5 GiB.
- Checkpointing.jl's `Revolve` keeps a `deepcopy` of the whole loop body as the reverse sweep's working copy (`checkpoint_alloc`). So the reverse pass holds three full states on the card: the state, its shadow and that copy. The evolved arrays that a checkpoint holds are only 0.60 GiB of the 4.66 GiB state at λ₀/16.
- CPU test T-1150 runs Revolve's whole schedule over a small version of the case. The forward sweep and a reverse sweep run with each adjoint replaced by a check, and every checkpoint, restore and u-turn matches the uninterrupted run bit for bit.

| | |
|---|---|
| Work package | WP-084 (`trace/wp/WP-084-reverse-mode-3d-trapped.md`); goals A0, A3, A4, A7, A8 |
| Commits | GPU runs: `6ce5cfd2f778ab68f0a39bd8851d5619e35580f7`, except `enzyme-trapped3d-grad-20261121T161539` at `09e2193c4e17375fbc2cd5214d69d163bf850f26`, which has neither the third patch nor the reverse pass on the start itself; `dirty = false` in every manifest. The branch's later commits change the scripts' comments, the runner's defaults and what the manifests record, not what the runs compute |
| Runs | the 14 `enzyme-trapped3d-*` directories in `wp-084-reverse-3d-trapped/`: each `grad` directory holds a `result.toml`, a `manifest.toml` and a `published.toml`, and each `gpu` directory holds the runner's `summary.txt` and a `published.toml`. The data, including the logs and the nvidia-smi samples, is on branch `runs/wp084-5090` at `1c16efd87de69c170910cc9c04e9deca6562f6e0`. The account's home directory is replaced by `~` in six logs |
| Input hashes | SHA-256 of `scripts/enzyme/{trapped3d,common,contract,active_reduce}.jl`, `scripts/enzyme/spike/unrolled_species.jl`, the 3D, 2D and born case files, the test environment's `Project.toml` and the root `Manifest.toml`, in each manifest. These manifests do not hash the third patch, `scripts/enzyme/spike/unrolled_current.jl` (named under `patches`; SHA-256 `713565820966050b46269dc96ec148d30d558e87fd7a7d3adc792b0ca2a99ca8` at 6ce5cfd). The driver now hashes every loaded patch |
| Environment | Julia 1.13.4; Enzyme 0.13.221, Checkpointing 0.13.0, KernelAbstractions 0.10.8, ForwardDiff 1.4.6; NVIDIA GeForce RTX 5090 (32 GB, driver 615.71.09, AMD Ryzen 9 9950X host); CUDA backend; `-t 2` |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## Setup

- **Case** (`scripts/optim/trapped3d_case.jl`): the 3D version of WP-065's trapped-beam case, ported from the grid study's 3D case.
  - Box and particles: 14·ppl cells along x and `rows` cells along y and z of π/4 each. Two plasma particles per cell (2 × 1 × 1) over a 40 c/ω₀ density ramp.
  - Ionization: a nitrogen dopant section with ADK ionization of the N⁶⁺ and N⁷⁺ levels, one atom per cell, in rows ±18·π/4 about the axis.
  - Laser: a Gaussian antenna pulse at a fixed pulse energy (the a₀ = 1.9 budget, `budget3`). The four free parameters are duration 6, waist 10, focus 80 and density 0.072; they enter through the antenna and the weights (A3).
  - Numerics: Yee solver at 0.95 of the 3D Courant limit. A moving window from t = 108. 12-cell PMLs on y and z and a back-edge PML. CompensatedGather(k = 1), Reference mode, Float32. The run ends at t = 300.
  - The objective J is the trapped charge of the first bucket, plasma and born electrons together, as in WP-065.
- **Reverse pass** (`scripts/enzyme/trapped3d.jl grad`):
  - The pass is `Enzyme.autodiff(set_runtime_activity(Reverse), ...)` over `@ad_checkpoint Revolve(c; storage = checkpoint_storage(; host)) for k in 1:n step!(...) end`, followed by the objective's per-particle summands. The state is `Duplicated`, and the parameters sit in a `Ref`, so they get their adjoint.
  - The gradient with respect to θ has two parts, joined by the chain rule. The start state's adjoint is contracted with the tangent of `initialize!` in each parameter, taken from a Dual{1} start. The parameters' adjoint is contracted with dp/dθ, taken from a Dual{4} problem.
  - The reverse pass runs on the start state itself, not a copy, to save one state of memory.
- **Checks.**
  - Each run compares against Dual{1} runs, one partial per parameter (`dual=`). On the warm-up box it also compares against a Dual{4} run.
  - Every run starts with the warm-up box (ppl = 8, rows = 64, the same t_end). It compiles all the kernels and the reverse pass, and checks the gradient against Dual{4}.
- **Memory.**
  - `peak` is the high-water mark of CUDA.jl's memory pool beyond the start, sampled every 2 ms from a second thread. It counts blocks that are freed but not yet reclaimed, so it is an upper bound.
  - It leaves out the device malloc heap and the context. nvidia-smi showed the card full (32187 MiB) during all four full runs, the one that fit too, because the pool keeps what it reserved.
  - The `state` that the logs print counts the floating-point arrays only, which is what the shadow mirrors. The driver now also prints all arrays.
- `scripts/enzyme/run_trapped3d.sh` ran the list below, each run in its own process.

## Results

### Gradient checks

| run | box | reverse against | relative differences (duration, waist, focus, density) |
|---|---|---|---|
| 165042 (device ckpt, c = 4) | warm-up | Dual{4} | 1.7·10⁻⁷, 2.5·10⁻⁶, 2.3·10⁻⁶, 2.7·10⁻⁶ |
| 171742 (host ckpt) | warm-up | Dual{4} | 6.0·10⁻⁷, 2.4·10⁻⁶, 2.2·10⁻⁶, 2.6·10⁻⁶ |
| 173007 (device ckpt) | warm-up | Dual{4} | 1.7·10⁻⁷, 2.5·10⁻⁶, 2.3·10⁻⁶, 2.5·10⁻⁶ |
| 174936 (host ckpt) | warm-up | Dual{4} | 2.5·10⁻⁷, 2.4·10⁻⁶, 2.3·10⁻⁶, 2.6·10⁻⁶ |
| 180824 (host ckpt, ppc 1 × 1 × 1, cap 0.2) | warm-up | Dual{4} | 5.0·10⁻⁶, 5.7·10⁻⁶, 4.4·10⁻⁶, 9.1·10⁻⁶ |
| 171742 (host ckpt, c = 12) | λ₀/16 | Dual{1} each | 1.29·10⁻⁵, 3.35·10⁻⁵, 2.09·10⁻⁶, 2.49·10⁻⁵ |

- The λ₀/16 gradient: dJ/dduration −4.2871·10⁻⁴, dJ/dwaist −9.2699·10⁻⁴, dJ/dfocus 8.5805·10⁻⁴, dJ/ddensity 0.63646.
  - The start state's part is nonzero only for the density, at 1.61·10⁻². The density enters the start through the weights. The other three parameters enter only through the antenna, so they are in `p`.
- The J of the reverse pass's own forward sweep equals the plain run's to Float32 rounding: 0.0036936969 against 0.0036936968.
- Two reverse passes of the same warm-up case differ by up to 5·10⁻⁷ relative. The Reference mode's bits do not carry over to the reverse pass, whose adjoint accumulation is atomic.
- The Dual runs' value of J differs from the plain run's by a few Float32 ulps (0.0036936989 against 0.0036936968 at λ₀/16). On the CPU the fields of a Dual{1} and a plain run part at the first step, at 10⁻⁴² while the antenna's emission is still subnormal (the physics breaker, below). The cause is not found; A4 does not ask Dual and Float32 runs for the same bits.
- The λ₀/16 differences against Dual{1} are about ten times those of the warm-up box against Dual{4}. That is consistent with Float32 accumulation over 1.4 times the steps and 16 times the particles, but it has not been shown.

### Cost

| box | plain | reverse | Dual{1} | Dual{4} | state (float arrays) | one checkpoint |
|---|---|---|---|---|---|---|
| warm-up, 112 × 64 × 64, 697 steps | 9.3 s, 1.54 GiB | 399 to 436 s, 10.9 to 12.2 GiB | | 34 s, 3.30 GiB | 0.63 GiB | 0.10 GiB |
| λ₀/16, 224 × 128 × 128, 985 steps | 27.2 s, 11.2 GiB | 778 s, 26.7 GiB (host ckpt, c = 12); out of memory with 6 device checkpoints | 68 to 74 s, 10.3 GiB | 227 s, 24.5 GiB (run 161539) | 4.66 GiB | 0.60 GiB |
| λ₀/32, 448 × 128 × 128, 1706 steps | 95.6 s, 23.5 GiB | out of memory, 3.4 GiB short (host ckpt, c = 12) | 250 s, 21.5 GiB | | 9.60 GiB | 1.34 GiB |
| λ₀/32, ppc 1 × 1 × 1, cap 0.2 | 82.2 s, 18.9 GiB | out of memory, 210 MiB short (host ckpt, c = 12) | 208 s, 18.0 GiB | | 8.44 GiB | 0.77 GiB |

- At λ₀/16 one reverse step costs 790 ms, against 27.6 ms for a plain step: 28.6 times. The grid study's periodic 3D plasma at the same size costs 10.6 plain steps per reverse step (runs/gridppc-v2-5090-enzyme3d). That case has no antenna, window, PMLs or ionization. Which part of the factor of 2.7 comes from those, and which from compilation, recomputation or host copies, is not measured.
- The memory at λ₀/16 is an estimate, not a measurement of each part. The state, its shadow and Revolve's working copy come to about 3 × 4.7 GiB. That leaves about 12 GiB for one step's tape and its temporaries. Host checkpoints save the 7.2 GiB that 12 device checkpoints would hold, which is what let the pass fit.
- A λ₀/32 reverse pass would need the card's memory plus about 3.4 GiB. Fewer particles alone got it to within 210 MiB.

### Failures on the way

- Without `spike/unrolled_current.jl`, the warm-up reverse pass stops with an `IllegalTypeAnalysisException` (a Union type) inside `current!` (run 161539, #384).
- With `strict_aliasing=0` instead, it stops with an `EnzymeNoTypeError` in `step!` (run 164915).
- The first λ₀/16 attempt (161539) ran out of memory with 12 device checkpoints and a copy of the start. The driver then dropped the copy and the runner moved to host checkpoints.
- Runs 154816, 162529 and 162539 ran out of memory at startup while other work held the card. They hold no result.

## Verification

- T-1150 (`lib/ParticleMeshElectromagnetic/test/checkpointing.jl`, CPU) runs Revolve with 3 checkpoints over 96 steps of a 72 × 16 × 16 version of the case in Float32. The version has the antenna, the window, 4-cell PMLs on y, z and the back edge, ionization and the compensated gather.
  - Checkpoints fall at steps 0, 61 and 88. Each one, restored into a state whose evolved arrays were overwritten, continues to the run's end bit for bit.
  - The reverse sweep runs as `rev_checkpoint_for` runs it, with each adjoint replaced by a step. Its 96 u-turns each see the uninterrupted run's state before that step, bit for bit, and so do its 95 restores. Those restores cover odd and even steps from 0 to 88, and its stores write slots a second time.
  - The window counters and the counter of the births without a slot are among the evolved arrays.
- Planted defects (the mutation breaker, below) that T-1150 now catches:
  - the open or back-edge PMLs or the atoms' populations left out of the checkpoint;
  - the counters left out;
  - a slot that is not written again on reuse;
  - a restore that skips the last array;
  - the compensated gather's copy refreshed only on odd steps.
- The physics breaker built the case next to the grid study's 3D case (`gridppc3d.jl`) on the same engine at ppl = 8, rows = 64. The problems, the 64 evolved arrays through step 395 and J agree bit for bit in Float32 at the base point and through step 93 in Float64. In Float32 the energy budget rounds differently in the two files, so a₀ differs by an ulp at 68 % of 2000 random points of the box and the Dual{4} problems' partials of a₀ differ in their last bits; the case file says so. `pulse_energy3` equals the antenna's `laser_energy` (relative 0 in Float64) and a numerical integral of the antenna's field over its plane to 2.1·10⁻⁶.
- T-1150's forward check also passes with 2, 5 and 9 checkpoints, over 192 steps, in Float64, and in Fast mode on one thread; on four threads Fast mode is not reproducible and the check says so.
- The reverse pass itself runs only on CUDA, so no CPU test differentiates the step: Enzyme fails on PoCL (#509). The GPU evidence is the runs above.

## Open issues

- λ₀/32 does not fit on a 32 GB card. Two things could make it fit, neither tried: Revolve's working copy holding only the evolved arrays instead of a `deepcopy` of the body, and a smaller one-step tape. Another option is a larger card.
- The reverse step costs 28.6 plain steps here, against 10.6 in the periodic case. The breakdown is not measured.
- In 3D, a₀ ∝ 1/(w√τ) at fixed energy, so the optimizer box of the 2D case spans a₀ from 1.1 to 3.9 here (1.3 to 3.0 in 2D). The box and the born electrons' capacity were sized for 2D and are not yet checked for 3D.
- The `current!` patch is script-only. The engine's `current!` keeps its `for` loops, and the same unrolling would close #384 for the 3D step.
