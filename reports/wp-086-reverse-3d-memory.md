# WP-086: the 3D reverse pass without script patches, and its memory

**Summary.**
- WP-084's two script-only patches for the 3D step now live in the engine:
  - the species loops of `current!` are `map`s (#384);
  - the density ramp goes through `sin` (#545), and the flag counts go through `add_count!`, which Enzyme takes as inactive (#551).
  The 3D trapped-beam case runs Enzyme's checkpointed reverse pass with WP-078's Active-argument patch alone (#546).
- Revolve's working copy of the loop body is now the engine's `StepBody` with `working_state`, which holds only the evolved arrays, the kinetic-energy cache and a recorder of its own. It used to be a `deepcopy` of the whole state. On the small 3D trapped case it holds 4.05 MB against 31.1 MB for the deepcopy (T-1170).
- The driver's shadow of the start shares the species' workspaces with the state, since no rule reads their adjoint. On the warm-up box its new arrays are 0.13 GiB, against 0.77 GiB for a full shadow.
- At λ₀/16 on the RTX 5090:
  - the reverse pass agrees with Dual{1} to 3.7·10⁻⁶ to 3.5·10⁻⁵ relative;
  - it takes 628 s with 12 host checkpoints, against 778 s in WP-084, and peaks at 25.5 GiB, against 26.7 GiB;
  - with 6 checkpoints in device memory, which ran out of memory in WP-084, it fits: 737 s and 27.5 GiB.
- λ₀/32 still does not fit in 32 GB. The four Dual{1} runs complete, but the reverse pass runs out with the pool at 27.4 GiB, during one reverse step. What remains is Enzyme's tape for the push kernel: 688 bytes per particle slot, 162 of its Float32 being the gathered field values, so 11.8 GiB for the 18.35M plasma slots.

| | |
|---|---|
| Work package | WP-086 (`trace/wp/WP-086-reverse-3d-memory.md`); goals A0, A4, A7, A8, A9 |
| Commits | the runs' commits are in each `manifest.toml`: `4c0e2b19` (run 153821), `9e0ec7ae` (173415, 180317), `5b80e862` (182615, 185444) |
| Runs | the 9 `enzyme-trapped3d-*` directories in `wp-086-reverse-3d-memory/`. Each `grad` directory holds `result.toml`, `manifest.toml` and `published.toml`; each `gpu` directory holds the runner's `summary.txt` and `published.toml`. The data, with the logs, is on branch `runs/wp086-5090` (commit `ed3a86bd`) |
| Input hashes | SHA-256 of the driver, `common.jl`, `contract.jl`, `active_reduce.jl`, the case files, the test environment's `Project.toml` and the root `Manifest.toml`, in each manifest |
| Environment | Julia 1.13.4; Enzyme 0.13.221, Checkpointing 0.13.0, KernelAbstractions 0.10.8, ForwardDiff 1.4.6; NVIDIA GeForce RTX 5090 (32 GB); CUDA backend; `-t 2`. The card held nothing else during the runs except the compositor (64 MiB) |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## Setup

- The case, the objective, the gradient's two parts and the checks are WP-084's (`reports/wp-084-reverse-3d-trapped.md`, Setup).
- The loop body is `checkpoint_for(StepBody(st, pr), Revolve(c; storage = checkpoint_storage(; host)), 1:n)` (`body=step`).
- `scripts/enzyme/trapped3d.jl` builds each state in a function of its own. The tuple that `start3` returns stays rooted in the frame that destructures it, and at λ₀/32 that kept the start state (11.7 GiB) on the card through the Dual{1} runs and the reverse pass.
- The Dual{1} runs now run before the reverse pass. The start's contraction with the tangent uses fused reductions and leaves out the shared workspaces.
- `peak` is as in WP-084, the high-water mark of CUDA.jl's pool beyond the start, an upper bound. The plain run's and the Dual runs' peaks no longer include a second copy of the start, which WP-084's did.

## Results

| run | commit | box | checkpoints | reverse | peak | against Dual{1} (duration, waist, focus, density) |
|---|---|---|---|---|---|---|
| 180317 / 182612 | 9e0ec7ae | λ₀/16, 224 × 128 × 128, 985 steps | 12, host | 627.6 s (637 ms per step) | 25.47 GiB | 2.21·10⁻⁵, 3.45·10⁻⁵, 3.65·10⁻⁶, 1.29·10⁻⁵ |
| 185444 / 191436 | 5b80e862 | λ₀/16 | 6, device | 737.0 s (748 ms per step) | 27.46 GiB | 1.79·10⁻⁵, 3.55·10⁻⁵, 3.58·10⁻⁶, 1.28·10⁻⁵ (run 182612's Dual{1}) |
| 153821 | 4c0e2b19 | λ₀/32, 448 × 128 × 128, 1706 steps | 12, host | out of memory allocating 11.76 GiB, the plasma push's tape | | |
| 173415 / 180314 | 9e0ec7ae | λ₀/32 | 12, host | out of memory allocating 4.84 GiB, the born push's tape (pool 27.2 GiB) | | |
| 182615 / 185442 | 5b80e862 | λ₀/32, a fifth of the born capacity | 12, host | out of memory allocating 70 MiB, a species snapshot of `deposit_flux!`'s rule (pool 27.4 GiB) | | |

- The λ₀/16 gradient (run 182612): dJ/dduration −4.2870·10⁻⁴, dJ/dwaist −9.2700·10⁻⁴, dJ/dfocus 8.5805·10⁻⁴, dJ/ddensity 0.63646. The two λ₀/16 reverse passes agree with each other to 4.2·10⁻⁶ (duration) and to 10⁻⁶ or less in the other three.
- Every warm-up gradient agrees with Dual{4} to at most 4.0·10⁻⁶ relative, in all five runs.
- At λ₀/32 the Dual{1} runs take 218 to 228 s and peak at 21.5 GiB (20.2 GiB with the smaller born capacity). The derivatives are dJ/dduration 3.9177·10⁻⁴, dJ/dwaist −1.2744·10⁻³, dJ/dfocus 6.1183·10⁻⁴ and dJ/ddensity 0.47241.
- A fifth of the born capacity (1,974,067 slots against 7,547,904) leaves the λ₀/32 case as it is. The plain run's J and every Dual{1} value and derivative are bitwise those of the full capacity, and the plain run checks that no birth lacked a slot.
- Runs 173415 and 180317 ran with a registry in `active_reduce.jl` that marked the sharpened fields as unmodified between the push kernel's forward and reverse. The λ₀/32 backtrace shows it reached the launch, but the tape's element type stayed `Tuple{Bool, UInt32 × 3, Float32 × 168}`, so it was withdrawn (5b80e862). Within the kernel the stores into the species' arrays may alias the fields, so Enzyme caches the gathered values either way.
- The reverse pass costs 25.6 plain runs at λ₀/16 (24.5 s plain), against 28.6 in WP-084. Which change made it faster is not measured.

## Verification

- T-1170 (CPU) covers the working copy on three cases (2D window with back PML, 3D window, 3D trapped). It checks the memory it adds, that its species do not own their arrays, that its cache is computed and marked fresh, and that its recorder is its own. It continues the run bit for bit, the stepped state's evolved arrays stay as they were, `checkpoint_copy!` restores them, and stepping the body leaves the recorder as it is. It also runs Revolve's whole schedule through Checkpointing.jl's `ArrayStorage`.
- T-1171 (CPU) runs Enzyme through `checkpoint_for` with a `StepBody` against the plain reverse pass.
- T-1172 (CPU) covers `add_count!`. T-1173 (CPU) covers the ramp's bits, value and derivative. T-1174 (CUDA) runs Enzyme on the device through the ramped profile and through `add_count!`.
- On the RTX 4080 SUPER, at 4c0e2b19, the module's whole suite with the GPU tests passed 151,135 of 151,138. The three failures were these:
  - T-610's pinned values moved by 1.1·10⁻⁵ through the ramp's rounding, and were re-pinned.
  - T-1174's positions measured 930 (Float32) and 21,800 (Float64) against a scale that counts a nearly vanishing slope's argument rounding as error. The CPU's own Float32 against Float64 gives 1,080 in that scale, and 0.37 in the scale with the Hessian's terms that T-1174 now takes. T-1174 has not yet run on a GPU in the new scale.
- Breakers:
  - The physics breaker found no bitwise discrepancy. It continued the working copy at many split points over 2D, 3D, 1D, Float32, Float64, Dual, Fast, resort, witness and ionization cases, and ran whole schedules through `rev_checkpoint_for`. It measured the ramp at 2.7 eps for `sin` against 1.5 for `sinpi`, and showed observables changed only at roundoff. Its doc findings and the one-element counter are fixed.
  - The mutation breaker planted 23 defects. 18 were caught and 1 was equivalent. The 4 survivors (the cache refresh, a shared recorder, a dropped observer) now fail T-1170.

## Open issues

- λ₀/32 on 32 GB needs a smaller push tape. Gathering E and B through the engine's `gather!` rule would tape a copy of the fields once per step, about 0.2 GiB at λ₀/32, instead of 162 values per particle. The push would then read six per-particle values.
- T-1174 in its new scale is still to be measured on a GPU.
- The memory per reverse step at λ₀/16 is not broken down: 25.5 GiB against a 5.6 GiB state, of which the push tapes are about 7.2 GiB.
