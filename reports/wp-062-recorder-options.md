# WP-062: what the recorder's options save

**Summary.**
- The history recorder takes `every` (record every n-th step) and `record` (the groups `:field_energy`, `:kinetic`, `:modes`), on `HistoryRecorder` and, for a state's own recorder, on `EMState`. A step computes the kinetic-energy cache once, after the window's shift, and only where the next step's observer reads it.
- What a recorder with options records has the bits of the same entries of the recorder that records every step and every group, and a run's evolved arrays are the same under every recorder, on the CPU (Float32, Float64, `Dual{Float32,3}`) and on CUDA (T-810, T-811, T-814; T-813 on CUDA).
- On an RTX 4080 SUPER the WP-045 case (224 × 64 cells) takes 0.29 ms per step with `record = ()` against 0.56 on `main` in Float32, and 2.29 against 4.75 in `Dual{Float64,2}`: 1.9 to 2.1 times fewer milliseconds. At 448 × 128 cells with 16 particles per cell the factor is 1.5.
- With the default recorder the step is 8 to 11 % faster than on `main`, from the one kinetic-energy launch per step where a moving window made two.
- Recording every 16th step costs within 6 % of recording nothing, except one case (M, `Dual{Float64,2}`: 27.7 against 23.1 ms, one run).

| | |
|---|---|
| Work package | WP-062 (`trace/wp/WP-062-recorder-options.md`); goals A4, A7, A8 |
| Commits | `main` at `fa4953f`; the branch at `ec76c38` |
| Runs | `optim-recorder-timing-cuda-20261116T175209` (`main`, the script copied into its tree as an untracked file) and `optim-recorder-timing-cuda-20261116T175518` (branch, clean), one after the other on one machine |
| Input hashes | SHA-256 of every input in each run's manifest: `wp-062-recorder-options/gpu-main/manifest.toml` and `gpu-branch/manifest.toml` |
| Environment | Julia 1.13.3, CUDACore 6.5.1, KernelAbstractions CUDA backend, NVIDIA GeForce RTX 4080 SUPER, Intel Core i9-14900K, 4 threads; a desktop session was on the same card |
| Figures | per run `recorder_timing.png` with its sidecar `recorder_timing.json`, and `tables.md` |
| Status | measured once per tree on one machine, not independently reviewed; no number here is to leave the repository before that review |

## 1. Setup

`scripts/optim/recorder_timing.jl cuda`: `wake_problem(7, 100)` in Reference mode at its own size (S: 224 × 64 cells, 4 particles per cell, 512 steps) and at M (448 × 128 cells, 16 particles per cell, 1024 steps), a state built with each recorder option, `run!` over the problem's steps with the device synchronized at both ends; three runs per case in one process, the least of the last two.

## 2. Results

Milliseconds per step, Reference mode (NVIDIA GeForce RTX 4080 SUPER). `main`:

| size | element type | `default` |
|---|---|---|
| S | `Float32` | 0.561 |
| S | `Float64` | 1.5 |
| S | `Dual{Float64,2}` | 4.75 |
| M | `Float32` | 2.39 |
| M | `Float64` | 8.22 |
| M | `Dual{Float64,2}` | 35.8 |

The branch:

| size | element type | `default` | `every = 16` | `field energy, every = 16` | `record = ()` |
|---|---|---|---|---|---|
| S | `Float32` | 0.51 | 0.304 | 0.293 | 0.29 |
| S | `Float64` | 1.33 | 0.753 | 0.72 | 0.758 |
| S | `Dual{Float64,2}` | 4.35 | 2.42 | 2.32 | 2.29 |
| M | `Float32` | 2.18 | 1.66 | 1.63 | 1.62 |
| M | `Float64` | 7.46 | 5.56 | 5.44 | 5.44 |
| M | `Dual{Float64,2}` | 32.4 | 27.7 | 24.1 | 23.1 |

## 3. Verification

The electromagnetic package's suite with `PM_TEST_GPU=1` on the same machine: 122 605 tests pass, the recorder's among them (T-810 to T-814). The bits of a default run are unchanged: the suite's recorded digests and determinism tests hold with the kinetic energy computed after the window's shift instead of before the mesh solve and again in the shift.

## 4. Not shown, and open

- No CPU timing: on the CPU the diagnostics were not the cost (WP-061's report).
- `observer = nothing` keeps the kinetic-energy cache current after every step, as before, so it does not get the saving; a state built with `record = ()` does. `scripts/optim/wake2d_case.jl`'s `wake_state` now builds its state that way.
- With a recorder that does not read the cache at the next step, `state.K` is stale after a step and marked so (`state.fresh`). An observer that reads it then (another recorder, a function, the step index started again) gets it recomputed first (T-814); a reader of `state.K` itself calls `refresh_caches!`, and `scripts/digest/digest.jl`, which hashes `state.K`, does not.
- `kinetic_energy_history` and `total_energy_history` average consecutive entries, so they are for histories of every step; given a recorder or a state of another cadence they refuse, given its bare histories they cannot tell.
- The sort and the flux patch kernel, the larger costs at scale, are not touched here.
