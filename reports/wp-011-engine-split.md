# WP-011: the engine/module split, before and after

| | |
|---|---|
| Work package | WP-011 (goals A0, A2, A3, A4, A7, A8, A9) |
| Commits | before: `4ed3667e61ffd4a2843254e6e4b3dea43728054b` (the spike's code); after: `d3782d7aafa22f2bcdc0e2f25e4f2972b6151711`; both clean trees |
| Runs | before: `landau-20261031T205207`, `two_stream-20261031T205228`, `timing-t1-20261031T205239`, `timing-t16-20261031T205255`; after: `landau-20261031T211537`, `two_stream-20261031T211558`, `timing-t1-20261031T211610`, `timing-t16-20261031T211626` |
| Input hashes | SHA-256 of every input file in each run's manifest: `wp-011-engine-split/<run>.manifest.json` |
| Environment | Julia 1.13.3, KernelAbstractions 0.10.4 CPU backend, AcceleratedKernels 0.5.11; Intel Core i9-14900K, 1 and 16 threads |
| Figures | `wp-011-engine-split/timing_t{1,16}_{before,after}.png`, each with its JSON sidecar |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## 1. Setup

WP-011 moves the spike's electrostatic step out of `lib/ParticleMeshCore/src` into test support (`lib/ParticleMeshCore/test/es_reference.jl`), routes its Gauss solve through the new engine operator `cumsum_periodic!`, and adds species extras, a deposited quantity and a launch keyword to the engine (`docs/plan/architecture-draft.md` §0.7). It is a refactor: the claim is that no bit changed. The scripts in `scripts/es1d/` ran once on each commit; the timing script is the spike's, with one more column (scratch with a two-array extras tuple).

## 2. Results

### Bits

- `landau.jl` and `two_stream.jl` at 16 threads give γ = −0.1492419736219506 and 0.3545690079405174 on both commits, the spike's recorded values. Every plot sidecar of the two runs (`*_mode_amplitude`, `*_growth`, `*_conservation`, `*_field`, `*_phase_space`) is byte-identical between the commits; only the wall time in `summary.json` differs.
- T-170 checks the same in the test suite: the two rates, FNV digests of the full T-141 and T-142 states, T-136's digest, and the state digests of twelve small cases (Float32, Float64, `Dual{Float32,3}`, orders 1 and 2, Landau and two-stream, 40 steps), all recorded on the before commit.
- Outside the suite, the same small cases in `Fast()` mode (atomics) were also bitwise equal between the commits on one thread. With four threads, Fast-mode results differ from run to run on either commit, as atomics do; Reference mode is bitwise at 1 and 4 threads on both.

### Time per particle-step (no thresholds)

ns per particle-step of the whole electrostatic step, 64 cells, 50 steps after warm-up, one run each (sidecars `timing_t{1,16}_{before,after}.json`). The spike's two particle counts:

| particles | threads | F32 Fast | F32 Ref | F64 Fast | F64 Ref |
|---|---|---|---|---|---|
| 16 384 | 1 | 18.7 → 22.5 | 27.0 → 28.3 | 19.4 → 23.5 | 30.2 → 28.9 |
| 262 144 | 1 | 17.6 → 17.9 | 27.4 → 27.0 | 17.9 → 18.5 | 29.4 → 32.2 |
| 16 384 | 16 | 40.5 → 35.0 | 32.2 → 26.0 | 29.4 → 29.7 | 23.8 → 39.9 |
| 262 144 | 16 | 31.9 → 31.9 | 7.4 → 7.6 | 26.0 → 26.0 | 13.4 → 8.7 |

The differences go both ways and are as large between two runs of one commit: an intermediate run of the after code on an uncommitted tree gave 30.4 (1 thread) and 10.2 (16 threads) for the Float64 Reference entries at 262 144 particles. With 16 threads the timings move by up to a factor of two from run to run. Nothing here shows a change caused by the refactor, and one run per commit cannot show a change below that noise.

### Scratch per particle

`Workspace` scratch: 33 B per particle for Float32 and 37 B for Float64 on both commits, and the same with the extras `(old_cell = Int32, old_offset = T)`, which reuse the core permutation buffers. An extra of another element type adds one buffer per element type (T-173).

### Allocation per step

With one thread, allocation per step does not grow with the particle count (T-144: 1856 B in Reference mode, 0 B in Fast mode, at every size, with or without extras). With more than one thread the Reference step allocates about 8 B per particle on both commits: 17 712 B at 512 particles, 79 840 B at 8192 and 4.2 MB at 524 288 with 4 threads. The cause is upstream: AcceleratedKernels' threaded CPU sample sort sorts each bucket without the workspace's scratch (#82). T-144 therefore fails at 4 threads; the CPU CI runs one thread.

## 3. Open

- T-144 at more than one thread waits on #82 or on a workaround in the engine (`questions/wp-011-t144-threads.md`).
- The 16-thread timings are too noisy for a comparison below a factor of about two. Repeated runs with a spread are a benchmark task, not this package's.
