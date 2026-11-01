# WP-021: the periodic spectral inverse of the discrete Laplacian at N = 1 and N = 2

| | |
|---|---|
| Work package | WP-021 (goals A4, A7, A8, A9, A10) |
| Commits | base `2948338`; the engine code, the tests, the timing script and this report in WP-021's pull request (rebased onto `08ed35e`, which changes, among the hashed inputs, only a comment in `lib/ParticleMeshElectromagnetic/src/step.jl`, a file the script does not load) |
| Runs | `engine2d-periodic-solve-t1-20261101T161519` (1 thread), `engine2d-periodic-solve-t4-20261101T161703` (4 threads) |
| Input hashes | SHA-256 of every input file in each run's manifest: `wp-021-periodic-solve-2d/<run>.manifest.toml`. Both say `dirty = true`. The 1-thread run names the base commit, because it started before the engine code was committed; the 4-thread run names `eac0306`, the branch's engine commit before the rebase. Both ran the same `lib/ParticleMeshCore/src/spectral.jl` and the same script (the same SHA-256 in both manifests and in the files as committed), with the uncommitted plan text and work package in the tree, which are not inputs of the runs. |
| Environment | Julia 1.13.3, KernelAbstractions 0.10 CPU backend; Intel Xeon Processor @ 2.10GHz (the model string the CPU reports), 4 vCPUs of a shared cloud VM; 1 and 4 threads |
| Figures | `wp-021-periodic-solve-2d/periodic_solve_t{1,4}.png`, each with its JSON sidecar (every timing row) |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review. Timings on a shared VM: each is the median of 5 samples, with the spread (min, max) in the sidecar. |

## 1. Setup

WP-021 adds the engine's periodic spectral solve (`lib/ParticleMeshCore/src/spectral.jl`, contract 14 of `docs/plan/architecture-draft.md` §0.7): `SpectralWorkspace(backend, T, grid)`, `inverse_laplacian_periodic!(φ, s, grid, ws)` and `gradient_periodic!(G, φ, grid)`, for `N = 1` and `N = 2`. The DFTs are written per axis in real arithmetic (separate real and imaginary arrays), one work item per output entry summing its line serially in increasing order; the twiddles come from tables built once from `sincospi` with the angle reduced exactly in integers, and every mode `k ≠ 0` is divided by minus the Yee-consistent symbol `Σ_d (2 sin(π k_d/n_d)/Δ_d)²`. The choices taken with the owner are in `questions/wp-021-periodic-solve.md`.

Timing method (`scripts/engine2d/periodic_solve_timing.jl`, with `timed` from `scripts/engine2d/common.jl`): one call of the solve followed by the gradient, on a smooth source with a nonzero mean; one warm-up call, then 5 samples, each the mean over enough calls to last about 0.1 s; the median is reported. Grids of cells 1/4 × 3/8 (N = 2) and 3/8 (N = 1), two guard cells. `cumsum_periodic!` is timed on the same 1D sources for comparison.

## 2. Tolerances (A7)

Measured on the CPU backend; the solve is bitwise across thread counts (T-312), so one measurement covers 1 and 4 threads. Every value is a proposal in `lib/ParticleMeshCore/test/tolerances.toml`.

| test | quantity | scale | measured (Float32, Float64, Dual) | proposed C |
|---|---|---|---|---|
| T-310 | divergence of the gradient minus (s − mean s), every node | (Σ_d n_d)·(max\|s\| + ‖L‖·max\|φ\|), ‖L‖ = Σ_d 4/Δ_d² | 0.0406, 0.0402, 0.0389 | 0.25 |
| T-310 | \|mean φ\| | (Σ_d n_d)·max\|φ\| | 0.00267, 0.00169, 0.00213 | 0.02 |
| T-311 | modes (1, 2) and (3, −5) on 16 × 12 against the closed form | (n₁ + n₂)·max\|reference\| | 0.109, 0.109, 0.109 | 0.5 |
| T-313 | N = 1 gradient against `cumsum_periodic!` | n·(max\|a\|·Δ + ‖L‖·Δ·max\|φ\|) | 0.0108, 0.0104, 0.0156 | 0.1 |
| T-314 | Dual primal against Float32; partials against the Float32 solve of the partials | (Σ_d n_d)·max\|reference\| | primal 0.0 (bitwise), partials 0.604 | 4 |

The scale of T-310 was chosen from a scan outside the test suite (the same input, the residual divided by `eps(T)`), from 16 × 12 to 256 × 192 cells and from 64 to 4096 cells. Without a size factor, the ratio `residual/(eps·(max|s| + ‖L‖·max|φ|))` grows from about 1 to 5 (N = 2) and from about 1 to 29 (N = 1), close to the square root of `Σ_d n_d` (divided by `√(Σ_d n_d)` it stays between 0.13 and 0.45); with the linear factor `Σ_d n_d` it falls from 0.04 to 0.004. The linear scale bounds the worst case of a sum of `n_d` terms, as T-172 does for the prefix sum, and the ratio does not grow under it, so the smallest grid sets the measured value. The scan was a scratch run without a manifest and is not committed; T-310 itself runs 16 × 12, 64 × 48 and 128 × 96 and 64 and 2085 cells and prints its ratios after every test run.

T-314's primal was bitwise equal to the Float32 run on every grid of the test: the Dual arithmetic on the value part performs the same operations, and the compiler fused none of them differently here. The test records this and does not assert it (A4). The largest partial ratio, 0.604, is that of G₂ along the first parameter on 16 × 12.

## 3. CPU time of one solve

Median seconds per call of the solve plus the gradient (`periodic_solve_t{1,4}.json`):

| cells | Float32, 1 thread | Float64, 1 thread | Dual, 1 thread | Float32, 4 threads | Float64, 4 threads | Dual, 4 threads |
|---|---|---|---|---|---|---|
| 16 × 12 | 0.27 ms | 0.30 ms | 0.35 ms | 0.44 ms | 0.55 ms | 0.52 ms |
| 64 × 48 | 1.9 ms | 1.5 ms | 3.4 ms | 1.1 ms | 1.1 ms | 1.6 ms |
| 128 × 96 | 8.8 ms | 9.0 ms | 19 ms | 3.9 ms | 4.5 ms | 9.7 ms |
| 256 × 192 | 71 ms | 68 ms | 226 ms | 19 ms | 23 ms | 55 ms |
| 512 × 384 | 0.56 s | 0.99 s | 2.4 s | 0.13 s | 0.23 s | 0.69 s |
| 1024 (N = 1) | 3.1 ms | 3.2 ms | 5.1 ms | 1.4 ms | 1.1 ms | 2.0 ms |
| 4096 (N = 1) | 46 ms | 45 ms | 78 ms | 22 ms | 13 ms | 22 ms |
| 16384 (N = 1) | 0.79 s | 0.76 s | 1.4 s | 0.21 s | 0.21 s | 0.36 s |

The cost follows the operation count `prod(n)·Σ_d n_d`: doubling both axes of the 2D grid multiplies the time by 7.9 (Float32, 256 × 192 to 512 × 384, 1 thread; 8 expected), and quadrupling a 1D grid by 15 to 17 (1024 to 4096 and 4096 to 16384 cells, 1 thread; 16 expected from the `n²` terms per transform). `cumsum_periodic!` takes 0.04 to 0.13 ms at every 1D size of the scan, so at `N = 1` the 1D module keeps its prefix sum (T-313 shows the two agree).

**Acceptable at CI scale up to 256 × 192 cells and 4096 cells at N = 1**: at most 0.23 s per solve at 1 thread in every type, and the solve runs once, at initialization. The test sizes (up to 128 × 96 and 2085 cells) take at most 19 ms per solve. At 512 × 384 one solve takes 0.6 to 2.4 s at 1 thread, still acceptable once per run in `scripts/`, not in a test that repeats it per type, mode and launch shape.

## 4. What a production solver would need

The solve is O(`prod(n)·Σ_d n_d`): at a few thousand cells per axis that is minutes on a CPU core (extrapolated from section 3 with the operation count, not measured), and it holds the whole grid on one device. A production path, still called once at initialization, would need:

- An O(`prod(n)·Σ_d log n_d`) FFT written as KernelAbstractions kernels with a fixed butterfly order (radix-2 or mixed radix, generic in `T` so that it keeps `Dual` and the A8 matrix), so that the association order stays a property of the operator and the result stays bitwise across launch shapes and thread counts (A4). FFTW takes only Float32 and Float64, and a vendor FFT (cuFFT) does not document its order, so neither can be the Reference path; either could be a Fast path behind the same interface, compared within a measured tolerance.
- A real-to-complex first axis (half the spectrum by conjugate symmetry), which halves the work and the spectrum arrays.
- For multi-GPU, a distributed transform (slab or pencil decomposition with all-to-all transposes between the axes), which joins the fill/fold/exchange communication points of §0.7's proposals; or, avoiding the transposes, a multigrid solve with a fixed number of sweeps in a fixed order.
- The accuracy holds as measured: the residual grows like the square root of `Σ_d n_d` (section 2), so a Float32 solve at a few thousand cells per axis would stay near `10⁻⁵` relative to `max|s| + ‖L‖·max|φ|` (extrapolated, not measured).

None of this is needed for the 2D slice of §0.8, whose CI sizes are at most a few hundred cells per axis.

## 5. Verification status and open items

- T-310 to T-316 pass on the CPU backend for Float32, Float64 and `Dual{Float32,3}`, T-312 at 1 and 4 threads and three launch shapes.
- Not run on CUDA (a non-goal of WP-021): the kernels follow §0.4, but the bits across CUDA launch shapes and the CPU–CUDA tolerance are unmeasured; #215 asks for the GPU runner.
- Not tested under Reactant (parked, A9).
- Not independently reviewed.
