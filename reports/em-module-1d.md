# WP-014: the 1D electromagnetic module on the three A5 cases

| | |
|---|---|
| Work package | WP-014 (goals A1 to A10) |
| Commits | case runs: `19e7d4aac062484ddb97c0043b4ca262b0c80af1`; timing runs: `6315a04885c11c8fd2bb213052ba461ae0f7ced8` (adds only a host-side bound check on the step index, which changes no result) |
| Runs | `em1d-weibel-20261031T222620`, `em1d-two-stream-20261031T222637`, `em1d-landau-20261031T222807`, `em1d-timing-t1-20261031T223332`, `em1d-timing-t16-20261031T223356` |
| Input hashes | SHA-256 of every input (engine and module sources, case file, theory, scripts, both manifests) in each run's manifest: `em-module-1d/<run>.manifest.toml`. The runs wrote absolute input paths. The copies here have them made relative to the repository root, with the hashes unchanged, and `scripts/em1d/common.jl` now writes them that way |
| Clean tree | Weibel and the timings ran on clean trees. The two-stream and Landau manifests say `dirty = true`: the one tracked difference was the `status` line of `trace/wp/WP-014-em-module-1d.md`, edited while they ran, which is not a run input |
| Environment | Julia 1.13.3, KernelAbstractions 0.10.4 CPU backend, AcceleratedKernels 0.5.11; Intel Core i9-14900K; case runs at 16 threads, timings at 1 and 16 |
| Figures | `em-module-1d/<case>_{signal,energies,conservation,phase_space}.png` and `timing-t{1,16}-*.png`, each with its JSON sidecar; `<run>.summary.json` per case run |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review. Every tolerance and expected value below is a proposal (A7) |

## 1. Setup

`lib/ParticleMeshElectromagnetic` is the first physics module (A9). It has the 1D Yee layout, with ρ, Ey and Ez at the cell edges and Ex, By and Bz at the centres, the relativistic Boris pusher, and its own step order, with the current from the engine's `deposit_flux!` (WP-009). The order is: store the old position, push, sort, deposit the flux and fold, then Yee. The initial Ex is `cumsum_periodic!` of the deposited charge. Units are A1's throughout: time in 1/ω₀, length in c/ω₀, density in n_c, fields in mₑcω₀/e.

`scripts/em1d/case.jl` reads the case files of `scripts/crosscode/cases/` unchanged. It fits the observable the case file registers, with the same functions as the Smilei runs (`scripts/crosscode/analysis.jl`). For Landau and two-stream it also runs the electrostatic reference (the WP-010 spike's step, `lib/ParticleMeshCore/test/es_reference.jl`) on the same grid, Δt, particles, amplitude and seed. All case runs are Float64, quadratic shape, `Reference()` mode.

| Run | Case file | Overrides | Particles | Steps |
|---|---|---|---|---|
| Weibel | `weibel.toml` (new, WP-014) | none | 2 × 128 × 64 = 16 384 | 1716 (Δt = 0.0233, t = 40) |
| Two-stream | `two-stream.toml` | none | 2 × 128 × 1000 = 256 000 | 5253 (Δt = 0.00762, t = 40) |
| Landau | `landau.toml` | 16 384 particles per cell (case: 400 000), amplitude 5 % (case: 1 %) | 1 048 576 | 2681 (Δt = 0.00933, t = 25) |

The Landau overrides are noise budget, not physics. The case file's 400 000 particles per cell are 25.6 million electrons, Smilei's setting. With 1 % amplitude, a few thousand particles per cell leave no maximum of the damped field energy above the case's floor criterion. The test configuration of T-214 (8192 particles per cell, 5 %) gets three, and 4096 gets two (`questions/wp-014-em-module-choices.md` §3).

## 2. Results

### Rates against linear theory and the electrostatic reference

Rates are of the field amplitude, in ω₀ (= ω_p: the electron density is 1 n_c in every case).

| Case | Observable | Measured γ | Theory γ | Relative to theory | Electrostatic reference γ | Module relative to the reference |
|---|---|---|---|---|---|---|
| Landau, kλ_D = 1/2 | maxima of U(Ex), t ≥ 3, above 10 × floor (3 maxima, t = 4.76 to 9.21) | 0.16259 | 0.15336 | +6.0 % | 0.16375 | −0.7 % |
| | frequency from the same maxima | ω = 1.41249 | 1.41566 | −0.2 % | | |
| Two-stream, kv₀ = √(3/8) ω_p | U(Ex) in [10⁻³, 3·10⁻²] of saturation (655 points, t = 13.8 to 18.8) | 0.34584 | 0.35088 (with γ₀³) | −1.4 % | 0.34816 | −0.7 % |
| Weibel, k = 2 ω_p/c, v₀ = 0.3 c | U(Bz) in [10⁻⁴, 10⁻²] of saturation (381 points, t = 12.3 to 21.2) | 0.26113 | 0.26373 | −1.0 % | none (no transverse fields in the reference) | |

Against Smilei, from the WP-003 stage 1 runs in `reports/crosscode-smilei-1d/README.md`:
- **Two-stream:** the case file is run identically, and Smilei gives 0.34561 ± 0.00090, so the module is +0.07 % from it.
- **Landau:** Smilei gives 0.1445 ± 0.0130, but at the case's 1 % amplitude and 400 000 particles per cell. These are not matched inputs, so the numbers are not comparable.
- WP-012's stage 1b numbers do not exist yet (WP-012 is `proposed`).

Readings:
- The Landau rate is noise-limited. Three maxima span 4.4/ω₀ and 0.97 e-foldings of the field energy. The module and the reference sit within 0.7 % of each other and both about 6 % above theory, as the reference did at the same particle count. The two curves of `landau_signal.png` overlap through t ≈ 15 and separate only in the noise.
- The two-stream rate is not noise-limited. A cold quiet start gives the same γ to 15 digits at 256 and at 1000 particles per cell. The −1.4 % against the cold-fluid γ₀³ value is a property of the discretization (Δx = 0.08 v₀/ω_p, the fit window), shared within 0.7 % with the electrostatic leapfrog and within 0.07 % with Smilei.
- Weibel's −1.0 % is the first measurement of the cold-fluid expected value proposed for A5. The relation is derived in `lib/ParticleMeshElectromagnetic/test/theory.jl`, its header in A1 units, and is unverified against the literature. `weibel_signal.png` shows an initial beat with the oscillating electromagnetic root (to t ≈ 10), the exponential phase parallel to theory, and saturation at t ≈ 32. Over the linear phase to saturation the beams lose 0.029 of kinetic energy, and the Bz energy rises by 0.020 (T-216). In phase space, the momenta u_y = ±γ₀v₀ = ±0.3145 bend into x-modulated filaments at saturation.

### Conservation

| Case | Gauss residual, max over checks every 50 steps, / (eps·n·max\|ρ\|) | Largest relative change of the total energy (fields + kinetic) | Final relative change |
|---|---|---|---|
| Landau | 2.42 (55 checks) | 2.1 × 10⁻⁶ | −2.1 × 10⁻⁶ |
| Two-stream | 1.63 (107 checks) | 5.7 × 10⁻⁴ | +1.3 × 10⁻⁵ |
| Weibel | 0.39 (36 checks) | 2.8 × 10⁻⁴ | +1.7 × 10⁻⁵ |

Gauss's law is never re-solved after t = 0. It holds to roundoff over thousands of steps because the flux deposition is charge-conserving (T-110) and Ex changes only by −Δt·Jx. The test suite's bound (T-212, every step, three element types, both modes) is 1.31·eps·n·max|ρ|·√k, proposed C = 4. The energy changes are the largest excursions during the nonlinear phase. The final values are an order of magnitude smaller, and the proposed T-217 bounds are 1e-5, 2e-3 and 1e-3.

In the two electrostatic cases, Ey, Ez, Bx, By, Bz, Jy and Jz, and the momenta u_y and u_z, stay +0.0 bit for bit for the whole run (T-213, at the test configurations).

### Time per particle-step (no thresholds)

ns per particle-step of the whole electromagnetic step (push, sort, flux, kinetic-energy deposit, Yee, diagnostics), Float64, 50 steps after a 5-step warm-up, one run each (`timing-t{1,16}-*.json`):

| Case (particles) | 1 thread, Reference | 1 thread, Fast | 16 threads, Reference | 16 threads, Fast |
|---|---|---|---|---|
| Landau, 64 cells (262 144) | 145.8 | 111.4 | 64.1 | 100.4 |
| Two-stream, 128 cells (256 000) | 142.0 | 115.6 | 41.1 | 52.3 |
| Weibel, 128 cells (131 072) | 127.7 | 112.3 | 45.7 | 47.9 |

- The Reference step at one thread is dominated by the Reference `deposit_flux!`, about 16 × `deposit!` on the CPU (#93, WP-009's, with no effect on its bits).
- At 16 threads, Fast is slower than Reference on 64 cells: four thousand particles per cell add atomically into a handful of nodes, and the contention costs more than the sorted reduction.
- The scaling from 1 to 16 threads is 2.3× to 3.5×. The WP-011 report measured run-to-run noise up to a factor of two at 16 threads; these are single runs.
- The throughput budget of A7 is on the GPU; nothing here is a budget.

## 3. Verification status

- The CPU test suite of `lib/ParticleMeshElectromagnetic` (T-210 to T-221) passes on one thread, with every measured ratio and proposed constant in `lib/ParticleMeshElectromagnetic/test/tolerances.toml`. Highlights:
  - **Yee:** the mode advances by the exact discrete ω to 0.17·eps·steps (T-210).
  - **Boris:** the gyrophase per step is 2·atan(ΩΔt/2γ) to 0.27·eps·steps (T-211).
  - **Determinism:** Reference mode is bitwise across reruns, launch shapes and 1 and 4 threads for all three cases and types (T-218).
  - **Gradient:** a ForwardDiff derivative through 322 module steps agrees with the `fd_scan` floor at 0.18 of it (T-219).
- Not reviewed independently. The Weibel relation is the module author's derivation, and its check is its agreement with the module to 1 %, which is not independent of the author. A literature check of the relation is open.

## 4. Open issues

- The electrostatic reference is still used by `scripts/es1d/`, by WP-012's column and by the engine's integration tests. Deleting it, and rewiring the scripts to the module, is follow-up #97.
- A matched Landau comparison with Smilei (same amplitude and particle count) is open. So is a Smilei Weibel run, which needs `transverse_drift` passed to Smilei (cross-code stage 2; `smilei.jl` refuses the key until then).
- The Fast-mode contention at high particle counts per cell, and the Reference flux cost (#93), are CPU observations for the GPU work of WP-013.
