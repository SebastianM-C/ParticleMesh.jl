# WP-038: the binomial current filter matched in ParticleMesh, Smilei and PIConGPU

| | |
|---|---|
| Work package | WP-038 (goals A2, A5, A7) |
| Commits | generators, check and cases `3586030`; the check's per-precision bound `4ae0613`; tests `ff37b18`; the analyses of the Smilei and PIConGPU runs at `1ad62f4` (clean) |
| Runs | ParticleMesh `em2d-pm-landau-oblique-2d-filter1-20261105T181425`, `em2d-pm-landau-oblique-2d-filter2-20261105T181539` (commit `3586030`, clean); Smilei `smilei-landau-oblique-2d-filter1-32ca4bf5`, `smilei-landau-oblique-2d-filter2-7da4b2d2` (prepared at `3586030`, run on the CPU box, branch `runs/wp038-smilei-filter-1`); PIConGPU `picongpu-landau-oblique-2d-filter1-fp64-32ca4bf5` (prepared at `3586030`, run on an RTX 4080 SUPER, branch `runs/wp038-picongpu-filter1`) |
| Input hashes | SHA-256 of every input in each run's `manifest.toml` under `wp-038-matched-filter/<run>/` |
| Environment | ParticleMesh: Julia 1.13.3, KernelAbstractions CPU backend, 2 threads, Reference mode, Float64. Smilei v5.1 (`438d43d`), CPU image, AMD EPYC 9655, 16 patches. PIConGPU 0.8.0 (`6bb3470e`), CUDA, NVIDIA GeForce RTX 4080 SUPER, FP64 |
| Figures | `wp-038-matched-filter/<run>/plots/filtered-gauss.png` with its JSON sidecar, per run |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## 1. What the codes do

Read from the pinned sources (paths in `scripts/crosscode/README.md`):

- **Smilei** `CurrentFilter(model = "binomial", passes = [p])` applies p passes of (1, 2, 1)/4 along x and then y to Jx, Jy and Jz on their own extents, with a guard exchange after each pass, after the current is summed and before the Yee update. That is ParticleMesh's `BinomialFilter(p)`. It has no GPU code path.
- **PIConGPU** `--currentInterpolation binomial` adds to E the current smoothed by (1, 2, 1; 2, 4, 2; 1, 2, 1)/16, which is one pass along each axis, with no pass count.

So the generators emit Smilei's block for any p on the CPU build, PIConGPU's binomial for p = 1, and refuse the rest (T-550, T-551).

## 2. The check

With the current filtered by F, which commutes with the backward differences, D⁻·Eⁿ − Fρⁿ is constant (T-441). So between two outputs, for every Fourier mode m of the plain residual r = D⁻·E − (ρ − mean ρ),

Δr̂(m) = (H_p(m) − 1) Δρ̂(m),   H_p(m) = Π_d cos^{2p}(π m_d / n_d).

The observable `filtered_gauss` (`scripts/crosscode/analysis.jl`) is max_m |Δr̂ − (H_p − 1)Δρ̂| / max_m |Δρ̂| between the first and the last field output, from each code's dumps of Ex, Ey and ρ (each code's staggers read from the dump positions, T-553). It is also evaluated for p − 1 and p + 1, to show that it tells the pass counts apart. The thermal noise fills every mode, so one run checks the whole transfer function, Nyquist rows included. Both cases (`landau-oblique-2d-filter1`, `-filter2`) are the oblique Landau case at WP-029 §3's size: 32 × 32 cells, 16 × 16 electrons per cell, 5 % perturbation, 610 steps to t = 18, field outputs every 10 steps. The pass bound is 1e-9 for Float64, registered before any run.

## 3. Results

| Run | p | deviation at t = 17.99 | largest over the 61 later outputs | p − 1 | p + 1 | modes above the floor | max\|Δρ̂\| |
|---|---|---|---|---|---|---|---|
| ParticleMesh | 1 | 2.7e-14 | 4.5e-13 | 0.070 | 0.045 | 1021 | 2.5e-2 |
| Smilei | 1 | 5.4e-15 | 7.7e-14 | 0.067 | 0.045 | 1021 | 2.5e-2 |
| PIConGPU | 1 | 9.3e-15 | 1.3e-13 | 0.081 | 0.045 | 1020 | 2.5e-2 |
| ParticleMesh | 2 | 2.4e-14 | 4.4e-13 | 0.045 | 0.043 | 1019 | 2.7e-2 |
| Smilei | 2 | 5.0e-15 | 8.2e-14 | 0.045 | 0.043 | 1023 | 2.7e-2 |

All three codes keep the filtered Gauss law to roundoff for their own pass count, and every neighbouring count misses it by 0.04 to 0.08, twelve orders of magnitude above. In each `filtered-gauss.png` the measured |Δr̂|/|Δρ̂| of every mode lies on 1 − H_p(m). The p + 1 entries of the two cases agree across codes to all printed digits (0.0451, 0.0429) because there the maximum sits on the perturbed mode (2, 1), where |H_p − H_{p+1}| is a fixed number (H₁ − H₂ = 0.0451 at (2, 1)). The p − 1 entries differ between codes because their maximum sits on noise modes, which differ.

Secondary, reported only: the rate of mode (2, 1) against the filtered theory √H·landau_root(|k|λ_D/√H) (γ = 0.1619 for p = 1, 0.1705 for p = 2). With one seed it is noise-limited, as WP-029 found at this size. ParticleMesh gives 0.155 ± 0.020 (p = 1) and 0.118 ± 0.033 (p = 2), Smilei 0.146 ± 0.003 and 0.143 ± 0.025, all from 3 or 4 maxima. PIConGPU's fit found no maxima above its floor, so it reports none. These rates say nothing about the filter.

The plain Gauss residual at the end is 0.063 (p = 1) and 0.088 (p = 2) of max|ρ| in ParticleMesh, as expected with a filter. Each code's t = 0 Gauss check closes: ParticleMesh 7.7e-16 of max|ρ|, Smilei 2.7e-12 (after its iterative Poisson solve), PIConGPU 6.1e-14. PIConGPU's dumps have no flagged dump (largest relative change of a species' total charge 2.3e-15).

## 4. Limits, from the breakers

- The roundoff floor scales with eps. ParticleMesh in Float32 gives 6.7e-6 at the end and 1.1e-4 over the outputs, so 1e-9 is a Float64 bound. The analysis picks the bound by the run's precision and refuses a run without one; the registered cases give only the Float64 bound. A PIConGPU FP32 run needs a case with `tolerance_float32`.
- The deviation is relative to max|Δρ̂|, so a run whose charge barely changes (a cold case, or early outputs) has a higher floor: 2.1e-8 at a 1e-7 perturbation. An output with ρ unchanged is skipped.
- Variants of the ParticleMesh run (32 × 24 and 24 × 40 cells, 4 × 4 per cell, another seed, Fast mode, 0, 3 and 4 passes) all stay at 3e-15 to 3e-14, with the neighbours at 0.04 to 0.32.

## 5. Verification status and open items

- Tests: T-550 (Smilei namelist, refusals, unfiltered namelists unchanged), T-551 (PIConGPU arguments and refusals), T-552 (the check on exact synthetic fields: roundoff for p, O(1) for p ± 1 and for a wrong stagger; the floor), T-553 (the stagger read from dump positions, both codes' conventions), T-554 (the ParticleMesh driver over all its steps, both cases).
- `scripts/em2d/crosscode.jl compare` has no reference entry for the filter cases.
- The LWFA case: matched filtering there is limited to 0 or 1 pass, ParticleMesh refuses passes with the absorbing layer (#334), and the codes' filter at the moving window's edges is not compared.
