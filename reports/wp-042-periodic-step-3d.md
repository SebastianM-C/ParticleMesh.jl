# WP-042: the periodic 3D3V step against linear Landau theory at full size

- With a 1 % density perturbation, the oblique Langmuir wave of mode (1, 1, 1) damps at the rate of linear theory within a few percent, inside the seed errors: +1.7 % ± 1.6 % at 64³ cells (57 million electrons, 8 seeds), −2.1 % ± 3.4 % at 48³ and +0.4 % ± 6.1 % at 32³, the errors the jackknife over seeds. The frequency is 0.5 % to 0.9 % below theory.
- Float32 and Float64 give the same rate: on the same four seeds at 64³ they agree to 2·10⁻⁵ of γ, and the mode amplitudes to 3·10⁻⁵ of their peak.
- At T-954's 10 % perturbation the rate is 13 % to 16 % above theory at every size, with errors under 1 %. The amplitude scan at 32³ (+16 %, +4.6 %, −1.8 % at 10 %, 5 %, 2 %) shows that this is the wave's nonlinearity, not the step: its initial bounce frequency √(kE) ≈ 0.32 is twice γ. T-954 keeps 10 % for a CI-sized signal and bounds the rate around its measured offset.
- Gauss's law holds at the end of every run to 1.2·10⁻⁵ of max|ρ| in Float32 and 1.7·10⁻¹⁴ in Float64, and the total energy drifts by at most 1.3·10⁻⁵.
- With the tests T-950 to T-958 this completes WP-042's sentences, so its status is `done`.

| | |
|---|---|
| Work package | WP-042 (goals A0, A2, A4, A8) |
| Code | `scripts/em3d/landau3d.jl`; the 10 % runs at d3fecc8, the 1 % runs at 5a90d76 (each run's `manifest.toml`, clean) |
| Runs | the ten directories under `wp-042-periodic-step-3d/`, each with its `manifest.toml`, `observables.json` and the `published.toml` of run branch `runs/wp042-landau3d` |
| Theory | `landau_rate_relativistic` (`scripts/crosscode/theory.jl`, T-160) for the Maxwell–Jüttner load at \|k\| = 10, v_th = 0.05: γ = 0.148574, ω = 1.411317 |
| Environment | Julia 1.13.4, CUDACore 6.5, an RTX 5090, Reference mode, quadratic shape; the wall times include waits for other jobs on the card |
| Status | measured, not independently reviewed |

## The case and the observable

T-954's case at full size: a cube of side 2π√3/|k| with |k| = 10 ω₀/c (so |k|λ_D = 0.5, as in the 2D case `landau-oblique-2d`), one electron species at v_th = 0.05 c over immobile ions, 6³ electrons per cell, Δt = 0.95 of the 3D Courant limit, run to t = 24. The observable is WP-023's `damping_rate_mode_ensemble`: the fit of |⟨k̂·Ê⟩_seeds|² at its maxima after t = 3 above ten times the noise floor (`scripts/crosscode/ensemble.jl`), with the jackknife over seeds as its error. The per-seed column applies the same rule to one seed at a time.

## Results

At 1 % (`julia +1.13 --project=scripts scripts/em3d/landau3d.jl table <dirs>`, pasted unchanged):

| size | particles | Δx/λ_D | steps | ensemble γ ± jackknife | rel. to theory | ω rel. | maxima | per-seed γ, mean ± std | Gauss, ∇·B at end (max) | energy drift (max) | s per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32³, 216 per cell, 8 seeds, Float32, δn/n 0.01 | 7077888 | 0.68 | 1287 | 0.1492 ± 0.0091 | 0.397 % ± 6.1 % | -0.89 % | 6 | 0.1449 ± 0.029 (8 fitted) | 8.5e-6, 2.7e-6 | 6.4e-6 | 10 |
| 48³, 216 per cell, 8 seeds, Float32, δn/n 0.01 | 23887872 | 0.453 | 1930 | 0.1455 ± 0.005 | -2.07 % ± 3.4 % | -0.71 % | 6 | 0.1443 ± 0.011 (8 fitted) | 8.6e-6, 3.0e-6 | 9.7e-6 | 72 |
| 64³, 216 per cell, 8 seeds, Float32, δn/n 0.01 | 56623104 | 0.34 | 2573 | 0.1511 ± 0.0023 | 1.7 % ± 1.6 % | -0.45 % | 6 | 0.15 ± 0.0093 (8 fitted) | 8.3e-6, 3.4e-6 | 1.3e-5 | 272 |
| 64³, 216 per cell, 4 seeds, Float64, δn/n 0.01 | 56623104 | 0.34 | 2573 | 0.1435 ± 0.003 | -3.45 % ± 2.0 % | -0.58 % | 5 | 0.1459 ± 0.01 (4 fitted) | 1.7e-14, 6.1e-15 | 1.0e-6 | 534 |

The Float64 row has the first four seeds only. The Float32 run's first four seeds give γ = 0.143449 against Float64's 0.143451, so the 5 % between the two rows is seed noise between four and eight seeds.

At larger amplitudes, the same table:

| size | particles | Δx/λ_D | steps | ensemble γ ± jackknife | rel. to theory | ω rel. | maxima | per-seed γ, mean ± std | Gauss, ∇·B at end (max) | energy drift (max) | s per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32³, 216 per cell, 4 seeds, Float32, δn/n 0.1 | 7077888 | 0.68 | 1287 | 0.1727 ± 0.0013 | 16.2 % ± 0.89 % | -2.0 % | 6 | 0.172 ± 0.002 (4 fitted) | 9.2e-6, 3.4e-6 | 9.6e-6 | 10 |
| 32³, 216 per cell, 4 seeds, Float32, δn/n 0.05 | 7077888 | 0.68 | 1287 | 0.1554 ± 0.0012 | 4.57 % ± 0.79 % | -1.0 % | 6 | 0.1554 ± 0.0028 (4 fitted) | 7.4e-6, 2.4e-6 | 3.5e-6 | 10 |
| 32³, 216 per cell, 4 seeds, Float32, δn/n 0.02 | 7077888 | 0.68 | 1287 | 0.1459 ± 0.0064 | -1.78 % ± 4.3 % | -0.96 % | 6 | 0.1471 ± 0.009 (4 fitted) | 6.7e-6, 2.7e-6 | 5.5e-6 | 10 |
| 48³, 216 per cell, 4 seeds, Float32, δn/n 0.1 | 23887872 | 0.453 | 1930 | 0.1681 ± 0.001 | 13.2 % ± 0.7 % | -1.1 % | 5 | 0.1681 ± 0.0021 (4 fitted) | 1.1e-5, 4.0e-6 | 7.7e-6 | 72 |
| 64³, 216 per cell, 4 seeds, Float32, δn/n 0.1 | 56623104 | 0.34 | 2573 | 0.1698 ± 0.00072 | 14.3 % ± 0.48 % | -1.5 % | 6 | 0.1688 ± 0.0034 (4 fitted) | 1.2e-5, 4.1e-6 | 5.1e-6 | 272 |
| 32³, 216 per cell, 4 seeds, Float64, δn/n 0.1 | 7077888 | 0.68 | 1287 | 0.1727 ± 0.0013 | 16.2 % ± 0.89 % | -2.0 % | 6 | 0.172 ± 0.002 (4 fitted) | 1.5e-14, 5.6e-15 | 7.2e-6 | 29 |

`wp-042-periodic-step-3d/landau3d.png` (sidecar `landau3d.json`) shows the 64³ Float32 run at 1 %: each seed's |k̂·Ê|², their mean, the fitted maxima and the theory's slope.

## Not shown

- No comparison with another code: neither Smilei nor PIConGPU has run this 3D case.
- No Weibel case in 3D at full size; the sentences of WP-042 ask for Landau or Weibel.
- `Dual{Float32,3}` at full size: T-955 checks its partials through the 3D step at test size.
- Fast mode at full size: T-951 checks Fast against Reference at test size.

## Data

`scripts/rundata/run-data.sh fetch reports/wp-042-periodic-step-3d/<run>/published.toml` brings back a run's per-seed series, its figure and its observables, and checks their hashes.
