# WP-031: the common offsets from linear theory are not discretization error

| | |
|---|---|
| Work package | WP-031 (goals A5, A7) |
| Question | Are the offsets of the cold cases' growth rates from linear theory, common to Smilei, PIConGPU and ParticleMesh, discretization error? Two-stream −1.5 % (#168, #226), 2D Weibel −1.1 % (`reports/wp-023-em-step-2d.md`, "not verified"), 1D Landau below theory (#167) |
| Code | `scripts/crosscode/convergence.jl` (scan points, box request, summary, report), `convergence_report.jl`, `convergence_pm.jl` (ParticleMesh on the same points), `theory.jl` (`cold_ivp`: the exact linear solution); tests T-400 to T-405 |
| Scans | `scripts/crosscode/scans/two-stream-offset.toml`, `weibel-2d-offset.toml`, fixed before the runs; points derived from the committed case files by `derive_case` (T-402: the 128- and 256-cell two-stream points give the committed namelists of `two-stream.toml` and `two-stream-256.toml`) |
| Runs | Smilei v5.1 (CPU image `localhost/particlemesh-smilei:e4f0148b8952fc87`, id `7e2b08d0…`) on the CPU box, AMD EPYC 9655, 16 cores: branch `runs/wp031-two-stream-1` (13 points, 21 min), `runs/wp031-weibel-2d-1` (WEIBEL); ParticleMesh (electromagnetic module, Reference mode, Float64) on an Intel Core i9-14900K, 16 threads, commit `f014378` |
| Outputs | `wp-031-offset-convergence/two-stream/`, `weibel-2d/`, `pm-two-stream/`: each holds `summary/` (the box's or the local run's series per point, the derived cases, the manifests), `results.json`, the figures with their JSON sidecars and `manifest.toml` |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## 1. Method

A rate observable of these cases is a least-squares fit of log U over the samples where U lies in a window below saturation (`growth_from_window`). Linear theory gives the growing root of the dispersion relation; a run starts from a density perturbation, which excites every root, and the window samples the sum. So the offset of a fitted rate from the root has three possible parts:

1. **the transient**: the other roots, still present in the window. It is a property of the case's initial state and fit rule, the same in every correct code at any resolution;
2. **the onset of saturation**: the window reaches up to 3 × 10⁻² (two-stream) or 10⁻² (Weibel) of the saturated energy, where nonlinearity already slows the growth; also independent of the resolution;
3. **discretization error**, which goes to zero with Δx and Δt.

The first part is computed exactly. `cold_ivp` (theory.jl) writes the linearized cold-fluid equations of the case's mode as dy/dt = M y, with the case's initial state (the loaded density perturbation, velocities unperturbed, B = 0, E from Gauss's law), and solves them by the matrix exponential. Its eigenvalues are the dispersion roots (T-400, to 10⁻¹²), Gauss's law is conserved along it (T-401, 10⁻¹³), and the field grows at the root on the growing eigenvector (T-401). The report step applies the run's fit to this solution at the **same sample times** as the run's window, giving γ_lin. The **residual** (γ − γ_lin)/θ, θ the root, removes part 1 exactly and is zero for a run that follows the linear solution (T-403: zero on the solution itself, and exactly δ on a solution distorted by e^{2δt}). What the residual keeps is parts 2 and 3; the scan separates them: part 3 goes to zero as Δx and Δt do, part 2 does not, and part 2 stays fixed when the perturbation amplitude α changes (the window sits at the same fraction of saturation), while part 1 changes with α (the window comes after more or fewer e-foldings).

Each scan varies Δx at the case's Courant number (Δt ∝ Δx), Δt at fixed Δx, α × 1/10 and × 10, the particles per cell, and the seed. The output cadence in steps scales with Δt so that every point writes at the case's output times. The cases are cold with regular positions: no random number enters, and the points `seed1` give results identical to all printed digits, so one run per point suffices.

## 2. Two-stream (`two-stream.toml`, θ = 0.350876)

`growth_rate`, the field energy of Ex, window [10⁻³, 3 × 10⁻²] of saturation (`two-stream/results.json`):

| point | cells | Δt | γ | γ/θ − 1 | linear solution, same samples | residual | U/U_lin at window start, end | wall (s) |
|---|---|---|---|---|---|---|---|---|
| n0064 | 64 | 0.0152302 | 0.345310 | −1.586 % | −1.035 % | −0.552 % | 1.0098, 0.9886 | 3 |
| n0128 (the case) | 128 | 0.0076151 | 0.345612 | −1.500 % | −1.035 % | −0.466 % | 1.0072, 0.9884 | 7 |
| n0256 | 256 | 0.00380755 | 0.345926 | −1.411 % | −0.961 % | −0.450 % | 1.0039, 0.9858 | 40 |
| n0512 | 512 | 0.00190377 | 0.345938 | −1.407 % | −0.961 % | −0.447 % | 1.0018, 0.9838 | 121 |
| n1024 | 1024 | 0.000951887 | 0.345939 | −1.407 % | −0.961 % | −0.446 % | 1.0006, 0.9827 | 480 |
| n0128-dt2 | 128 | 0.00380755 | 0.345841 | −1.435 % | −0.961 % | −0.474 % | 1.0020, 0.9832 | 13 |
| n0128-dt4 | 128 | 0.00190377 | 0.345832 | −1.437 % | −0.961 % | −0.477 % | 0.9994, 0.9806 | 24 |
| n0128-dt8 | 128 | 0.000951887 | 0.345828 | −1.439 % | −0.961 % | −0.478 % | 0.9981, 0.9794 | 46 |
| n0512-dt4 | 512 | 0.000475944 | 0.345932 | −1.409 % | −0.961 % | −0.448 % | 0.9999, 0.9819 | 461 |
| n0128-a0.1 | 128 | 0.0076151 | 0.349748 | −0.321 % | +0.169 % | −0.490 % | 1.0063, 0.9872 | 9 |
| n0128-a10 | 128 | 0.0076151 | 0.290401 | −17.235 % | −16.754 % | −0.482 % | 1.0121, 0.9889 | 12 |
| n0128-ppc0.1 (100 per beam) | 128 | 0.0076151 | 0.345612 | −1.500 % | −1.035 % | −0.466 % | 1.0072, 0.9884 | 4 |
| n0128-seed1 | 128 | 0.0076151 | 0.345612 | −1.500 % | −1.035 % | −0.466 % | 1.0072, 0.9884 | 10 |

The 128- and 256-cell points reproduce the committed i9-14900K runs of WP-012 (0.3456117, 0.3459258) to the printed digits.

- **The offset does not go to zero.** It converges to −1.407 % (1024 cells, Courant number kept; −1.409 % at 512 cells and a quarter of the Δt). Between the case and the finest point it moves by 0.09 %, of which 0.07 % is the window taking one sample fewer from 256 cells on (163 instead of 164; the linear solution over the same samples moves by the same 0.07 %).
- **The transient is −0.96 % to −1.04 %**, the linear solution through the run's own samples.
- **The residual converges to −0.446 %** at second order in Δx (observed orders 2.5, 2.1 on the first three refinements; Richardson to order 2: −0.4462 %). At the case's resolution its discretization part is −0.019 % (Δx at the Courant number kept) and +0.012 % (Δt alone, at 128 cells: −0.466 % → −0.478 % at Δt/8); at most 2 × 10⁻⁴ of θ. The Δt part converges at first order (observed orders 1.8, then 1.0), as a term from the time of the loaded momenta does: Smilei takes them at −Δt/2, the linear solution at t = 0 (U/U_lin at the window start moves 1.0072 → 0.9981 from Δt to Δt/8).
- **The residual is the onset of saturation.** At α × 1/10 and × 10 the offset moves from −0.32 % to −17.2 %, the linear solution with it (+0.17 %, −16.75 %), while the residual stays at −0.49 % and −0.48 %. It sits at a fixed fraction of saturation, as nonlinearity does. The run follows the linear solution to 0.2 % until the window (U/U_lin at its start: 1.0006 at 1024 cells), then falls behind it to 0.983 at the window's end at every resolution (`ratio-to-linear.png`); the local rate (`local-rate.png`) overlays the linear solution's to t ≈ 16 and drops below it inside the window.
- **Particles per cell and the seed do not enter**: 100 instead of 1000 electrons per beam and cell, and seed 1 instead of 20261031, give the same γ to all printed digits.

![two-stream offset and residual](wp-031-offset-convergence/two-stream/offset-vs-resolution.png)
![two-stream ratio to the linear solution](wp-031-offset-convergence/two-stream/ratio-to-linear.png)
![two-stream local rate](wp-031-offset-convergence/two-stream/local-rate.png)

So the −1.5 % is −1.0 % transient, −0.45 % nonlinear onset and −0.03 % discretization. The secondary estimate from |Ê₁|² of the dumps (20 points, #168's second question) is the same story at a coarser cadence: −3.94 % in the limit, −3.65 % transient, −0.30 % residual, independent of Δx beyond 256 cells.

**The window sensitivity (#168), labelled post hoc.** Over five windows at 1024 cells the measured offset and the linear solution's move together, and the residual grows with the window's upper end:

| window (fraction of saturation) | points | γ/θ − 1 | linear solution | residual |
|---|---|---|---|---|
| [10⁻³, 3 × 10⁻²] (registered) | 163 | −1.407 % | −0.961 % | −0.446 % |
| [10⁻³, 10⁻²] | 112 | −4.974 % | −4.757 % | −0.217 % |
| [3 × 10⁻³, 3 × 10⁻²] | 106 | +2.854 % | +3.556 % | −0.702 % |
| [10⁻², 3 × 10⁻²] | 51 | +0.377 % | +1.506 % | −1.129 % |
| [3.3 × 10⁻⁴, 3 × 10⁻²] | 208 | −1.748 % | −1.420 % | −0.328 % |

The several-per-cent swings between windows are the transient (column 4); the residual is largest in the window nearest saturation.

### ParticleMesh at the same Δt (#226)

ParticleMesh (`convergence_pm.jl`, the settings of `scripts/em1d/crosscode.jl`) on the 128-cell points at Δt, Δt/2, Δt/4, Δt/8 against Smilei on the same points (`pm-two-stream/comparison.json`): the fit over Smilei's window samples, and U_PM/U_Smilei.

| Δt | γ_PM/γ_Smilei − 1, same samples | U_PM/U_Smilei at t = 0 | at window start | at window end | ParticleMesh's own rule: offset, residual |
|---|---|---|---|---|---|
| 0.0076151 | −9.92 × 10⁻⁵ | 1.000171 | 0.98992 | 0.98975 | −1.417 %, −0.485 % |
| 0.0038076 | −4.84 × 10⁻⁵ | 1.000171 | 0.99503 | 0.99494 | −1.417 %, −0.485 % |
| 0.0019038 | −2.39 × 10⁻⁵ | 1.000171 | 0.99760 | 0.99755 | −1.417 %, −0.485 % |
| 0.0009519 | −1.16 × 10⁻⁵ | 1.000171 | 0.99888 | 0.99886 | −1.417 %, −0.485 % |

The difference #226 found, 1 % in U and 10⁻⁴ in γ, halves with each halving of Δt: it is first order in Δt and vanishes in the limit, as the half-step kick of the initial momenta (ParticleMesh takes the loaded momenta at t = 0, Smilei at −Δt/2) predicts. The constant 1.00017 at t = 0 is the other term #226 names (per-particle against cell-centred density sampling); it does not depend on Δt. ParticleMesh's own fit reports the same nonlinear residual as Smilei's within 0.02 %. ParticleMesh was run at 128 cells only.

![ParticleMesh over Smilei](wp-031-offset-convergence/pm-two-stream/code-ratio.png)

## 3. 2D Weibel (`weibel-2d.toml`, θ = 0.263725)

WEIBEL

## 4. 1D Landau (#167): not scanned

The 1D Landau ensemble rate is noise-limited: the 8-seed ensembles have standard errors of 0.012 to 0.016 (8 % to 10 % of θ, WP-012), and the deficit #167 asks about is 2.0 pooled errors. The discretization parts measured here are 2 × 10⁻⁴ of θ (two-stream, kΔx = 0.049) and WEIBELDISC (Weibel); at the Landau case's kΔx = 0.098 a second-order error would be about four times the two-stream one, near 10⁻³ (an extrapolation, not measured). Resolving a change of 10⁻³ against a 9 % standard error takes about 8 × 10³ times the eight seeds per resolution, far beyond any affordable scan, so the same scan cannot resolve it and was not run. Part of #167's deficit is the relativistic push with each code's load (#268: −3.6 % to −4.6 % in 2D at this temperature); the rest stays a question of noise.

## 5. Verification status

- Verified: T-400 to T-405 on the CPU; the scans' Smilei runs reproduce the committed WP-012 runs; the figures were inspected before the numbers were recorded.
- Not verified: an independent review of this report; the linear solution against an independent implementation (it is checked against the dispersion roots and Gauss's law, which are this repository's own derivations, the Weibel relation unverified against the literature, `theory.jl`).

## 6. Cost

Smilei on the AMD EPYC 9655 (16 cores; one thread per patch, so the 1024-cell two-stream point ran 32 threads on 16 cores). Wall times per point in the tables. ParticleMesh two-stream at 128 cells, 16 threads of the i9-14900K: 60 s, 118 s, 236 s, 476 s for Δt to Δt/8.
