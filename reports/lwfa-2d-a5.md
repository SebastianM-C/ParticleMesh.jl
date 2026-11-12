# LWFA 2D: A5's first case in ParticleMesh, Smilei and PIConGPU

| | |
|---|---|
| Work package | WP-053 (`trace/wp/WP-053-lwfa-three-codes.md`); goals A5, A7, and A3 for #362. Follows PR #336, which brought the PIConGPU column |
| Case | `scripts/lwfa/cases/lwfa-2d.toml`, λ₀/48, with `laser.envelope = "wavefront"` (sha256 `e7bc9e3f…`, in the run manifests), and `scripts/lwfa/cases/lwfa-2d-focus0.toml`, the same with the focus on the injection plane, in vacuum only |
| Commits | ParticleMesh's and PIConGPU's runs of §4 used the code of `6ead5bd` (the box requests `5b729a7` and `9f8a388` are that commit plus the prepared run directories); Smilei's runs of §4 the request `e79e5b7`, which is `df75c5c` plus its run directories. The comparisons: the commit in each `manifest.toml` under `lwfa-2d-a5/` (`dirty = false`) |
| Runs | ParticleMesh: CPU box requests `runs/lwfa-a5-2` (result `ab315a6`) and `runs/lwfa-a5-focus0` (result `e86a2b8`). Smilei: CPU box request `runs/lwfa-a5-3-smilei` (result `088fd90`), with the amplitude it is given corrected for its boundary condition; its runs on the two earlier requests, without the correction, are history (§3.2). The summaries stay on those branches. PIConGPU: gpu-run jobs `a5-2-pic-vacuum`, `a5-2-pic-plasma` and the focus-0 vacuum run; ParticleMesh on CUDA: `lwfa-a5-2-cuda-pm-plasma`. The summaries of these four are on no branch and are in `lwfa-2d-a5/runs/`. First runs (§3.1): `runs/lwfa-a5-1` and `lwfa-a5pic` |
| Input hashes | each run's `manifest.toml` as `manifest-<run>.toml` next to its comparison; the boxes' `box-status.toml` (`lwfa-a5-2-compare/`, `lwfa-a5-focus0-compare/`) and `box-status-smilei.toml` (`lwfa-a5-3-compare/`); the sha256 of the summaries compared under `summaries` in each comparison's `manifest.toml` |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode; 16 threads on an AMD EPYC 9655 (the CPU box), and CUDA on an RTX 4080 SUPER. Smilei v5.1 (`438d43d`, image `particlemesh-smilei:e4f0148b8952fc87`), 16 OpenMP threads, the same box. PIConGPU 0.8.0 (`6bb3470e`), CUDA, Float64, the RTX 4080 SUPER |
| Figures | `lwfa-2d-a5/lwfa-a5-3-compare/` (A5's case), `lwfa-a5-3-focus0-compare/` (focus on the plane), each PNG with its JSON sidecar; `lwfa-a5-2-cuda-pair/pair.json` (ParticleMesh on two backends); `lwfa-a5-smilei-pair/pair.json` (Smilei with and without the correction). History: `lwfa-a5-2-compare/comparison.json` and `lwfa-a5-focus0-compare/comparison.json` (Smilei uncorrected), `lwfa-a5-three-compare/comparison.json` (first runs); their figures are in this file's history, at `df75c5c` and `c9de0de`. The measured values below are keys of the `comparison.json` or `pair.json` named with them; ranges, largest values, counts and ratios are computed from those keys, the wall times and the box's step times are from the runs' summaries and the box status files, and the few numbers computed from the summaries or the maps directly say so where they stand |
| Status | measured; reviewed twice, the present version not yet. No number here is to leave the repository before the review of this version |
| Review | Two independent reviews by fresh numerics-reviewer subagents (claude-fable-5-2, 2026-11-12). The first, on the unmatched first runs, found that the three codes had launched three different pulses and that the spectra covered different regions (§3.1, §4.4). The second, on `c9de0de`, reproduced every table of the runs with Smilei uncorrected and found two wrong numbers, a hypothesis stated as fact, the wake's measure ending on a bucket's edge, and the pulse compared on the axis only; `df75c5c` answered those and derived Smilei's amplitude factor. Neither has seen the runs with Smilei corrected (§4) or §6. A third pass follows this commit |

`pm-plasma`'s manifest says `dirty = true`: the step before it on the box, `pm-vacuum` (same commit, `dirty = false`), rewrites its own tracked `manifest.toml`. This is inferred from the two manifests, not recorded by the run.

## 1. Setup

The case, in A1's units: a₀ = 2, waist 28, τ = 26 (1/e half width of the field), focus at x = 320; electrons of 0.01 n_c (λ_p = 62.8) behind 260 of vacuum and a 60 sin² ramp, 2 × 2 per cell, cold, ions immobile; Δx = λ₀/48, Δy = 4Δx, Δt at 0.95 of Yee's 2D limit; a window of 2304 × 384 cells moving at c from t = 256; 10 777 steps, 14 outputs 829 steps apart (t ≈ 0, 100, …, 1300). Yee, Esirkepov, quadratic shapes, Boris, no current filter, Float64 in all three.

One pulse on the lab plane x = 0 in all three codes: a cosine carrier counted from the envelope's peak, the 2D Gouy phase, envelope and carrier on one time along the curved wavefront, and one amplitude. How each code is made to launch it, with the source lines read, is in the headers of `scripts/lwfa/lwfa.jl` and `scripts/lwfa/picongpu.jl`: PIConGPU's `GaussianPulse` is that pulse; ParticleMesh launches it with `GaussianLaser(envelope = WavefrontTime())`; Smilei's namelist gets a focus counted from its ghost node, an envelope centre, a phase offset, and the case's a₀ divided by what its boundary condition launches per unit (`smilei_laser`, §3.2). The three share Δx, Δy and Δt, so their grids are one lattice in the lab frame and every comparison matches nodes by index, with nothing interpolated.

## 2. Measures

`compare_rows` in `scripts/lwfa/lwfa.jl`; a pair is written a–b (PM–S ParticleMesh–Smilei, PM–P ParticleMesh–PIConGPU, S–P Smilei–PIConGPU), and every relative rms difference is relative to b's rms.

- **Carrier phase, amplitude** (`carrier_phase`, `amplitude_relrms`, `amplitude_max`): from the lattice's own analytic signal on the axis, z(m) = E_y(m) + i·(9·(E_y(m − q) − E_y(m + q)) + E_y(m − 3q) − E_y(m + 3q))/16 with q a quarter wavelength (12 points). What is left of the carrier in |z| is of order (qΔx/τ)⁴ = 10⁻⁵ for this pulse (T-691, T-692).
- **Centroid** of E_y² on the axis (`centroid_x`).
- **Wake**: E_x on the axis from ParticleMesh's envelope peak back one and a half plasma wavelengths (720 cells), the same lattice points for all three pairs (`wake_Ex_relrms`, `wake_Ex_min`, `wake_Ex_max`); `wake_Ex_shift` is the whole-cell displacement of b's wake that fits a's best. An output has these only when |E_x| reaches 10⁻³ of the wave-breaking field in the window in both codes: from t = 400 here. The window ends inside the second bucket. A window of two plasma wavelengths (960 cells), which the first reviewed version used, ends from t = 600 on the jump of E_x at the back of the second bucket: the window's anchor moves in steps of half a laser wavelength (24 cells), and a few cells at the jump then carry a large part of every pair's squared difference (§4.3). `wake_sensitivity` repeats the measures for windows of 1.0, 2.0 and 2.5 λ_p, and `wake_anchor_sensitivity` for the window moved by 24 cells either way.
- **Off the axis** (`maps`, `map_rows`): the relative rms difference of the 2D maps of E_y, kept on every sixth column and every row at the three outputs t = 300, 700, 1200 of the plasma runs, on the lattice points two maps share: over all rows, over the beam's flank (0.9 to 1.6 waists from the axis, 25.2 ≤ |y| < 44.8) and over its far wing (2.15 to 2.85 waists, 60.2 ≤ |y| < 79.8). The two envelope conventions of §3.1 are one field on the axis of the launch plane and differ off it, so this is where a remaining mismatch of that kind would show. The 48 cells (a laser wavelength) at the back of the shared columns are left out, for the reason given in §4.3; the keys ending in `_with_back` keep them.
- **Spectra** in one unit, electrons per unit depth in n_c (c/ω₀)²; `bunch` is the highest energy group of a spectrum, the occupied bins above its widest empty gap (`bunch_group`); `top` the upper edge of the highest occupied bin; `above` the weight above 1, 5, 10, 20 and 40 mₑc².

In `comparison.json` for reference, not counted: the box maximum of |E_y| (sampled on the lattice, up to 1 − cos(π/48) = 0.21 % below a crest); the running-max envelope (`envelope_relrms`, which reads a carrier phase as a difference); ρ_e on the axis (`wake_rho_relrms`: a few narrow spikes, whose relative rms moves by tens of percent with a displacement of a cell or with the code that places the window).

## 3. The runs before these: two rounds of mismatched launches

### 3.1 The first runs: three different pulses

The runs of 5 November (`lwfa-2d-a5/lwfa-a5-three-compare/comparison.json`) compared three pulses, and this is the record of what the mismatch was. The first version of this report, with their tables, is in this file's history (`1080c9a`); their figures and PIConGPU's first-run summaries are in the history at `c9de0de`.

| on the axis in vacuum, t = 200 to 1300 | PM–S | PM–P | S–P |
|---|---|---|---|
| carrier phase a − b (rad) | +1.9785 to +1.9790 | −0.0011 to +0.0004 | −1.9795 to −1.9785 |
| centroid a − b (c/ω₀) | +0.249 falling to −0.145 | −0.014 falling to −0.405 | −0.264 to −0.260 |

- **Smilei.** `LaserGaussian2D` gives the boundary envelope(t − φ(y))·sin(t − φ(y)): a sine of the run's time and no Gouy phase, where the others have a cosine from the envelope's peak; and the Silver–Müller condition applies it on the outermost ghost node, two cells behind xmin. For the case's numbers that is a phase of delay + ½·atan(320/392) − π/2 − 2Δx = 1.9787 rad mod 2π and a lag of 2Δx = 0.2618 c/ω₀: the two constant offsets measured.
- **ParticleMesh.** Its default envelope is in t − delay alone, the same at every y (`AxisTime`); PIConGPU's and Smilei's follow the curved wavefront. Past the plane the `AxisTime` pulse arrives on the axis later by (w²/(4|R|))/(1 + κ²w⁴), κ = (1/L − 1/|R|)/2, with w = 36.1 and |R| = 800.2 on the plane and L the distance travelled (T-696, and launched through the antenna in T-709): up to 0.408 c/ω₀. That closed form, with no free parameter, is the measured motion of ParticleMesh's centroid against PIConGPU's to 0.0044 c/ω₀ at each of the twelve outputs. Neither convention is a defect; `AxisTime` stays ParticleMesh's default and `WavefrontTime` is an opt-in (T-695 to T-697).
- **So the first runs' Smilei–PIConGPU distance was Smilei's two launch offsets**, not a spread for equivalent inputs: 5 % in the running-max envelope, and in the wake's E_x (two plasma wavelengths behind the peak) 4.2 % to 6.2 % at the nine outputs t = 500 to 1300, which a shift of two cells took down to 1.1 % to 1.7 %. At t = 400, where the wake is just forming, it was 2.17 %, the best shift one cell, and 1.87 % was left. Over the three pairs the first runs' wake distances ran from 0.17 % (ParticleMesh–PIConGPU at t = 400) to 6.2 %.
- **#362.** ParticleMesh's `ekin_max` was 1.17 to 2.41 mₑc² before the laser reached the plasma: the window's loader fills the vacuum with particles of zero weight (the density enters through the weights, A3), and the laser makes them quiver. `pm_spectrum` now takes the maximum over particles of positive weight (T-690).

### 3.2 The second runs: Smilei's amplitude

With carrier and envelope matched (`lwfa-a5-2-compare/comparison.json`, `lwfa-a5-focus0-compare/comparison.json`; the tables are in this file's history at `df75c5c`), the three carriers agreed to 0.7 mrad and the centroids to 0.002 c/ω₀ in vacuum, and Smilei's amplitude was above the other two at every output: its peak 0.166 % to 0.214 % above PIConGPU's with the focus ahead, 0.164 % to 0.186 % with it on the plane.

**The factor is Smilei's boundary condition.** Its Silver–Müller update at xmin (`ElectroMagnBC2D_SM.cpp:63` to `66` and `245`) is, at normal incidence, (B₀ + B₁)/2 + (E₀ half a step before + E₀ half a step after)/2 = 2b: the mean of the new B_z over the two points either side of the boundary node plus the mean of that node's E_y over the step around the new B's time, set to twice the profile b the namelist prescribes at that time. For the grid's own right-going plane wave those two means are cos(kΔx/2) and cos(ω₀Δt/2) of the wave, k the Yee wavenumber, so the wave launched is

    2/(cos(kΔx/2) + cos(ω₀Δt/2))

of the amplitude prescribed: 1.001984 on this grid, 1 + (Δx² + Δt²)/16 to second order, and 1.004474 at λ₀/32. Smilei's update run on a 1D Yee grid launches that factor to 10⁻¹⁰, for both case files and three other pairs of Δx and Δt (T-710). `smilei_laser` gives Smilei the case's a₀ divided by the factor (T-510), and the Smilei runs of §4 are made so. The prediction written before those runs, a Smilei amplitude within a few hundredths of a percent of the other two, held (§4.1).

**What 0.198 % of amplitude does, in one code.** Smilei's plasma run without the correction against the one with it (`lwfa-a5-smilei-pair/pair.json`, a the uncorrected run): the same code and inputs but for a₀. Amplitude and wake as relative rms differences in percent, the phase a − b in mrad, the centroid in 10⁻³ c/ω₀, the extrema as a/b − 1 in percent:

| t | amplitude | carrier phase | centroid | wake E_x | wake minimum | wake maximum | wake / amplitude |
|---|---|---|---|---|---|---|---|
| 400 | 0.198 | +0.05 | +0.03 | 0.185 | +0.13 | +0.24 | 0.93 |
| 500 | 0.196 | +0.43 | +1.40 | 0.512 | +0.24 | +0.18 | 2.61 |
| 600 | 0.200 | +0.93 | +3.06 | 0.598 | +0.24 | +0.27 | 2.99 |
| 700 | 0.211 | +1.38 | +2.59 | 0.636 | +0.28 | +0.38 | 3.02 |
| 800 | 0.221 | +1.78 | +2.15 | 0.637 | +0.27 | +0.50 | 2.89 |
| 900 | 0.224 | +2.18 | +2.37 | 0.608 | +0.35 | +0.27 | 2.72 |
| 1000 | 0.224 | +2.59 | +2.87 | 0.614 | +0.19 | +0.21 | 2.74 |
| 1100 | 0.223 | +3.02 | +3.35 | 0.617 | +0.25 | +0.29 | 2.76 |
| 1200 | 0.223 | +3.46 | +4.01 | 0.594 | +0.34 | +0.41 | 2.66 |
| 1300 | 0.225 | +3.91 | +3.96 | 0.572 | +0.33 | +0.33 | 2.54 |

Before the plasma (t = 100 to 300) the amplitude difference is 0.1984 %, the factor, and the phase and centroid differences are roundoff. In the plasma the wake's E_x changes by 2.5 to 3.0 times the amplitude's change from t = 500, the carrier phase by 3.9 mrad by the end, the centroid by up to 0.004 c/ω₀, and the wake's maximum by 0.50 % at t = 800 against 0.18 % to 0.41 % at the other outputs. These are the differences the second runs showed between Smilei and the other two codes (a wake 0.19 % to 0.57 % from PIConGPU's, a carrier drifting to 3.4 mrad from it, a wake maximum at t = 800 twice as far off as at its neighbours), and the previous version of this report listed them as hypotheses or as unexplained. They were the 0.2 % of amplitude.

## 4. The matched runs

ParticleMesh's and PIConGPU's runs are those of the second runs; Smilei's are new, with the corrected amplitude.

### 4.1 Do the three codes launch one pulse? The laser in vacuum

![the distances of the three pairs against time, matched runs](lwfa-2d-a5/lwfa-a5-3-compare/pair_distances.png)

A5's case, focus 320 ahead of the plane (`vacuum`, `picongpu_vacuum`, `smilei_picongpu_vacuum` in `lwfa-a5-3-compare/comparison.json`). Carrier phase a − b in mrad and centroid a − b in 10⁻³ c/ω₀:

| t | phase: PM–S | PM–P | S–P | centroid: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|
| 200 | +0.23 | −0.23 | −0.46 | +0.46 | −1.52 | −1.98 |
| 300 | +0.13 | −0.40 | −0.53 | +0.10 | −1.54 | −1.64 |
| 400 | −0.01 | −0.56 | −0.55 | −0.40 | −1.28 | −0.88 |
| 500 | −0.17 | −0.67 | −0.50 | −0.70 | −0.48 | +0.22 |
| 600 | −0.29 | −0.70 | −0.41 | −0.74 | +0.32 | +1.06 |
| 700 | −0.34 | −0.65 | −0.31 | −0.58 | +0.93 | +1.52 |
| 800 | −0.36 | −0.58 | −0.22 | −0.40 | +1.24 | +1.64 |
| 900 | −0.35 | −0.51 | −0.16 | −0.26 | +1.35 | +1.60 |
| 1000 | −0.33 | −0.44 | −0.11 | −0.17 | +1.33 | +1.50 |
| 1100 | −0.31 | −0.39 | −0.07 | −0.11 | +1.26 | +1.38 |
| 1200 | −0.29 | −0.34 | −0.05 | −0.07 | +1.16 | +1.22 |
| 1300 | −0.27 | −0.30 | −0.03 | −0.08 | +1.01 | +1.09 |

Amplitude: relative rms difference and the peak's ratio a/b − 1, in percent (`amplitude_relrms`, `amplitude_max`):

| t | rms: PM–S | PM–P | S–P | peak: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|
| 200 | 0.049 | 0.066 | 0.018 | +0.050 | +0.065 | +0.016 |
| 300 | 0.059 | 0.065 | 0.009 | +0.059 | +0.065 | +0.005 |
| 400 | 0.064 | 0.056 | 0.008 | +0.064 | +0.055 | −0.008 |
| 500 | 0.061 | 0.040 | 0.021 | +0.060 | +0.039 | −0.021 |
| 600 | 0.052 | 0.022 | 0.030 | +0.051 | +0.021 | −0.029 |
| 700 | 0.041 | 0.008 | 0.034 | +0.040 | +0.007 | −0.033 |
| 800 | 0.031 | 0.005 | 0.034 | +0.030 | −0.002 | −0.032 |
| 900 | 0.024 | 0.009 | 0.032 | +0.023 | −0.008 | −0.031 |
| 1000 | 0.018 | 0.012 | 0.030 | +0.017 | −0.011 | −0.028 |
| 1100 | 0.014 | 0.013 | 0.027 | +0.013 | −0.013 | −0.026 |
| 1200 | 0.011 | 0.014 | 0.025 | +0.010 | −0.013 | −0.023 |
| 1300 | 0.009 | 0.014 | 0.023 | +0.008 | −0.013 | −0.022 |

- **The prediction held.** Smilei's peak amplitude is within 0.033 % of PIConGPU's (+0.016 % to −0.033 %) and within 0.064 % of ParticleMesh's; it was 0.166 % to 0.214 % above PIConGPU's. At t = 200 the difference to PIConGPU went from +0.214 % to +0.016 %, the factor's 0.198 %. The phases and centroids are those of the second runs to the digits shown, as a change of amplitude leaves them in vacuum.
- **One pulse, to these numbers.** The three carriers agree to 0.70 mrad, the centroids to 0.002 c/ω₀ (a sixty-fifth of a cell), the amplitudes to 0.07 %. No pair stands apart: Smilei–PIConGPU is the closest in amplitude (0.03 %), ParticleMesh–Smilei in phase and centroid.
- **From t = 500 Smilei is 0.02 % to 0.03 % below PIConGPU.** The estimate made before the run for what the plane-wave factor leaves out was of this sign and size: the condition is the normal-incidence one and launches a component at angle α with E_y lower by about α²/4, 0.03 % for ⟨α²⟩ = 1/waist². The estimate was not checked beyond that agreement.

### 4.2 The focus on the injection plane

`lwfa-2d-focus0.toml`: the wavefront is flat where the pulse is launched, so nothing of the launch depends on its curvature (`lwfa-a5-3-focus0-compare/comparison.json`). Phase in mrad, centroid in 10⁻³ c/ω₀, amplitude rms in percent:

| t | phase: PM–S | PM–P | S–P | centroid: PM–S | PM–P | S–P | amplitude: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|
| 200 | −0.20 | −0.69 | −0.49 | −0.21 | −0.76 | −0.56 | 0.060 | 0.036 | 0.024 |
| 300 | −0.30 | −0.69 | −0.39 | −0.51 | −0.15 | +0.36 | 0.050 | 0.018 | 0.032 |
| 400 | −0.35 | −0.64 | −0.29 | −0.56 | +0.40 | +0.95 | 0.039 | 0.006 | 0.034 |
| 500 | −0.36 | −0.56 | −0.20 | −0.49 | +0.75 | +1.24 | 0.030 | 0.005 | 0.034 |
| 600 | −0.35 | −0.49 | −0.14 | −0.41 | +0.92 | +1.33 | 0.023 | 0.009 | 0.032 |
| 700 | −0.33 | −0.43 | −0.10 | −0.35 | +0.98 | +1.32 | 0.018 | 0.012 | 0.029 |
| 800 | −0.31 | −0.38 | −0.07 | −0.30 | +0.98 | +1.28 | 0.014 | 0.013 | 0.027 |
| 900 | −0.28 | −0.33 | −0.05 | −0.25 | +0.96 | +1.20 | 0.011 | 0.014 | 0.024 |
| 1000 | −0.26 | −0.30 | −0.03 | −0.24 | +0.80 | +1.04 | 0.009 | 0.013 | 0.022 |
| 1100 | −0.26 | −0.28 | −0.02 | −0.52 | +0.62 | +1.15 | 0.008 | 0.014 | 0.022 |
| 1200 | −0.27 | −0.25 | +0.02 | −0.48 | +1.24 | +1.73 | 0.005 | 0.018 | 0.022 |
| 1300 | −0.22 | −0.15 | +0.07 | +0.65 | +2.17 | +1.52 | 0.005 | 0.020 | 0.017 |

The same sizes as with the focus 320 ahead: phases within 0.7 mrad, centroids within 0.0022 c/ω₀, amplitudes within 0.06 %, Smilei's peak 0.013 % to 0.034 % below PIConGPU's. What the three launches still differ by does not come from the wavefront's curvature.

### 4.3 The laser and the wake in plasma

![E_x and ρ_e on the axis at t ≈ 500, 800, 1200, three codes](lwfa-2d-a5/lwfa-a5-3-compare/wake_axis.png)

The laser (`plasma`, `picongpu_plasma`, `smilei_picongpu_plasma`). Phase in mrad, centroid in 10⁻³ c/ω₀, amplitude rms in percent:

| t | phase: PM–S | PM–P | S–P | centroid: PM–S | PM–P | S–P | amplitude: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|
| 400 | +0.01 | −0.55 | −0.56 | −0.37 | −1.26 | −0.88 | 0.064 | 0.056 | 0.008 |
| 500 | +0.03 | −0.53 | −0.56 | −0.01 | +0.08 | +0.09 | 0.063 | 0.040 | 0.023 |
| 600 | +0.06 | −0.46 | −0.52 | −0.88 | +0.65 | +1.54 | 0.062 | 0.027 | 0.036 |
| 700 | +0.09 | −0.37 | −0.45 | −0.60 | +1.04 | +1.63 | 0.057 | 0.017 | 0.045 |
| 800 | +0.16 | −0.25 | −0.41 | +0.08 | +0.86 | +0.78 | 0.052 | 0.014 | 0.046 |
| 900 | +0.25 | −0.15 | −0.40 | +0.55 | +0.30 | −0.25 | 0.042 | 0.013 | 0.040 |
| 1000 | +0.35 | −0.07 | −0.42 | +0.77 | −0.35 | −1.12 | 0.033 | 0.009 | 0.031 |
| 1100 | +0.44 | −0.01 | −0.46 | +0.75 | −0.94 | −1.69 | 0.022 | 0.006 | 0.021 |
| 1200 | +0.53 | +0.03 | −0.50 | +0.58 | −1.45 | −2.03 | 0.018 | 0.008 | 0.016 |
| 1300 | +0.61 | +0.07 | −0.54 | +0.50 | −1.57 | −2.07 | 0.017 | 0.010 | 0.015 |

The wake's E_x on the axis: relative rms difference in percent, over one and a half plasma wavelengths behind the peak (`wake_Ex_relrms`, the measure) and over two (`wake_sensitivity`). The best whole-cell shift is 0 for every pair at every output and both windows (`wake_Ex_shift`), so the difference after the shift is the same number.

| t | 1.5 λ_p: PM–S | PM–P | S–P | 2 λ_p: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|
| 400 | 0.081 | 0.062 | 0.036 | 0.081 | 0.062 | 0.036 |
| 500 | 0.179 | 0.122 | 0.079 | 0.195 | 0.135 | 0.094 |
| 600 | 0.225 | 0.108 | 0.132 | 0.272 | 0.162 | 0.128 |
| 700 | 0.214 | 0.066 | 0.166 | 0.326 | 0.137 | 0.266 |
| 800 | 0.232 | 0.044 | 0.211 | 0.411 | 0.152 | 0.370 |
| 900 | 0.174 | 0.036 | 0.168 | 0.334 | 0.168 | 0.336 |
| 1000 | 0.139 | 0.023 | 0.137 | 0.218 | 0.165 | 0.289 |
| 1100 | 0.113 | 0.021 | 0.106 | 0.143 | 0.029 | 0.129 |
| 1200 | 0.090 | 0.025 | 0.083 | 0.115 | 0.032 | 0.098 |
| 1300 | 0.084 | 0.030 | 0.075 | 0.103 | 0.039 | 0.084 |

Why the two-wavelength number is the secondary one. From t = 600 the jump of E_x at the back of the second bucket lies 914 to 954 cells behind ParticleMesh's envelope peak, in the last 46 cells of that 960-cell window, and within 20 cells of it sit 31 % to 57 % of ParticleMesh–Smilei's squared difference, 34 % to 97 % of ParticleMesh–PIConGPU's and 20 % to 77 % of Smilei–PIConGPU's; of the 720-cell window's, none. (Computed from the summaries with `compare`'s own functions, the jump taken as the largest change of E_x over 8 cells between 1.5 and 2.2 λ_p behind the peak; not keys of `comparison.json`.) The two-wavelength number is therefore in large part how each code places one steep front and where the window's end, which moves in 24-cell steps with the anchor, cuts it. ParticleMesh–PIConGPU's falls from 0.165 % at t = 1000 to 0.029 % at t = 1100 on that window and smoothly on the interior one.

The largest value over the ten outputs for other windows (`wake_sensitivity`, `wake_anchor_sensitivity`):

| window | PM–S | PM–P | S–P |
|---|---|---|---|
| 1.0 λ_p | 0.224 | 0.131 | 0.165 |
| 1.5 λ_p, the measure | 0.232 | 0.122 | 0.211 |
| 1.5 λ_p, moved 24 cells back | 0.243 | 0.124 | 0.221 |
| 1.5 λ_p, moved 24 cells forward | 0.220 | 0.121 | 0.200 |
| 2.0 λ_p | 0.411 | 0.168 | 0.370 |
| 2.5 λ_p | 0.956 | 0.197 | 0.958 |

The 2.5 λ_p window is longer than what the three windows share behind the peak (at t = 700 the maps' shared columns begin 1093 cells behind it), so it ends on the first cells of Smilei's window (see the maps below).

The wake's extrema in the 1.5 λ_p window, m_ecω₀/e (`wake_Ex_min`, `wake_Ex_max`), and a/b − 1 in percent:

| t | min: PM | Smilei | PIConGPU | PM–S | PM–P | S–P | max: PM | Smilei | PIConGPU | PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | −0.00481 | −0.00480 | −0.00480 | +0.099 | +0.100 | +0.000 | 0.02934 | 0.02932 | 0.02933 | +0.056 | +0.038 | −0.018 |
| 500 | −0.08907 | −0.08900 | −0.08903 | +0.077 | +0.051 | −0.026 | 0.07853 | 0.07852 | 0.07851 | +0.013 | +0.018 | +0.005 |
| 600 | −0.09362 | −0.09358 | −0.09359 | +0.044 | +0.032 | −0.012 | 0.10354 | 0.10347 | 0.10350 | +0.073 | +0.042 | −0.031 |
| 700 | −0.09405 | −0.09399 | −0.09403 | +0.067 | +0.021 | −0.046 | 0.10987 | 0.10969 | 0.10984 | +0.162 | +0.031 | −0.131 |
| 800 | −0.09262 | −0.09258 | −0.09262 | +0.047 | +0.001 | −0.046 | 0.10960 | 0.10956 | 0.10963 | +0.043 | −0.021 | −0.064 |
| 900 | −0.08980 | −0.08970 | −0.08979 | +0.113 | +0.015 | −0.098 | 0.10456 | 0.10449 | 0.10455 | +0.063 | +0.009 | −0.054 |
| 1000 | −0.08621 | −0.08620 | −0.08620 | +0.014 | +0.006 | −0.009 | 0.09982 | 0.09980 | 0.09982 | +0.017 | +0.003 | −0.014 |
| 1100 | −0.08299 | −0.08303 | −0.08297 | −0.053 | +0.019 | +0.072 | 0.09428 | 0.09428 | 0.09429 | −0.009 | −0.011 | −0.002 |
| 1200 | −0.07980 | −0.07980 | −0.07980 | +0.003 | −0.001 | −0.004 | 0.08933 | 0.08930 | 0.08933 | +0.042 | +0.008 | −0.034 |
| 1300 | −0.07699 | −0.07695 | −0.07698 | +0.051 | +0.011 | −0.040 | 0.08476 | 0.08474 | 0.08476 | +0.024 | +0.001 | −0.023 |

The wake's difference against the laser's, per output: `wake_Ex_relrms` over `amplitude_relrms` of the same pair.

| t | PM–S | PM–P | S–P |
|---|---|---|---|
| 400 | 1.27 | 1.11 | 4.30 |
| 500 | 2.87 | 3.03 | 3.47 |
| 600 | 3.64 | 4.03 | 3.62 |
| 700 | 3.79 | 3.94 | 3.73 |
| 800 | 4.44 | 3.02 | 4.56 |
| 900 | 4.15 | 2.72 | 4.18 |
| 1000 | 4.25 | 2.58 | 4.35 |
| 1100 | 5.17 | 3.69 | 5.15 |
| 1200 | 4.95 | 3.09 | 5.03 |
| 1300 | 5.06 | 3.12 | 5.03 |

- **The wake is the same in the three codes to 0.23 %** in E_x over 1.5 λ_p and to 0.16 % in its extrema, and no pair is displaced by a cell. With Smilei uncorrected the pairs with it were 0.17 % to 0.57 % apart; they are 0.04 % to 0.23 % now.
- **The pairs with Smilei are still the further ones from t = 700**: 0.07 % to 0.23 %, where ParticleMesh–PIConGPU is 0.02 % to 0.07 %. Before t = 700 ParticleMesh–PIConGPU (0.06 % to 0.12 %) is no closer than Smilei–PIConGPU (0.04 % to 0.13 %).
- **Not all of what is left is amplitude.** One code's wake changes by 2.5 to 3.0 times a change of its laser's amplitude (§3.2). ParticleMesh–PIConGPU's wake difference is 2.6 to 4.0 times its amplitude difference from t = 500; the two pairs with Smilei are at 2.9 to 5.2 and 3.5 to 5.2. At t = 800, Smilei–PIConGPU's 0.046 % of amplitude would account for 0.13 % of wake at a ratio of 2.9, and 0.21 % is measured. What the rest is was not determined.
- **The wake's maximum at t = 700** is the one extremum where a code stands out: Smilei's is 0.13 % and 0.16 % below the others', where every other entry of the pairs with Smilei is within 0.12 %.
- **The laser in plasma.** Phases within 0.61 mrad, centroids within 0.0021 c/ω₀, amplitudes within 0.064 %, for all three pairs. Smilei's carrier, which drifted 3.4 mrad from PIConGPU's in the second runs, now keeps the offset it has in vacuum (−0.40 to −0.56 mrad). ParticleMesh's carrier moves by 0.6 mrad against both other codes over the run (+0.01 to +0.61 against Smilei, −0.55 to +0.07 against PIConGPU); in vacuum it moves against both by 0.5 mrad as well (§4.1).

Off the axis: the 2D maps of E_y, relative rms difference in percent on the lattice points two maps share, without the 48 cells at the back of the shared columns (`maps`). t = 300 is before the pulse reaches the plasma.

| t | rows | PM–S | PM–P | S–P |
|---|---|---|---|---|
| 300 | all | 0.058 | 0.070 | 0.054 |
| 300 | flank, 0.9 to 1.6 waists | 0.099 | 0.121 | 0.091 |
| 300 | far wing, 2.15 to 2.85 waists | 0.552 | 0.685 | 0.503 |
| 700 | all | 0.098 | 0.062 | 0.092 |
| 700 | flank | 0.138 | 0.129 | 0.143 |
| 700 | far wing | 0.435 | 0.443 | 0.427 |
| 1200 | all | 0.104 | 0.056 | 0.103 |
| 1200 | flank | 0.118 | 0.074 | 0.120 |
| 1200 | far wing | 0.386 | 0.293 | 0.419 |

- **One pulse off the axis too, to these numbers.** Over the whole map every pair is within 0.10 % (with Smilei uncorrected its pairs were at 0.17 % to 0.36 %), and within 0.14 % in the flank. For scale, computed from `laser_field` and not measured in a run: on the launch plane the two envelope conventions of §3.1 differ by 1.4 % over all rows, 2.6 % in the flank and 10 % in the far wing.
- **The far wing is where all three pairs are furthest apart**, 0.29 % to 0.69 %, of similar size for the three pairs. At the focus the field in that band is a hundredth of the axis's and less. Not explained.
- **The back of Smilei's window.** With the back 48 cells kept (`…_with_back`), the far wing at t = 700 is 4.18 % for both pairs with Smilei and 0.44 % for ParticleMesh–PIConGPU; at t = 300, before there is a wake, the two versions are equal. The shared columns begin at the back of Smilei's window, and the difference is Smilei's first cells: in the two kept columns nearest that back at t = 700 (12 cells), and the three nearest at t = 1200 (18 cells), Smilei's E_y differs from ParticleMesh's by 20 % to 40 % of the column's field (the root of the squares summed over its rows), where PIConGPU's differs by 1 %. In the columns after, Smilei's differs by 1 % to 5 % and PIConGPU's by 0.3 % to 1.7 %; these are columns far behind the pulse, where the field is small. These column numbers are from the runs' `maps.h5`, not keys of `comparison.json`. The numbers are those of the second runs: correcting Smilei's amplitude did not change them. The cause is not established; what stands there is the wake's transverse field at Smilei's xmin boundary, where its absorbing condition acts. The wake's window on the axis ends more than 300 cells ahead of these columns at both outputs.

![n_e/n₀ at t ≈ 1200, three codes](lwfa-2d-a5/lwfa-a5-3-compare/density_maps.png)

The three density maps at t ≈ 1200: the same two buckets, crests and laser-driven ripples at the same lab positions. The figure's sidecar holds the plotted values to four decimals (1.9 MB), which is finer than the colour scale shows.

### 4.4 Electrons

![the electron spectrum at the end, three codes, one unit](lwfa-2d-a5/lwfa-a5-3-compare/spectrum.png)

What each histogram covers (`spectrum` in `comparison.json`; `region` there says "each code's own"):

- **ParticleMesh and Smilei: one region**, the lattice cells both windows' plasma covers at the output (`spectrum_region`, e.g. cells 7992 to 10277 at the end). Smilei's two strip diagnostics worked (`spectrum_failed` is empty in its summary). The two totals agree to 3 × 10⁻⁴ or better at every output (1.2 × 10⁻⁵ at the end: 602.878 against 602.871).
- **PIConGPU: its whole simulated domain**, up to a local domain of 2328 cells behind the window's back. Its deck writes no electrons: with every species in the file output, the plasma run computed its steps and then did not exit, and left a file that could not be read (`scripts/lwfa/picongpu.jl`'s header; #412). So PIConGPU's totals and low energies do not compare with the others' (above 5 mₑc² it holds 0.63 at t = 1100, the others 0.0013). That the excess is electrons behind the window, which the other two codes have dropped, is an inference from those totals and from what the plugin covers; no dump was there to check it against. What is compared is the highest energy group and the highest occupied bin, on the assumption that they do not depend on the region. That too is not checked against a dump; what supports it is that PIConGPU's bunch weighs 1.248813 × 10⁻³ at all six outputs from t = 800, whatever its domain holds behind the window at each. The reader that would bin an electron dump over the region is tested on synthetic files only (T-704, T-705) and has not met PIConGPU.

The bunch, the highest energy group of each spectrum (`bunch`):

| | ParticleMesh | Smilei | PIConGPU |
|---|---|---|---|
| weight (n_c (c/ω₀)²) | 1.247061 × 10⁻³ from t = 800 | 1.282582 × 10⁻³ at t = 800 and 900, 1.264578 × 10⁻³ from t = 1000 | 1.248813 × 10⁻³ from t = 800 |
| energy range at t = 1300 (mₑc²) | 55.2 to 67.2 | 57.8 to 67.2 | 57.2 to 67.2 |
| occupied bins at t = 1300 (0.2 mₑc² each) | 36 | 36 | 36 |
| of those, bins with less than half a plateau macro-particle's weight | 32 | 31 | 32 |
| empty gap below the group at t = 1300 (mₑc²) | 44.0 | 35.2 | 34.6 |
| highest occupied bin ends at, t = 800 … 1300 | 24.2, 33.6, 42.4, 51.2, 59.4, 67.2 | 24.2, 33.6, 42.4, 51.0, 59.4, 67.2 | 24.2, 33.6, 42.4, 51.0, 59.4, 67.2 |

- **Maximum energy.** Smilei's and PIConGPU's highest occupied bins are the same at all six outputs; ParticleMesh's is theirs at five and one bin (0.2 mₑc²) higher at t = 1100.
- **Weight.** ParticleMesh is 0.14 % below PIConGPU. Smilei's group from t = 1000 is 1.26 % above PIConGPU's, and ParticleMesh's 1.39 % below Smilei's; at t = 800 and 900 the gap rule gives Smilei a group that reaches lower (13.8 to 24.2 mₑc² at t = 800, where ParticleMesh's is 17.8 to 24.2), and the two numbers are 2.70 % and 2.77 %. At t = 800 PIConGPU's group is separated from the rest by one mₑc² only (`gap`), and Smilei's by 2.6.
- **Smilei's heavier bunch is not its laser.** With Smilei uncorrected its group weighed 1.269786 × 10⁻³, 1.68 % above PIConGPU's, and its stronger laser was one of two candidates. With the amplitude matched the excess is still there, and by a measure that does not depend on the gap rule it grew: over the region ParticleMesh and Smilei share, Smilei's weight above 10 mₑc² is 3.2 % to 3.3 % above ParticleMesh's at t = 800 to 1200 (1.289 × 10⁻³ against 1.249 × 10⁻³; 2.3 % to 2.6 % in the second runs), and above 20 mₑc² 2.1 % to 2.5 % from t = 900 (1.8 % in the second runs). The difference is 0.23 of one plateau macro-particle's weight.
- **What the bunch is.** One plateau macro-particle carries n₀·Δx·Δy/4 = 1.713 × 10⁻⁴ (`macro_weight`); the bunch weighs 7.3 to 7.4 of those but occupies 36 bins, 31 or 32 of them with less than half of one. It is at least about 35 macro-particles, most of them light: a particle's weight is the density where it was loaded, so they come from the foot of the ramp, where the three codes assign weights by three rules (ParticleMesh at the particle, PIConGPU at the cell centre, Smilei where its particle creator puts it, `lwfa.jl`'s header). That leaves Smilei's load rule as the candidate for its heavier bunch; it was not tested. ParticleMesh and PIConGPU load by different rules and agree to 0.14 %. The case does not test injection (A5 plans a second, injecting one).
- **ParticleMesh against Smilei over the common region**, weight above 1 mₑc² (`above`): within 0.5 % at six of the ten outputs from t = 400 (t = 400 to 600 and 1100 to 1300), and 1.5 % to 3.6 % at t = 700 to 1000 (3.6 % at t = 900: 4.090 against 3.947), on a population that rises to 4 and falls to 1 while the window passes over it. In the second runs the same comparison had the other sign at t = 1000 (−4.0 %); it is sensitive to more than the codes' agreement.

ParticleMesh's `ekin_max` (#362) is 0 at t = 0 to 200 and 8 × 10⁻⁶ mₑc² at t = 300 in this run (`ekin_max_pm`), where the first run had 1.17 to 2.41, and from t = 400 it lies in the highest occupied bin of ParticleMesh's histogram at every output (67.15 in the bin that ends at 67.2 at t = 1300).

### 4.5 ParticleMesh on two backends

The plasma case in Reference mode, Float64, on the CPU box and on CUDA (`lwfa-a5-2-cuda-pair/pair.json`, `compare-pair`; a is the CPU run): the same code in the same mode, so the pair is what roundoff and the order of the reductions do to these observables over 10 777 steps.

| over the outputs with the measure | largest CPU–CUDA difference |
|---|---|
| carrier phase (rad) | 1.6 × 10⁻¹⁶ |
| centroid (c/ω₀) | 3.4 × 10⁻¹³ |
| amplitude, relative rms | 3.5 × 10⁻¹³ |
| E_y, E_x, ρ_e on the axis, largest pointwise difference | 3.6 × 10⁻¹², 2.8 × 10⁻¹², 2.4 × 10⁻¹¹ |
| wake E_x, relative rms | 1.7 × 10⁻¹³ |
| wake extrema, a/b − 1 | 2.9 × 10⁻¹³ |
| spectrum | totals, the bunch's bins and range and the highest bin identical at every output; the bunch's weight to 3 × 10⁻¹⁶ |

The two backends are not bit for bit the same run (Reference mode's reproducibility is per backend, A4) and differ at 10⁻¹¹ and below at the end. That is seven orders or more below the differences between two codes in §4.3, so none of those is roundoff.

### 4.6 Cost, as measured

| Run | Wall time | What the time covers | Hardware |
|---|---|---|---|
| ParticleMesh vacuum / plasma | 58.3 s / 2589.0 s | the stepping loop with its outputs (`seconds`); the box's steps with start-up took 82 s / 2606 s | AMD EPYC 9655, 16 threads |
| ParticleMesh plasma, CUDA | 393.8 s | the stepping loop with its outputs, the diagnostics on host copies | RTX 4080 SUPER |
| Smilei vacuum / plasma | 46.5 s / 386.9 s | the `smilei` process; the box's steps 47 s / 387 s | AMD EPYC 9655, 16 OpenMP threads |
| PIConGPU vacuum / plasma | 63.7 s / 142.2 s | the `picongpu` run after compilation; compiling took 78.0 s and 75.4 s more | RTX 4080 SUPER |
| focus on the plane, vacuum | ParticleMesh 63.1 s, Smilei 47.5 s, PIConGPU 63.7 s | as above | as above |

Single runs. No performance conclusion is drawn: ParticleMesh ran in Reference mode, which sorts every step, in Float64, and has had no optimization round; Smilei's ions are frozen particles; the codes ran on different hardware and with different diagnostics. A5's performance comparison (the same card, particle-steps per second) is not this.

## 5. Verification status and open issues

- **Reviewed twice, this version not** (the Review row). One run of each code at one resolution. Nothing here is a validation claim.
- **What is shown.** With carrier, envelope and amplitude matched, the three codes agree on this case to 0.7 mrad, 0.002 c/ω₀ and 0.07 % in the laser in vacuum, to 0.23 % in the wake's E_x over 1.5 λ_p and 0.16 % in its extrema, to 0.10 % in the maps of E_y over all rows, and to one bin of 0.2 mₑc² in the maximum energy. The trapped bunch's weight agrees to 0.14 % between ParticleMesh and PIConGPU and to 1.4 % with Smilei. The three launch mismatches that separated the earlier runs are understood from the sources, and each correction was measured in a run (§3, §4.1).
- **Settled by the corrected Smilei run** (§3.2, §4):
  - Smilei's amplitude: the launch factor 1.001984 of its boundary condition, predicted and found.
  - Smilei's carrier drift in plasma (3.4 mrad): its amplitude.
  - Smilei's wake excess (0.17 % to 0.57 % in E_x) and its wake maximum at t = 800: its amplitude, for the larger part; 0.04 % to 0.23 % is left.
- **Refuted by it.** That Smilei's heavier bunch was its stronger laser: the excess stays with the amplitude matched (§4.4).
- **Unexplained**, each with its numbers above:
  - Smilei's bunch, 1.3 % heavier than the other two by the group and 2.1 % to 3.2 % heavier than ParticleMesh's by the weight above 20 and 10 mₑc² over the shared region. Its load rule at the ramp's foot is the candidate, untested.
  - What is left of the wake's difference for the pairs with Smilei from t = 700 (0.07 % to 0.23 %, against 0.02 % to 0.07 % for ParticleMesh–PIConGPU), more than their amplitude differences account for at the measured response; and Smilei's wake maximum at t = 700, 0.13 % and 0.16 % below the others'.
  - Smilei's amplitude 0.01 % to 0.04 % below PIConGPU's from t = 500 after the correction. The oblique-incidence estimate has this sign and size and was not checked.
  - The far wing of the maps: 0.29 % to 0.69 % between every pair, where the rest of each map agrees to 0.14 %.
  - Smilei's E_y in the 12 to 18 cells at the back of its window, different from both other codes' by 20 % to 40 % of the field there, unchanged by the amplitude correction. The cause proposed, its boundary acting on the wake's field, was not tested.
  - ParticleMesh's carrier phase, moving by 0.5 to 0.6 mrad against both other codes in vacuum and in plasma, and its centroid, up to 0.0016 c/ω₀ from PIConGPU's with the focus ahead and 0.0022 with it on the plane; and ParticleMesh's amplitude 0.05 % to 0.07 % above both others at t = 200 to 400, falling to 0.01 % by the end.
  - The weight above 1 mₑc² over the shared region, ParticleMesh against Smilei: 1.5 % to 3.6 % at t = 700 to 1000 where it is within 0.5 % at the other outputs.
- **Not done.**
  - PIConGPU's spectrum over the common region (§4.4, #412), and with it the reader of its electron dump on a real file.
  - A second resolution (§6).
  - Float32, and the Fast mode, for ParticleMesh; A5's performance comparison.
  - The engine on `main` after `6ead5bd`: ParticleMesh's runs are that commit's.
- **The tests.** `scripts/lwfa/test/runtests.jl` (T-510 to T-518, T-690 to T-694, T-698 to T-708, T-710, T-711): the generators, the readers and the measures on synthetic data, Smilei's boundary update on a 1D grid, and a small ParticleMesh run end to end. `lib/ParticleMeshElectromagnetic/test/antenna.jl` (T-695 to T-697, T-709): the wavefront envelope, as a function and launched through the antenna. They do not test the physics of the case.

## 6. Proposed for A7

A7 bases the benchmark thresholds on the measured Smilei–PIConGPU spread for equivalent inputs and asks that ParticleMesh agree with each of them to within it. It has accepted thresholds for the thermal and cold cases and none for LWFA. What follows is measured on the runs of §4, the first whose inputs are equivalent as far as is known; nothing is accepted, and the owner sets A7.

### 6.1 The spread and ParticleMesh's distances

The counted observables over the outputs where they exist (vacuum: t = 200 to 1300, 12 outputs; plasma: t = 400 to 1300, 10 outputs; maps: t = 300, 700, 1200). Each entry is the range of the magnitude; "outside" counts the outputs where ParticleMesh's distance exceeds the Smilei–PIConGPU distance at the same output, with the largest excess; the last two columns are ParticleMesh's largest distance over the run divided by the largest Smilei–PIConGPU distance over the run.

| Observable | S–P spread | PM–S | PM–P | PM–S outside | PM–P outside | PM–S / S–P | PM–P / S–P |
|---|---|---|---|---|---|---|---|
| vacuum: carrier phase (mrad) | 0.03 to 0.55 | 0.01 to 0.36 | 0.23 to 0.70 | 7 (0.24 at t = 1200) | 10 (0.36 at t = 800) | 0.65 | 1.26 |
| vacuum: centroid (10⁻³ c/ω₀) | 0.22 to 1.98 | 0.07 to 0.74 | 0.32 to 1.54 | 1 (0.48 at t = 500) | 2 (0.40 at t = 400) | 0.37 | 0.78 |
| vacuum: amplitude, relative rms (%) | 0.008 to 0.034 | 0.009 to 0.064 | 0.005 to 0.066 | 6 (0.055 at t = 400) | 4 (0.056 at t = 300) | 1.88 | 1.94 |
| focus on the plane: carrier phase (mrad) | 0.02 to 0.49 | 0.20 to 0.36 | 0.15 to 0.69 | 10 (0.25 at t = 1200) | 12 (0.36 at t = 500) | 0.73 | 1.41 |
| focus on the plane: centroid (10⁻³ c/ω₀) | 0.36 to 1.73 | 0.21 to 0.65 | 0.15 to 2.17 | 1 (0.15 at t = 300) | 2 (0.65 at t = 1300) | 0.38 | 1.26 |
| focus on the plane: amplitude, relative rms (%) | 0.017 to 0.034 | 0.005 to 0.060 | 0.005 to 0.036 | 3 (0.036 at t = 200) | 2 (0.012 at t = 200) | 1.74 | 1.05 |
| plasma: carrier phase (mrad) | 0.40 to 0.56 | 0.01 to 0.61 | 0.01 to 0.55 | 2 (0.07 at t = 1300) | 0 | 1.09 | 0.98 |
| plasma: centroid (10⁻³ c/ω₀) | 0.09 to 2.07 | 0.01 to 0.88 | 0.08 to 1.57 | 1 (0.30 at t = 900) | 3 (0.37 at t = 400) | 0.43 | 0.76 |
| plasma: amplitude, relative rms (%) | 0.008 to 0.046 | 0.017 to 0.064 | 0.006 to 0.056 | 10 (0.055 at t = 400) | 2 (0.048 at t = 400) | 1.38 | 1.22 |
| wake E_x over 1.5 λ_p, relative rms (%) | 0.04 to 0.21 | 0.08 to 0.23 | 0.02 to 0.12 | 10 (0.101 at t = 500) | 2 (0.044 at t = 500) | 1.10 | 0.58 |
| wake minimum, \|a/b − 1\| (%) | 0.00 to 0.10 | 0.00 to 0.11 | 0.00 to 0.10 | 8 (0.099 at t = 400) | 3 (0.099 at t = 400) | 1.16 | 1.02 |
| wake maximum, \|a/b − 1\| (%) | 0.00 to 0.13 | 0.01 to 0.16 | 0.00 to 0.04 | 9 (0.043 at t = 600) | 4 (0.020 at t = 400) | 1.24 | 0.32 |
| maps of E_y, all rows, relative rms (%) | 0.05 to 0.10 | 0.06 to 0.10 | 0.06 to 0.07 | 3 of 3 (0.006 at t = 700) | 1 (0.016 at t = 300) | 1.01 | 0.68 |
| maps of E_y, flank, relative rms (%) | 0.09 to 0.14 | 0.10 to 0.14 | 0.07 to 0.13 | 1 (0.008 at t = 300) | 1 (0.030 at t = 300) | 0.97 | 0.90 |
| bunch weight, \|a/b − 1\| (%), t = 1000 to 1300 | 1.26 | 1.39 | 0.14 | 4 of 4 (0.12) | 0 | 1.10 | 0.11 |
| highest occupied bin, \|a − b\| (mₑc²), t = 800 to 1300 | 0 | 0 to 0.2 | 0 to 0.2 | 1 (0.2 at t = 1100) | 1 (0.2 at t = 1100) | no ratio | no ratio |

How these read:

- **With the inputs equivalent, the three distances are of one size.** The Smilei–PIConGPU spread is no larger than ParticleMesh's distance to either code: in the amplitude it is the smallest of the three, in the wake's E_x it lies between them, in the plasma centroid it is the largest. In the second runs it looked as if ParticleMesh were close to PIConGPU and Smilei the code apart; that was Smilei's amplitude.
- **So "within the spread, output by output" is decided by hundredths of a percent and by signs.** ParticleMesh is outside at some output in every row, against Smilei at all ten outputs in the wake's E_x and in the plasma amplitude (by at most 0.10 % and 0.055 %), and in the maximum energy because the two reference codes happen to fill the same bin at t = 1100 and ParticleMesh the next one.
- **The laser's phase and centroid are signed and cross zero**, so the distance of two codes at a single output can be arbitrarily small (0.02 mrad) while every pair stays below 0.7 mrad and 0.0022 c/ω₀. An output-by-output comparison of such a quantity with the spread does not measure agreement.

### 6.2 What a rule should be able to reject

Two sets of runs with a known mismatch of the inputs exist to try a rule on, besides the matched ones:

- the first runs (§3.1): a carrier 1.98 rad off, a centroid 0.26 to 0.41 c/ω₀ off;
- Smilei uncorrected (§3.2): 0.198 % more amplitude than the other two and nothing else. Its largest distances to PIConGPU over the run (`lwfa-a5-2-compare/comparison.json`): amplitude 0.215 % in vacuum and 0.218 % in plasma, plasma phase 3.37 mrad, plasma centroid 0.0046 c/ω₀, wake E_x 0.568 %, wake minimum 0.335 % and maximum 0.435 %, maps over all rows 0.356 %, bunch 1.68 %, highest bin 0.2 mₑc².

A mismatch of 0.2 % in a₀ is a small one; whether a rule for this case should reject it is the owner's choice, and the options below differ in exactly that.

### 6.3 Three options

Each is a rule for the counted observables of §6.1. In all three the signed quantities (phase, centroid) are held on the run's largest magnitude, not output by output, and the spread is the matched runs', stored with the rule, not the spread of whichever runs are being judged: against the first runs' own Smilei–PIConGPU centroid spread of 0.264 c/ω₀, ParticleMesh's first-run centroid distance of 0.405 c/ω₀ is 1.5 of it.

**Option 1: A7's sentence as it stands, output by output.** |ParticleMesh − each code| ≤ |Smilei − PIConGPU| at the same output, for every counted observable.

- On the matched runs: ParticleMesh fails every row of §6.1, by the "outside" entries.
- It would also fail either reference code put in ParticleMesh's place in some rows: the three distances are of one size, and whichever is largest at an output is outside.
- Not proposed; listed because it is what A7 says today.

**Option 2: a stated multiple of the stored spread's largest value over the run, per observable**, the form A7 accepted for the cold two-stream. ParticleMesh's largest distance over the largest spread is in the last two columns of §6.1; the largest of all is 1.94 (vacuum amplitude against PIConGPU), then 1.88, 1.74, 1.41 and 1.38.

| multiple | ParticleMesh on the matched runs | Smilei uncorrected, against PIConGPU | first runs |
|---|---|---|---|
| 1 | fails 12 of the 15 rows with a ratio (largest 1.94) | rejected in every row but the vacuum phase and centroid | rejected |
| 2 | passes every row with a ratio; the closest is 1.94 | rejected in the amplitude (6.3, 5.3 and 4.7 times the spread in the two vacuum sets and in plasma), the plasma phase (6.0), the wake's E_x (2.7), minimum (3.4) and maximum (3.3), the plasma centroid (2.2) and the maps (3.5 over all rows, 2.1 in the flank) | rejected (centroid 200 times) |
| 3 | passes, the closest at 0.65 of the bound | rejected in the amplitude, the plasma phase, the wake's extrema and the maps over all rows; passes the wake's E_x, the plasma centroid and the maps' flank | rejected |
| 5 | passes, the closest at 0.39 of the bound | rejected in the vacuum amplitude (6.3, and 5.3 with the focus on the plane) and the plasma phase (6.0) only | rejected |

The maximum energy has no multiple: the spread is zero, so this form needs a floor of one bin there. The bunch's weight passes at a multiple of 2 or more both for ParticleMesh (1.10) and for Smilei uncorrected (1.3), so it does not tell the matched runs from the mismatched ones.

**Option 3: fixed thresholds per observable, set once from the matched runs.** Round numbers at about one and a half to two times the largest distance of any of the three pairs, in either vacuum set or in plasma:

| Observable | threshold | largest distance of any pair on the matched runs | Smilei uncorrected, against PIConGPU |
|---|---|---|---|
| carrier phase, run's largest magnitude | 1 mrad | 0.70 (PM–P, vacuum) | 3.37 in plasma: rejected (0.55 in vacuum: passes, as it should) |
| centroid, run's largest magnitude | 0.004 c/ω₀ | 0.0022 (PM–P, focus on the plane) | 0.0046 in plasma: rejected |
| amplitude on the axis, relative rms | 0.1 % | 0.066 (PM–P, vacuum) | 0.215: rejected |
| wake E_x over 1.5 λ_p, relative rms | 0.4 % | 0.232 (PM–S) | 0.568: rejected |
| wake extrema, \|a/b − 1\| | 0.25 % | 0.162 (PM–S, maximum at t = 700) | 0.335 and 0.435: rejected |
| maps of E_y over all rows, relative rms, back 48 cells left out | 0.2 % | 0.104 (PM–S) | 0.356: rejected |
| highest occupied bin | one bin, 0.2 mₑc² | 0.2 (PM against both, t = 1100) | 0.2: passes |
| bunch weight | not counted (or 3 %) | 1.39 from t = 1000; 2.77 at t = 800 and 900 | 1.68 |

ParticleMesh passes all of them against both codes, with its distance at 0.52 to 0.70 of the threshold in the rows where it is the largest; the first runs fail the first two by factors of 60 and more; Smilei uncorrected fails every laser and wake row. They read more plainly than a multiple, do not move when a reference run is repeated, need no floor, and can be asked of ParticleMesh against one reference code where the other is not available. They are tied to this case, this resolution and these windows, and the factor of one and a half to two over the measured distances is a choice with nothing behind it but the wish to reject a 0.2 % mismatch of a₀.

Common to options 2 and 3, as part of the rule's statement:

- the wake's measure on a window that ends inside a bucket, with its length (§4.3);
- a measure off the axis among the counted ones, with the cells at the back of the windows left out (§4.3);
- the bunch's weight not counted, or counted loosely: on this case it is 7 macro-particle weights carried by 35 or more particles from the ramp's foot, the gap rule that defines it gives Smilei a different group at two outputs, and the 1.3 % that separates Smilei from the other two is not understood (§4.4).

### 6.4 What the spread means, and what a second resolution would add

Each code ran once at one resolution, and the runs are deterministic (cold regular loads, no seeds), so no difference here is noise and none can be averaged away: every number of §6.1 is a systematic difference of three implementations of one scheme at λ₀/48, with launches matched as far as the sources were read. Three things limit what it means.

- **It is one point.** A spread from one run per code has no error bar. The history of this report is the caution: the Smilei–PIConGPU "spread" in the wake's E_x was 6 % in the first runs (over two plasma wavelengths), 0.57 % in the second and is 0.21 % now (over one and a half), each time because a mismatch of the inputs was found and removed. Nothing shows that the present one is the last; the unexplained items of §5 are where another could be.
- **It does not say how close any code is to the equations.** The three share the scheme, so they share its errors: at λ₀/48 the observables are converged to some percent that these runs do not measure, and a threshold of a few tenths of a percent is a statement about three codes sharing an error, not about any of them being right.
- **Its dependence on the grid is unknown, except for one part.** Smilei's launch factor is discretization of a known size, (Δx² + Δt²)/16 above 1: 0.198 % at λ₀/48, 0.447 % at λ₀/32. For the rest, one resolution cannot say whether a difference shrinks as the grid is refined or stays (a load rule would stay).

A second resolution in all three codes, λ₀/32 (the grid of `lwfa-2d-a2.toml`) or λ₀/64, would give the spread's dependence on Δ, and each code's own change with resolution, which is the scale a threshold has to be compared with. It would also show whether thresholds set at one resolution (option 3) hold at another, or whether the rule has to be the relative one (option 2).
