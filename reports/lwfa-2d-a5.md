# LWFA 2D: A5's first case in ParticleMesh, Smilei and PIConGPU

| | |
|---|---|
| Work package | WP-053 (`trace/wp/WP-053-lwfa-three-codes.md`); goals A5, A7, and A3 for #362. Follows PR #336, which brought the PIConGPU column |
| Case | `scripts/lwfa/cases/lwfa-2d.toml`, λ₀/48, with `laser.envelope = "wavefront"` (sha256 `e7bc9e3f…`, in the run manifests), and `scripts/lwfa/cases/lwfa-2d-focus0.toml`, the same with the focus on the injection plane, in vacuum only |
| Commits | every run of §4 used the code of `6ead5bd` (the box requests `5b729a7` and `9f8a388` are that commit plus the prepared run directories). The comparisons: the commit in each `manifest.toml` under `lwfa-2d-a5/` (`dirty = false`) |
| Runs | ParticleMesh and Smilei: CPU box requests `runs/lwfa-a5-2` (result `ab315a6`) and `runs/lwfa-a5-focus0` (result `e86a2b8`); their summaries stay on those branches. PIConGPU: gpu-run jobs `a5-2-pic-vacuum`, `a5-2-pic-plasma` and the focus-0 vacuum run; ParticleMesh on CUDA: `lwfa-a5-2-cuda-pm-plasma`. The summaries of these four are on no branch and are in `lwfa-2d-a5/runs/`. First runs (§3): `runs/lwfa-a5-1` and `lwfa-a5pic` |
| Input hashes | each run's `manifest.toml` as `manifest-<run>.toml` next to its comparison; the box's `box-status.toml` there; the sha256 of the summaries compared under `summaries` in each comparison's `manifest.toml` |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode; 16 threads on an AMD EPYC 9655 (the CPU box), and CUDA on an RTX 4080 SUPER. Smilei v5.1 (`438d43d`, image `particlemesh-smilei:e4f0148b8952fc87`), 16 OpenMP threads, the same box. PIConGPU 0.8.0 (`6bb3470e`), CUDA, Float64, the RTX 4080 SUPER |
| Figures | `lwfa-2d-a5/lwfa-a5-2-compare/` (A5's case), `lwfa-a5-focus0-compare/` (focus on the plane), each PNG with its JSON sidecar; `lwfa-a5-2-cuda-pair/pair.json` (the two backends); `lwfa-a5-three-compare/comparison.json` (first runs; their figures are in this file's history, `c9de0de`). The measured values below are keys of the `comparison.json` or `pair.json` named with them; ranges, largest values, counts and ratios are computed from those keys, the wall times and the box's step times are from the runs' summaries and `box-status.toml`, and the few numbers computed from the summaries or the maps directly say so where they stand |
| Status | measured; reviewed twice, the present version not yet. No number here is to leave the repository before the review of this version |
| Review | Two independent reviews by fresh numerics-reviewer subagents (claude-fable-5-2, 2026-11-12). The first, on the unmatched first runs, found that the three codes had launched three different pulses and that the spectra covered different regions (§3, §4.4). The second, on `c9de0de`, reproduced every table of the matched runs and found two wrong numbers, a hypothesis stated as fact, the wake's measure ending on a bucket's edge, and the pulse compared on the axis only; this version answers those. A third pass will follow |

`pm-plasma`'s manifest says `dirty = true`: the step before it on the box, `pm-vacuum` (same commit, `dirty = false`), rewrites its own tracked `manifest.toml`. This is inferred from the two manifests, not recorded by the run.

## 1. Setup

The case, in A1's units: a₀ = 2, waist 28, τ = 26 (1/e half width of the field), focus at x = 320; electrons of 0.01 n_c (λ_p = 62.8) behind 260 of vacuum and a 60 sin² ramp, 2 × 2 per cell, cold, ions immobile; Δx = λ₀/48, Δy = 4Δx, Δt at 0.95 of Yee's 2D limit; a window of 2304 × 384 cells moving at c from t = 256; 10 777 steps, 14 outputs 829 steps apart (t ≈ 0, 100, …, 1300). Yee, Esirkepov, quadratic shapes, Boris, no current filter, Float64 in all three.

One pulse on the lab plane x = 0 in all three codes: a cosine carrier counted from the envelope's peak, the 2D Gouy phase, and envelope and carrier on one time along the curved wavefront. How each code is made to launch it, with the source lines read, is in the headers of `scripts/lwfa/lwfa.jl` and `scripts/lwfa/picongpu.jl`: PIConGPU's `GaussianPulse` is that pulse; ParticleMesh launches it with `GaussianLaser(envelope = WavefrontTime())`; Smilei's namelist gets a focus counted from its ghost node, an envelope centre and a phase offset (`smilei_laser`). The three share Δx, Δy and Δt, so their grids are one lattice in the lab frame and every comparison matches nodes by index, with nothing interpolated.

## 2. Measures

`compare_rows` in `scripts/lwfa/lwfa.jl`; a pair is written a–b (PM–S ParticleMesh–Smilei, PM–P ParticleMesh–PIConGPU, S–P Smilei–PIConGPU), and every relative rms difference is relative to b's rms.

- **Carrier phase, amplitude** (`carrier_phase`, `amplitude_relrms`, `amplitude_max`): from the lattice's own analytic signal on the axis, z(m) = E_y(m) + i·(9·(E_y(m − q) − E_y(m + q)) + E_y(m − 3q) − E_y(m + 3q))/16 with q a quarter wavelength (12 points). What is left of the carrier in |z| is of order (qΔx/τ)⁴ = 10⁻⁵ for this pulse (T-691, T-692).
- **Centroid** of E_y² on the axis (`centroid_x`).
- **Wake**: E_x on the axis from ParticleMesh's envelope peak back one and a half plasma wavelengths (720 cells), the same lattice points for all three pairs (`wake_Ex_relrms`, `wake_Ex_min`, `wake_Ex_max`); `wake_Ex_shift` is the whole-cell displacement of b's wake that fits a's best. An output has these only when |E_x| reaches 10⁻³ of the wave-breaking field in the window in both codes: from t = 400 here. The window ends inside the second bucket. The version reviewed used two plasma wavelengths (960 cells), which from t = 500 ends on the jump of E_x at the back of the second bucket, 913 to 958 cells behind the peak: the window's anchor moves in steps of half a laser wavelength (24 cells), and a few cells at the jump then carry most of every pair's squared difference (§4.3). `wake_sensitivity` repeats the measures for windows of 1.0, 2.0 and 2.5 λ_p, and `wake_anchor_sensitivity` for the window moved by 24 cells either way.
- **Off the axis** (`maps`, `map_rows`): the relative rms difference of the 2D maps of E_y, kept on every sixth column and every row at the three outputs t = 300, 700, 1200 of the plasma runs, on the lattice points two maps share: over all rows, over the beam's flank (0.9 to 1.6 waists from the axis, 25.2 ≤ |y| < 44.8) and over its far wing (2.15 to 2.85 waists, 60.2 ≤ |y| < 79.8). The two envelope conventions of §3 are one field on the axis of the launch plane and differ off it, so this is where a remaining mismatch of that kind would show. The 48 cells (a laser wavelength) at the back of the shared columns are left out, for the reason given in §4.3; the keys ending in `_with_back` keep them.
- **Spectra** in one unit, electrons per unit depth in n_c (c/ω₀)²; `bunch` is the highest energy group of a spectrum, the occupied bins above its widest empty gap (`bunch_group`); `top` the upper edge of the highest occupied bin.

In `comparison.json` for reference, not counted: the box maximum of |E_y| (sampled on the lattice, up to 1 − cos(π/48) = 0.21 % below a crest); the running-max envelope (`envelope_relrms`, which reads a carrier phase as a difference); ρ_e on the axis (`wake_rho_relrms`: a few narrow spikes, whose relative rms moves by tens of percent with a displacement of a cell or with the code that places the window).

## 3. The first runs: three different pulses

The runs of 5 November (`lwfa-2d-a5/lwfa-a5-three-compare/comparison.json`) compared three pulses, and this is the record of what the mismatch was. The first version of this report, with their tables, is in this file's history (`1080c9a`); the first runs' figures and PIConGPU's first-run summaries left `lwfa-2d-a5/` with this version and are in the history at `c9de0de`, and their `comparison.json` stays, with its manifests.

| on the axis in vacuum, t = 200 to 1300 | PM–S | PM–P | S–P |
|---|---|---|---|
| carrier phase a − b (rad) | +1.9785 to +1.9790 | −0.0011 to +0.0004 | −1.9795 to −1.9785 |
| centroid a − b (c/ω₀) | +0.249 falling to −0.145 | −0.014 falling to −0.405 | −0.264 to −0.260 |

- **Smilei.** `LaserGaussian2D` gives the boundary envelope(t − φ(y))·sin(t − φ(y)): a sine of the run's time and no Gouy phase, where the others have a cosine from the envelope's peak; and the Silver–Müller condition applies it on the outermost ghost node, two cells behind xmin. For the case's numbers that is a phase of delay + ½·atan(320/392) − π/2 − 2Δx = 1.9787 rad mod 2π and a lag of 2Δx = 0.2618 c/ω₀: the two constant offsets measured.
- **ParticleMesh.** Its default envelope is in t − delay alone, the same at every y (`AxisTime`); PIConGPU's and Smilei's follow the curved wavefront. Past the plane the `AxisTime` pulse arrives on the axis later by (w²/(4|R|))/(1 + κ²w⁴), κ = (1/L − 1/|R|)/2, with w = 36.1 and |R| = 800.2 on the plane and L the distance travelled (T-696): up to 0.408 c/ω₀. That closed form, with no free parameter, is the measured motion of ParticleMesh's centroid against PIConGPU's to 0.0044 c/ω₀ at each of the twelve outputs. Neither convention is a defect; `AxisTime` stays ParticleMesh's default and `WavefrontTime` is an opt-in (T-695 to T-697).
- **So the first runs' Smilei–PIConGPU distance was Smilei's two launch offsets**, not a spread for equivalent inputs: 5 % in the running-max envelope, and in the wake's E_x (two plasma wavelengths behind the peak) 4.2 % to 6.2 % at the nine outputs t = 500 to 1300, which a shift of two cells took down to 1.1 % to 1.7 %. At t = 400, where the wake is just forming, it was 2.17 %, the best shift one cell, and 1.87 % was left.
- **#362.** ParticleMesh's `ekin_max` was 1.17 to 2.41 mₑc² before the laser reached the plasma: the window's loader fills the vacuum with particles of zero weight (the density enters through the weights, A3), and the laser makes them quiver. `pm_spectrum` now takes the maximum over particles of positive weight (T-690).

## 4. The matched runs

### 4.1 Do the three codes now launch one pulse? The laser in vacuum

![the distances of the three pairs against time, matched runs](lwfa-2d-a5/lwfa-a5-2-compare/pair_distances.png)

A5's case, focus 320 ahead of the plane (`vacuum`, `picongpu_vacuum`, `smilei_picongpu_vacuum` in `lwfa-a5-2-compare/comparison.json`). Carrier phase a − b in mrad and centroid a − b in 10⁻³ c/ω₀:

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
| 200 | 0.149 | 0.066 | 0.215 | −0.148 | +0.065 | +0.214 |
| 300 | 0.139 | 0.065 | 0.205 | −0.139 | +0.065 | +0.204 |
| 400 | 0.135 | 0.056 | 0.191 | −0.135 | +0.055 | +0.190 |
| 500 | 0.137 | 0.040 | 0.177 | −0.138 | +0.039 | +0.177 |
| 600 | 0.146 | 0.022 | 0.169 | −0.147 | +0.021 | +0.169 |
| 700 | 0.157 | 0.008 | 0.165 | −0.158 | +0.007 | +0.166 |
| 800 | 0.167 | 0.005 | 0.165 | −0.168 | −0.002 | +0.166 |
| 900 | 0.174 | 0.009 | 0.167 | −0.175 | −0.008 | +0.168 |
| 1000 | 0.180 | 0.012 | 0.169 | −0.181 | −0.011 | +0.170 |
| 1100 | 0.184 | 0.013 | 0.172 | −0.185 | −0.013 | +0.173 |
| 1200 | 0.187 | 0.014 | 0.174 | −0.188 | −0.013 | +0.175 |
| 1300 | 0.189 | 0.014 | 0.176 | −0.190 | −0.013 | +0.177 |

- **Smilei's corrected launch behaves as its source said.** Its carrier is within 0.55 mrad of PIConGPU's and 0.36 mrad of ParticleMesh's (it was 1.979 rad), and its centroid within 0.002 c/ω₀ of both (it was 0.26 behind). This is the first time that namelist met Smilei.
- **`WavefrontTime` in a real run.** ParticleMesh's centroid stays within 0.0016 c/ω₀ of PIConGPU's through the focus and after (it fell 0.405 behind with `AxisTime`), and its carrier within 0.70 mrad.
- **What is left.** The three carriers agree to 0.7 mrad, the centroids to 0.002 c/ω₀ (a sixty-fifth of a cell), ParticleMesh's and PIConGPU's amplitudes to 0.07 %. Smilei's amplitude is 0.14 % to 0.21 % above both, at every output and with the pulse's shape (the rms difference is the peak's): a factor on its field, not a displacement. §4.2 says where the factor comes from.

### 4.2 The focus on the injection plane

`lwfa-2d-focus0.toml`: the wavefront is flat where the pulse is launched, so nothing of the launch depends on its curvature (`lwfa-a5-focus0-compare/comparison.json`). Phase in mrad, centroid in 10⁻³ c/ω₀, amplitude rms in percent:

| t | phase: PM–S | PM–P | S–P | centroid: PM–S | PM–P | S–P | amplitude: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|
| 200 | −0.20 | −0.69 | −0.49 | −0.21 | −0.76 | −0.56 | 0.138 | 0.036 | 0.174 |
| 300 | −0.30 | −0.69 | −0.39 | −0.51 | −0.15 | +0.36 | 0.148 | 0.018 | 0.167 |
| 400 | −0.35 | −0.64 | −0.29 | −0.56 | +0.40 | +0.95 | 0.159 | 0.006 | 0.164 |
| 500 | −0.36 | −0.56 | −0.20 | −0.49 | +0.75 | +1.24 | 0.168 | 0.005 | 0.165 |
| 600 | −0.35 | −0.49 | −0.14 | −0.41 | +0.92 | +1.33 | 0.175 | 0.009 | 0.167 |
| 700 | −0.33 | −0.43 | −0.10 | −0.35 | +0.98 | +1.32 | 0.180 | 0.012 | 0.170 |
| 800 | −0.31 | −0.38 | −0.07 | −0.30 | +0.98 | +1.28 | 0.184 | 0.013 | 0.172 |
| 900 | −0.28 | −0.33 | −0.05 | −0.25 | +0.96 | +1.20 | 0.187 | 0.014 | 0.175 |
| 1000 | −0.26 | −0.30 | −0.03 | −0.24 | +0.80 | +1.04 | 0.189 | 0.013 | 0.177 |
| 1100 | −0.26 | −0.28 | −0.02 | −0.52 | +0.62 | +1.15 | 0.190 | 0.014 | 0.177 |
| 1200 | −0.27 | −0.25 | +0.02 | −0.48 | +1.24 | +1.73 | 0.194 | 0.018 | 0.178 |
| 1300 | −0.22 | −0.15 | +0.07 | +0.65 | +2.17 | +1.52 | 0.201 | 0.020 | 0.183 |

The same sizes as with the focus 320 ahead: phases within 0.7 mrad, centroids within 0.0022 c/ω₀, Smilei's amplitude 0.14 % to 0.20 % above the others. So what the three launches still differ by does not come from the wavefront's curvature, and the larger part of it, Smilei's amplitude, is in the launch itself and not in the propagation: it is there at t = 200 and changes by 0.06 over the run.

**The factor is Smilei's boundary condition, a third mismatch of the launch.** Its Silver–Müller update at xmin (`ElectroMagnBC2D_SM.cpp:63` to `66` and `245`) is, at normal incidence, (B₀ + B₁)/2 + (E₀ half a step before + E₀ half a step after)/2 = 2b: the mean of the new B_z over the two points either side of the boundary node plus the mean of that node's E_y over the step around the new B's time, set to twice the profile b the namelist prescribes at that time. For the grid's own right-going plane wave those two means are cos(kΔx/2) and cos(ω₀Δt/2) of the wave, k the Yee wavenumber, so the wave launched is

    2/(cos(kΔx/2) + cos(ω₀Δt/2))

of the amplitude prescribed: 1.001984 on this grid, 1 + (Δx² + Δt²)/16 to second order, and 1.004474 at λ₀/32. Smilei's update run on a 1D Yee grid launches that factor to 10⁻¹⁰, for both case files and three other pairs of Δx and Δt (T-710); it is not 1/cos(Δt/2), the guess of the version reviewed, which is 1.001822. Measured, Smilei's peak amplitude is above PIConGPU's by 0.166 % to 0.214 % with the focus ahead and 0.164 % to 0.186 % with it on the plane (above ParticleMesh's by 0.135 % to 0.190 % and 0.139 % to 0.202 %), against 0.198 % for the plane wave. Not computed for the beam, an estimate only: the condition is the normal-incidence one and launches a component at angle α with E_y lower by about α²/4, which is 0.03 % for ⟨α²⟩ = 1/waist². `smilei_laser` now gives Smilei the case's a₀ divided by the factor (T-510). No run with that correction is in this report (§6).

### 4.3 The laser and the wake in plasma

![E_x and ρ_e on the axis at t ≈ 500, 800, 1200, three codes](lwfa-2d-a5/lwfa-a5-2-compare/wake_axis.png)

The laser (`plasma`, `picongpu_plasma`, `smilei_picongpu_plasma`). Phase in mrad, centroid in 10⁻³ c/ω₀, amplitude rms in percent:

| t | phase: PM–S | PM–P | S–P | centroid: PM–S | PM–P | S–P | amplitude: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|
| 400 | −0.04 | −0.55 | −0.51 | −0.40 | −1.26 | −0.85 | 0.134 | 0.056 | 0.191 |
| 500 | −0.40 | −0.53 | −0.13 | −1.41 | +0.08 | +1.49 | 0.133 | 0.040 | 0.174 |
| 600 | −0.88 | −0.46 | +0.42 | −3.94 | +0.65 | +4.60 | 0.140 | 0.027 | 0.166 |
| 700 | −1.29 | −0.37 | +0.93 | −3.18 | +1.04 | +4.22 | 0.162 | 0.017 | 0.174 |
| 800 | −1.63 | −0.25 | +1.38 | −2.07 | +0.86 | +2.93 | 0.183 | 0.014 | 0.187 |
| 900 | −1.93 | −0.15 | +1.78 | −1.82 | +0.30 | +2.12 | 0.196 | 0.013 | 0.198 |
| 1000 | −2.25 | −0.07 | +2.17 | −2.10 | −0.35 | +1.75 | 0.202 | 0.009 | 0.203 |
| 1100 | −2.57 | −0.01 | +2.56 | −2.60 | −0.94 | +1.66 | 0.206 | 0.006 | 0.209 |
| 1200 | −2.93 | +0.03 | +2.96 | −3.43 | −1.45 | +1.98 | 0.208 | 0.008 | 0.213 |
| 1300 | −3.30 | +0.07 | +3.37 | −3.46 | −1.57 | +1.89 | 0.212 | 0.010 | 0.218 |

The wake's E_x on the axis: relative rms difference in percent, over one and a half plasma wavelengths behind the peak (`wake_Ex_relrms`, the measure) and over two (`wake_sensitivity`, the version reviewed). The best whole-cell shift is 0 for every pair at every output and both windows (`wake_Ex_shift`), so the difference after the shift is the same number.

| t | 1.5 λ_p: PM–S | PM–P | S–P | 2 λ_p: PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|
| 400 | 0.170 | 0.062 | 0.192 | 0.170 | 0.062 | 0.192 |
| 500 | 0.351 | 0.122 | 0.471 | 0.363 | 0.135 | 0.490 |
| 600 | 0.401 | 0.108 | 0.504 | 0.552 | 0.162 | 0.710 |
| 700 | 0.465 | 0.066 | 0.525 | 0.665 | 0.137 | 0.769 |
| 800 | 0.527 | 0.044 | 0.552 | 0.844 | 0.152 | 0.954 |
| 900 | 0.522 | 0.036 | 0.537 | 0.750 | 0.168 | 0.817 |
| 1000 | 0.541 | 0.023 | 0.550 | 0.894 | 0.165 | 0.847 |
| 1100 | 0.556 | 0.021 | 0.568 | 0.785 | 0.029 | 0.805 |
| 1200 | 0.547 | 0.025 | 0.567 | 0.794 | 0.032 | 0.820 |
| 1300 | 0.534 | 0.030 | 0.559 | 0.770 | 0.039 | 0.804 |

Why the two-wavelength number is the secondary one. From t = 600 the jump of E_x at the back of the second bucket lies 914 to 954 cells behind ParticleMesh's envelope peak, in the last 46 cells of that 960-cell window, and within 20 cells of it sit 50 % to 65 % of the squared difference of each pair with Smilei and 34 % to 97 % of ParticleMesh–PIConGPU's (60 % to 97 % at t = 600 to 1000, 34 % to 43 % after); of the 720-cell window's, none. (Computed from the summaries with `compare`'s own functions, the jump taken as the largest change of E_x over 8 cells between 1.5 and 2.2 λ_p behind the peak; not keys of `comparison.json`.) The two-wavelength number is therefore mostly how each code places one steep front and where the window's end, which moves in 24-cell steps with the anchor, cuts it.

The largest value over the ten outputs for other windows (`wake_sensitivity`, `wake_anchor_sensitivity`):

| window | PM–S | PM–P | S–P | outputs where PM–S exceeds S–P | where PM–P does |
|---|---|---|---|---|---|
| 1.0 λ_p | 0.556 | 0.131 | 0.567 | none | none |
| 1.5 λ_p, the measure | 0.556 | 0.122 | 0.568 | none | none |
| 1.5 λ_p, moved 24 cells back | 0.569 | 0.124 | 0.580 | none | none |
| 1.5 λ_p, moved 24 cells forward | 0.543 | 0.121 | 0.561 | none | none |
| 2.0 λ_p | 0.894 | 0.168 | 0.954 | t = 1000 (0.894 against 0.847) | none |
| 2.5 λ_p | 1.183 | 0.197 | 1.153 | t = 1000 (1.183 against 1.153) | none |

The 2.5 λ_p window is longer than what the three windows share behind the peak (at t = 700 the maps' shared columns begin 1093 cells behind it), so it ends on the first cells of Smilei's window (see the maps below).

The wake's extrema in the 1.5 λ_p window, m_ecω₀/e (`wake_Ex_min`, `wake_Ex_max`), and a/b − 1 in percent:

| t | min: PM | Smilei | PIConGPU | PM–S | PM–P | S–P | max: PM | Smilei | PIConGPU | PM–S | PM–P | S–P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | −0.00481 | −0.00481 | −0.00480 | −0.033 | +0.100 | +0.133 | 0.02934 | 0.02939 | 0.02933 | −0.181 | +0.038 | +0.219 |
| 500 | −0.08907 | −0.08921 | −0.08903 | −0.158 | +0.051 | +0.209 | 0.07853 | 0.07866 | 0.07851 | −0.169 | +0.018 | +0.187 |
| 600 | −0.09362 | −0.09380 | −0.09359 | −0.191 | +0.032 | +0.224 | 0.10354 | 0.10375 | 0.10350 | −0.194 | +0.042 | +0.237 |
| 700 | −0.09405 | −0.09425 | −0.09403 | −0.209 | +0.021 | +0.231 | 0.10987 | 0.11011 | 0.10984 | −0.221 | +0.031 | +0.252 |
| 800 | −0.09262 | −0.09283 | −0.09262 | −0.225 | +0.001 | +0.226 | 0.10960 | 0.11010 | 0.10963 | −0.454 | −0.021 | +0.435 |
| 900 | −0.08980 | −0.09001 | −0.08979 | −0.232 | +0.015 | +0.248 | 0.10456 | 0.10477 | 0.10455 | −0.203 | +0.009 | +0.213 |
| 1000 | −0.08621 | −0.08637 | −0.08620 | −0.180 | +0.006 | +0.186 | 0.09982 | 0.10001 | 0.09982 | −0.189 | +0.003 | +0.192 |
| 1100 | −0.08299 | −0.08324 | −0.08297 | −0.299 | +0.019 | +0.318 | 0.09428 | 0.09456 | 0.09429 | −0.303 | −0.011 | +0.293 |
| 1200 | −0.07980 | −0.08007 | −0.07980 | −0.335 | −0.001 | +0.335 | 0.08933 | 0.08966 | 0.08933 | −0.364 | +0.008 | +0.374 |
| 1300 | −0.07699 | −0.07720 | −0.07698 | −0.277 | +0.011 | +0.289 | 0.08476 | 0.08502 | 0.08476 | −0.301 | +0.001 | +0.304 |

The wake's difference against the laser's, per output: `wake_Ex_relrms` over `amplitude_relrms` of the same pair.

| t | PM–S | PM–P | S–P |
|---|---|---|---|
| 400 | 1.27 | 1.11 | 1.01 |
| 500 | 2.63 | 3.03 | 2.70 |
| 600 | 2.86 | 4.03 | 3.03 |
| 700 | 2.88 | 3.94 | 3.01 |
| 800 | 2.88 | 3.02 | 2.95 |
| 900 | 2.67 | 2.72 | 2.72 |
| 1000 | 2.68 | 2.58 | 2.71 |
| 1100 | 2.70 | 3.69 | 2.72 |
| 1200 | 2.63 | 3.09 | 2.66 |
| 1300 | 2.52 | 3.12 | 2.56 |

- **The wake is the same in the three codes to 0.6 %.** On the two-wavelength window the first runs' pairs were 0.17 % to 6.2 % apart (ParticleMesh–PIConGPU at t = 400, Smilei–PIConGPU at t = 900), the matched runs' 0.03 % to 0.95 %; on the interior window the matched runs' are 0.02 % to 0.57 %, and no pair is displaced by a cell.
- **ParticleMesh and PIConGPU agree several times closer than either does with Smilei**: 0.02 % to 0.12 % in E_x and within 0.10 % in the extrema, against 0.17 % to 0.57 % and 0.03 % to 0.45 % for the pairs with Smilei. Smilei's wake is the larger one, by 0.19 % to 0.44 % in the extrema against PIConGPU's at every output from t = 500, as its laser is the stronger one, by 0.17 % to 0.22 %.
- **Every pair's wake difference is about three times its laser's amplitude difference** from t = 500: 2.5 to 3.0 for the two pairs with Smilei, 2.6 to 4.0 for ParticleMesh–PIConGPU, whose two differences fall together over the run (amplitude 0.040 % to 0.006 %, wake 0.122 % to 0.021 %). Read as a hypothesis, not shown here: the wake differences of these runs are the lasers' amplitude differences, Smilei's being its launch factor of §4.2. The run of Smilei with the corrected amplitude tests it (§6).
- **The laser in plasma.** ParticleMesh and PIConGPU: phase within 0.55 mrad, centroid within 0.0016 c/ω₀, amplitude within 0.06 %. Smilei's carrier moves away from both, steadily, to 3.3 and 3.4 mrad at the end, and its centroid by up to 0.0046 c/ω₀.

Off the axis: the 2D maps of E_y, relative rms difference in percent on the lattice points two maps share, without the 48 cells at the back of the shared columns (`maps`). t = 300 is before the pulse reaches the plasma.

| t | rows | PM–S | PM–P | S–P |
|---|---|---|---|---|
| 300 | all | 0.173 | 0.070 | 0.200 |
| 300 | flank, 0.9 to 1.6 waists | 0.270 | 0.121 | 0.172 |
| 300 | far wing, 2.15 to 2.85 waists | 0.723 | 0.685 | 0.459 |
| 700 | all | 0.219 | 0.062 | 0.239 |
| 700 | flank | 0.223 | 0.129 | 0.296 |
| 700 | far wing | 0.615 | 0.443 | 0.652 |
| 1200 | all | 0.349 | 0.056 | 0.356 |
| 1200 | flank | 0.290 | 0.074 | 0.293 |
| 1200 | far wing | 0.525 | 0.293 | 0.514 |

- **One pulse off the axis too, to these numbers.** Over the whole map ParticleMesh and PIConGPU are within 0.07 %, and each within 0.36 % of Smilei, of which a factor of 1.002 on Smilei's field alone would give 0.2 %. For scale, computed from `laser_field` and not measured in a run: on the launch plane the two envelope conventions of §3 differ by 1.4 % over all rows, 2.6 % in the flank and 10 % in the far wing.
- **The far wing is where all three pairs are furthest apart**, 0.29 % to 0.72 %, ParticleMesh–PIConGPU included (0.29 % to 0.69 %, against 0.06 % to 0.13 % on the rest). At the focus the field in that band is a hundredth of the axis's and less. Not explained.
- **The back of Smilei's window.** With the back 48 cells kept (`…_with_back`), the far wing at t = 700 is 4.18 % and 4.20 % for the pairs with Smilei and 0.44 % for ParticleMesh–PIConGPU; at t = 300, before there is a wake, the two versions are equal. The shared columns begin at the back of Smilei's window, and the difference is Smilei's first cells: in the two kept columns nearest that back at t = 700 (12 cells), and the three nearest at t = 1200 (18 cells), Smilei's E_y differs from ParticleMesh's by 21 % to 40 % of the column's field (the root of the squares summed over its rows), where PIConGPU's differs by 1 %. In the columns after, Smilei's differs by 1.4 % to 8 % and PIConGPU's by 0.3 % to 1.7 %; these are columns far behind the pulse, where the field is small. These column numbers are from the runs' `maps.h5`, not keys of `comparison.json`. The cause is not established; what stands there is the wake's transverse field at Smilei's xmin boundary, where its absorbing condition acts. The wake's window on the axis ends more than 300 cells ahead of these columns at both outputs.

![n_e/n₀ at t ≈ 1200, three codes](lwfa-2d-a5/lwfa-a5-2-compare/density_maps.png)

The three density maps at t ≈ 1200 (the maps now come back at the case's stride): the same two buckets, crests and laser-driven ripples at the same lab positions. The figure's sidecar holds the plotted values to four decimals (1.9 MB; 3.8 MB at full precision), which is finer than the colour scale shows.

### 4.4 Electrons

![the electron spectrum at the end, three codes, one unit](lwfa-2d-a5/lwfa-a5-2-compare/spectrum.png)

What each histogram covers (`spectrum` in `comparison.json`; `region` there says "each code's own"):

- **ParticleMesh and Smilei: one region**, the lattice cells both windows' plasma covers at the output (`spectrum_region`, e.g. cells 7992 to 10277 at the end). Smilei's two strip diagnostics, new in this run, worked on the real code (`spectrum_failed` is empty in its summary). The two totals agree to 3 × 10⁻⁴ or better at every output (1.5 × 10⁻⁵ at the end: 602.878 against 602.887).
- **PIConGPU: its whole simulated domain**, up to a local domain of 2328 cells behind the window's back. Its deck writes no electrons: with every species in the file output, the plasma run computed its steps and then did not exit, and left a file that could not be read (`scripts/lwfa/picongpu.jl`'s header; #412). So PIConGPU's totals and low energies do not compare with the others' (above 5 mₑc² it holds 0.63 at t = 1100, the others 0.0013). That the excess is electrons behind the window, which the other two codes have dropped, is an inference from those totals and from what the plugin covers; no dump was there to check it against. What is compared is the highest energy group and the highest occupied bin, on the assumption that they do not depend on the region. That too is not checked against a dump; what supports it is that PIConGPU's bunch weighs 1.248813 × 10⁻³ at all six outputs from t = 800, whatever its domain holds behind the window at each. The reader that would bin an electron dump over the region is tested on synthetic files only (T-704, T-705) and has not met PIConGPU.

The bunch, the highest energy group of each spectrum (`bunch`):

| | ParticleMesh | Smilei | PIConGPU |
|---|---|---|---|
| weight, every output from t = 800 (n_c (c/ω₀)²) | 1.247061 × 10⁻³ | 1.269786 × 10⁻³ | 1.248813 × 10⁻³ |
| energy range at t = 1300 (mₑc²) | 55.2 to 67.2 | 56.4 to 67.4 | 57.2 to 67.2 |
| occupied bins at t = 1300 (0.2 mₑc² each) | 36 | 35 | 36 |
| of those, bins with less than half a plateau macro-particle's weight | 32 | 30 | 32 |
| empty gap below the group at t = 1300 (mₑc²) | 44.0 | 40.6 | 34.6 |
| highest occupied bin ends at, t = 800 … 1300 | 24.2, 33.6, 42.4, 51.2, 59.4, 67.2 | 24.4, 33.6, 42.6, 51.2, 59.6, 67.4 | 24.2, 33.6, 42.4, 51.0, 59.4, 67.2 |

- **Weight.** ParticleMesh is 0.14 % below PIConGPU and 1.79 % below Smilei; Smilei is 1.68 % above PIConGPU. At t = 800 PIConGPU's group is separated from the rest by one mₑc² only (`gap`); from t = 900 by 4.6 or more.
- **What the bunch is.** One plateau macro-particle carries n₀·Δx·Δy/4 = 1.713 × 10⁻⁴ (`macro_weight`); the bunch weighs 7.3 to 7.4 of those but occupies 35 or 36 bins, 30 to 32 of them with less than half of one. It is at least about 35 macro-particles, most of them light: a particle's weight is the density where it was loaded, so they come from the foot of the ramp, where the three codes assign weights by three rules (ParticleMesh at the particle, PIConGPU at the cell centre, Smilei where its particle creator puts it, `lwfa.jl`'s header). Smilei's heavier bunch has two candidates, not separated here: its load rule, and its laser, stronger by 0.2 % in these runs (§5). ParticleMesh and PIConGPU load by different rules and agree to 0.14 %, so a difference of rules need not cost more than that; the run of Smilei with its amplitude corrected (§6) will say how much is the laser. The case does not test injection (A5 plans a second, injecting one).
- **Maximum energy.** The highest occupied bins of the three are within one bin (0.2 mₑc²) of each other at every output.
- **ParticleMesh against Smilei over the common region**, weight above 1 mₑc² (`above`): within 0.6 % to 1.8 % at eight of the ten outputs from t = 400, 2.9 % at the end and 4.0 % at t = 1000 (2.013 against 2.097), on a population that falls from 4 to 1 while the window passes over it. With the first runs' different regions the same two differed by up to 15 %.

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

The two backends are not bit for bit the same run (Reference mode's reproducibility is per backend, A4) and differ at 10⁻¹¹ and below at the end. That is eight orders below the smallest difference between two codes in §4.3, so none of those differences is roundoff.

### 4.6 Cost, as measured

| Run | Wall time | What the time covers | Hardware |
|---|---|---|---|
| ParticleMesh vacuum / plasma | 58.3 s / 2589.0 s | the stepping loop with its outputs (`seconds`); the box's steps with start-up took 82 s / 2606 s | AMD EPYC 9655, 16 threads |
| ParticleMesh plasma, CUDA | 393.8 s | the stepping loop with its outputs, the diagnostics on host copies | RTX 4080 SUPER |
| Smilei vacuum / plasma | 47.3 s / 385.6 s | the `smilei` process; the box's steps 48 s / 386 s | AMD EPYC 9655, 16 OpenMP threads |
| PIConGPU vacuum / plasma | 63.7 s / 142.2 s | the `picongpu` run after compilation; compiling took 78.0 s and 75.4 s more | RTX 4080 SUPER |
| focus on the plane, vacuum | ParticleMesh 63.1 s, Smilei 46.1 s, PIConGPU 63.7 s | as above | as above |

Single runs. No performance conclusion is drawn: ParticleMesh ran in Reference mode, which sorts every step, in Float64, and has had no optimization round; Smilei's ions are frozen particles; the codes ran on different hardware and with different diagnostics. A5's performance comparison (the same card, particle-steps per second) is not this.

## 5. Verification status and open issues

- **Reviewed twice, this version not** (the Review row). One run of each code at one resolution. Nothing here is a validation claim.
- **What is shown.** With the envelope and the carrier matched, the three codes agree on this case to 0.7 mrad and 0.002 c/ω₀ in the laser in vacuum, to 0.6 % in the wake's E_x over 1.5 λ_p and 0.5 % in its extrema, to 0.4 % in the maps of E_y off the axis outside the far wing, to within one bin of 0.2 mₑc² in the maximum energy and to 1.8 % in the trapped bunch's weight; ParticleMesh and PIConGPU agree to 0.07 % in the laser's amplitude, 0.12 % in the wake and 0.14 % in the bunch. The two launch conventions that separated the first runs are understood from the sources and their corrections measured (§4.1).
- **Derived, and not yet tested in a run.** Smilei's amplitude, 0.13 % to 0.22 % above the other two in both vacuum sets and in plasma: its boundary condition launches 1.001984 of the amplitude it is given (§4.2, T-710). The correction is in `smilei_laser`; the runs of §4 were made without it.
- **Hypotheses that the corrected Smilei run tests** (§6):
  - Smilei's wake, 0.16 % to 0.45 % larger than the others' in the extrema from t = 500 and 0.17 % to 0.57 % from them in E_x, is its stronger laser's. For it: the sign, and the ratio of about three between the wake's and the laser's differences in all three pairs (§4.3). The size was not computed from a model.
  - Smilei's bunch, 1.7 % to 1.8 % heavier than the other two, is its stronger laser's and not its load rule's (§4.4).
  - Smilei's carrier phase in plasma, drifting to 3.3 mrad from both others by the end where ParticleMesh and PIConGPU stay within 0.55 mrad, follows from the amplitude. Nothing measured here speaks for or against this one.
- **Unexplained**, each with its numbers above:
  - The far wing of the maps: 0.29 % to 0.72 % between every pair, ParticleMesh–PIConGPU included, where the rest of that pair's map agrees to 0.06 % to 0.13 %.
  - Smilei's E_y in the 12 to 18 cells at the back of its window, different from both other codes' by 21 % to 40 % of the field there (§4.3). The cause proposed, its boundary acting on the wake's field, was not tested.
  - The wake's maximum at t = 800, where Smilei is 0.44 % and 0.45 % above the others, about twice its excess at the neighbouring outputs (0.20 % to 0.25 %).
  - ParticleMesh against PIConGPU: a carrier phase of up to 0.70 mrad and a centroid difference of up to 0.0016 c/ω₀ in vacuum with the focus ahead (0.69 mrad and 0.0022 with it on the plane), and an amplitude difference of 0.066 % at t = 200 that falls to 0.005 % to 0.014 % from t = 700.
- **No longer on this list.** The version reviewed listed a wake difference between ParticleMesh and PIConGPU that fell from 0.17 % to 0.03 % after t = 1000, and a wake minimum at t = 800 that differed by 0.72 % and 0.79 % for the pairs with Smilei. Both were the jump at the end of the two-wavelength window (§4.3): on the interior window the first falls smoothly, 0.12 % at t = 500 to 0.02 % at t = 1100, with the two lasers' amplitude difference, and the second is 0.23 %, as at the outputs around it.
- **Not done.**
  - The run of Smilei with its amplitude corrected: prepared, not submitted (§6).
  - PIConGPU's spectrum over the common region (§4.4, #412), and with it the reader of its electron dump on a real file.
  - A second resolution (§6).
  - Float32, and the Fast mode, for ParticleMesh; A5's performance comparison.
  - The engine on `main` after `6ead5bd`: the runs are that commit's.
- **The tests.** `scripts/lwfa/test/runtests.jl` (T-510 to T-518, T-690 to T-694, T-698 to T-708, T-710, T-711): the generators, the readers and the measures on synthetic data, Smilei's boundary update on a 1D grid, and a small ParticleMesh run end to end. `lib/ParticleMeshElectromagnetic/test/antenna.jl` (T-695 to T-697, T-709): the wavefront envelope, as a function and launched through the antenna. They do not test the physics of the case.

## 6. For A7: what these runs give, and what they cannot give yet

A7 bases the benchmark thresholds on the measured Smilei–PIConGPU spread for equivalent inputs and asks that ParticleMesh agree with each of them to within it. It has accepted thresholds for the thermal and cold cases and none for LWFA. Nothing here is accepted, the owner sets A7, and this section proposes no threshold: the matched runs' inputs were not yet equivalent, and one run decides what the spread is.

**The spread of the matched runs is not yet A7's spread.** Smilei launched 1.002 of the amplitude the other two did (§4.2). In the observables that scale with the laser this is most or all of the Smilei–PIConGPU distance: the amplitude itself, and, if the hypotheses of §5 hold, the wake, its extrema and the bunch. A spread measured like this is a mismatch of the inputs of 0.2 %, as the first runs' spread was a mismatch of the launch conventions, and it is not a property of two implementations of one scheme.

**The test, prepared and not run.** Smilei alone with a₀ divided by the factor, in the three sets: the focus on the plane in vacuum, A5's case in vacuum, A5's case in plasma (`scripts/runs/lwfa-a5-3/` and `lwfa-a5-3-focus0/`, one CPU box request). ParticleMesh's and PIConGPU's runs of §4 stay, and `compare` is run again with the new Smilei summaries. What the derivation predicts for it: Smilei's amplitude within a few hundredths of a percent of the other two (the measured 0.166 % to 0.214 % against PIConGPU less the plane wave's 0.198 %, and the estimate for the beam's oblique components is of that size).

**The two outcomes.**

- *Smilei's amplitude then agrees with the other two.* The three codes launch one pulse in amplitude as well, and the Smilei–PIConGPU distances of that comparison are the first for equivalent inputs. Each hypothesis of §5 is then read off: an observable whose Smilei–PIConGPU distance comes down to the ParticleMesh–PIConGPU level was the laser's; one that stays is a difference of the two implementations, and is the spread. The table below is replaced by the new one and thresholds can be proposed from it. If everything comes down, the spread and both of ParticleMesh's distances will be of one size (ParticleMesh–PIConGPU's is 0.02 % to 0.12 % in the wake now), and whether ParticleMesh is "within the spread" at an output will be a matter of sign; the form of the rule then matters more than now (below).
- *It does not.* Then the derivation, which is exact for a plane wave on a 1D grid, does not describe what Smilei's boundary does in the 2D run, the amplitude difference is unexplained again, and the numbers below stand as the spread with a difference of the launch of unknown origin in them. The next thing to read would be the transverse terms of the 2D update (`ElectroMagnBC2D_SM.cpp`), which T-710 does not exercise.

**What is measured now**, for either outcome to be compared with: the counted observables over the outputs where they exist (vacuum: t = 200 to 1300, 12 outputs; plasma: t = 400 to 1300, 10 outputs). Each entry is the range of the magnitude; "outside" counts the outputs where ParticleMesh's distance exceeds the Smilei–PIConGPU distance at the same output, with the largest excess; the last two columns are the run's largest distance over the run's largest Smilei–PIConGPU distance.

| Observable | S–P | PM–S | PM–P | PM–S outside | PM–P outside | PM–S / S–P | PM–P / S–P |
|---|---|---|---|---|---|---|---|
| vacuum: carrier phase (mrad) | 0.03 to 0.55 | 0.01 to 0.36 | 0.23 to 0.70 | 7 (0.24 at t = 1200) | 10 (0.36 at t = 800) | 0.65 | 1.26 |
| vacuum: centroid (10⁻³ c/ω₀) | 0.22 to 1.98 | 0.07 to 0.74 | 0.32 to 1.54 | 1 (0.48 at t = 500) | 2 (0.40 at t = 400) | 0.37 | 0.78 |
| vacuum: amplitude, relative rms (%) | 0.165 to 0.215 | 0.135 to 0.189 | 0.005 to 0.066 | 6 (0.013 at t = 1200) | 0 | 0.88 | 0.31 |
| plasma: carrier phase (mrad) | 0.13 to 3.37 | 0.04 to 3.30 | 0.01 to 0.55 | 7 (0.46 at t = 600) | 3 (0.40 at t = 500) | 0.98 | 0.16 |
| plasma: centroid (10⁻³ c/ω₀) | 0.85 to 4.60 | 0.40 to 3.94 | 0.08 to 1.57 | 4 (1.57 at t = 1300) | 1 (0.40 at t = 400) | 0.86 | 0.34 |
| plasma: amplitude, relative rms (%) | 0.166 to 0.218 | 0.133 to 0.212 | 0.006 to 0.056 | 0 | 0 | 0.97 | 0.26 |
| wake E_x over 1.5 λ_p, relative rms (%) | 0.19 to 0.57 | 0.17 to 0.56 | 0.02 to 0.12 | 0 | 0 | 0.98 | 0.22 |
| wake minimum, \|a/b − 1\| (%) | 0.13 to 0.34 | 0.03 to 0.34 | 0.00 to 0.10 | 0 | 0 | 1.00 | 0.30 |
| wake maximum, \|a/b − 1\| (%) | 0.19 to 0.44 | 0.17 to 0.45 | 0.00 to 0.04 | 2 (0.019 at t = 800) | 0 | 1.05 | 0.10 |
| bunch weight, \|a/b − 1\| (%), t = 800 to 1300 | 1.68 | 1.79 | 0.14 | 6 of 6 (0.11) | 0 | 1.07 | 0.08 |
| highest occupied bin, \|a − b\| (mₑc²), t = 800 to 1300 | 0 to 0.2 | 0 to 0.2 | 0 to 0.2 | 0 | 0 | 1.0 | 1.0 |

How these read:

- **ParticleMesh is close to PIConGPU, and Smilei is the code apart**, as far as these runs go: in the observables with a scale, the largest ParticleMesh–PIConGPU distance is 0.31 of the largest Smilei–PIConGPU one in the vacuum amplitude, 0.26 in the plasma amplitude, 0.22 in the wake's E_x, 0.30 and 0.10 in its minimum and maximum, 0.08 in the bunch's weight. Output by output the wake's ratio is 0.32 at t = 400, 0.26 at t = 500, 0.21 at t = 600, 0.13 at t = 700 and 0.04 to 0.08 after. ParticleMesh's distance to Smilei is then the Smilei–PIConGPU distance itself, give or take ParticleMesh–PIConGPU, and falls inside or outside it by hundredths of a percent, by the sign of that small term: that is all the "outside" entries of the vacuum amplitude, the wake's maximum and the bunch are.
- **The laser's phase and centroid are signed and cross zero**, so the distance of two codes at a single output can be arbitrarily small (0.03 mrad) while every pair stays below 0.7 mrad and 0.002 c/ω₀ in vacuum. An output-by-output comparison of such a quantity with the spread does not measure agreement.

The forms a rule for this case could take, and what the runs so far say about each. No numbers are proposed for any of them here.

1. **A7's sentence as it stands, output by output**: |ParticleMesh − each code| ≤ |Smilei − PIConGPU| at the same output, for every counted observable. On these runs ParticleMesh fails it in the "outside" cells above, for the two reasons just given, and the rule would also fail a code that reproduced PIConGPU exactly (its distance to Smilei would be the spread, and equal is the edge). With the inputs made equivalent the three distances may be of one size, and this form then decides by sign.
2. **A stated multiple of the spread's largest value over the run, per observable**, the form A7 accepted for the cold two-stream. It removes the chance of sign. It rejects a mismatched launch only if the spread is the matched runs', measured once and stored, and not the spread of whichever runs are being judged: held to the first runs' own Smilei–PIConGPU centroid spread of 0.264 c/ω₀, ParticleMesh's first-run centroid distance of 0.405 c/ω₀ from PIConGPU is 1.5 of it and passes a multiple of 2, while against the matched runs' spread it is 200 times too far.
3. **Fixed thresholds per observable, set once from matched runs.** They read more plainly than a multiple, do not move when a reference run is repeated, and can be asked of ParticleMesh against one reference code where the other is not available; they are tied to this case, this resolution and these windows.

Whichever form:

- the wake's measure on a window that ends inside a bucket (§4.3), and stated with its length;
- the laser's phase and centroid on the run's largest value or an absolute number, not output by output;
- a measure off the axis among the counted ones, since the launch conventions that separated the first runs are equal on the axis of the plane (§4.3), with the cells at the back of the windows left out;
- the bunch's weight as a weak measure on this case: it is 7 macro-particle weights carried by 35 or more particles from the ramp's foot (§4.4).

What a second resolution would add. Each code ran once at one resolution, and the runs are deterministic (cold regular loads, no seeds), so no difference here is noise. One part of the Smilei–PIConGPU distance is now known to be discretization of a known size: the launch factor, (Δx² + Δt²)/16 above 1, is 0.198 % at λ₀/48 and 0.447 % at λ₀/32. For the rest, one resolution cannot say whether a difference shrinks as the grid is refined or stays (the load rules would stay). A second resolution in all three codes, λ₀/32 (the grid of `lwfa-2d-a2.toml`) or λ₀/64, would give the spread's dependence on Δ and each code's own change with resolution, which is the scale a threshold has to be compared with: at λ₀/48 the observables themselves are converged to some percent that these runs do not measure, and a threshold tighter than that is a statement about two codes sharing an error, not about either being right.
