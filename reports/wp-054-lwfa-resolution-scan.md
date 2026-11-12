# LWFA 2D on three grids: A5's first case at λ₀/32, λ₀/48 and λ₀/64 in ParticleMesh, Smilei and PIConGPU

**Summary.**
- The scan: A5's first 2D laser-wakefield case on Δx = λ₀/32, λ₀/48 and λ₀/64 (Δy = 4Δx, Δt ∝ Δx, the outputs at the same times) in ParticleMesh (CUDA), Smilei v5.1 and PIConGPU 0.8.0: 18 runs, compared on the lattice points the three grids share.
- Converges, in each code: the laser, at second order. Carrier phase and centroid differences between grids have the order 2.01 at all twelve vacuum outputs in ParticleMesh and in Smilei (1.74 to 2.58 in PIConGPU) and are the axial Yee dispersion relation's to 2.5 %. At the end λ₀/48 is 56 mrad and 0.17 c/ω₀ from λ₀/64, and λ₀/64 by extrapolation another 72 mrad and 0.22 c/ω₀ from the limit.
- Converges without a constant order: the wake. Its E_x at λ₀/48 is 0.5 % to 1.4 % (relative rms over 1.5 λ_p) from λ₀/64, its maximum 0.3 % to 2.0 %; the trapped bunch's weight is 8 % to 11 % higher at λ₀/32 than at λ₀/64 and within 1.3 % at λ₀/48.
- So at λ₀/48 a code is further from its own λ₀/64 run than from another code by a factor of 80 to 1100 in the phase, 80 to 220 in the centroid and 3 to 12 in the wake: the three share the scheme's error, and a code-to-code distance measures whether two implementations solve one discrete problem, not how accurately.
- Does not converge: ParticleMesh against Smilei. Only their carrier phase falls with the grid (0.118, 0.052, 0.036 mrad, about third order); the vacuum amplitude (0.004 %), the centroid (0.0006 to 0.0008 c/ω₀), the wake's E_x (0.18, 0.15, 0.14 %) and its extrema (0.06 % to 0.12 %) stay within a factor of 1.35 across the grids or within their own uncertainty, and the maps of E_y fall by a third (0.075, 0.058, 0.050 %). These are differences of what the two codes launch or solve, and they are not explained.
- The distances to PIConGPU follow its launch term, which a 1D model built on the λ₀/48 runs gave for the other two grids before they ran: 1.75 and 0.18 of the oblique term predicted at λ₀/32 and λ₀/64, 1.73 and 0.175 fitted (0.83 and 0.82 at λ₀/48). At λ₀/64, where the term is smallest, ParticleMesh and PIConGPU are 0.016 % (vacuum amplitude), 0.20 mrad, 0.022 % (wake E_x) and 0.020 % (maps) apart. The centroid distances to PIConGPU (0.0015 to 0.0046 c/ω₀) follow neither the grid nor that term.
- What WP-053 left open: Smilei's wake difference is on every grid (0.11 % to 0.18 % at t = 700 to 900) and at λ₀/64 it is Smilei against two codes whose wakes agree to 0.02 %; Smilei's heavier bunch is 0.02 to 0.10 of one macro-particle weight on every grid (2.0 %, 1.4 %, 0.6 % of the bunch); its E_y at the back of its window is off by 23 % to 46 % in the first kept column on every grid, and the depth of that region in six maps here and two of WP-053 is c times the time since Smilei's window last moved; the far wing against Smilei falls slowly (0.34, 0.28, 0.24 %). The first two and the last are not explained.
- For a threshold on this case (§7, four options, none proposed): fixed bounds at 1.6 to 2.5 times the ParticleMesh–Smilei distances pass that pair on all three grids and fail ParticleMesh–PIConGPU in 10, 5 and 3 of 12 rows; a bound relative to Smilei–PIConGPU passes at a factor 1.5 and fails at 1; neither bounds the error of the solution, which only a bound across grids does.
- Not shown: ParticleMesh on the CPU (CUDA only here), Float32 and its Fast mode; a fourth grid, so every order is one equation with no check; Δy, Δt or the particles per cell varied alone; a PIConGPU run that tests the model of its launch; PIConGPU's spectrum over the common region; a repeat of any run. This is not an A7 pass.

| | |
|---|---|
| Work package | WP-054 (`trace/wp/WP-054-lwfa-resolution-scan.md`); goals A5, A7. Follows WP-053 (`reports/lwfa-2d-a5.md`, PR #413) |
| Cases | `scripts/lwfa/cases/lwfa-2d-r32.toml` (sha256 `9be9531b…`), `lwfa-2d-r48.toml` (`51b1d5fc…`), `lwfa-2d-r64.toml` (`0b0dc953…`); each run's manifest has the hash of its copy |
| Commits | ParticleMesh ran the code of `1e36a43` (its six manifests, `dirty = false`). Smilei's and PIConGPU's inputs were generated at `a991f96` (their manifests); `1e36a43` differs from it in the comparison code and its tests only (`git diff --stat a991f96 1e36a43`: `scripts/lwfa/lwfa.jl`, `scripts/lwfa/test/runtests.jl`). The comparison ran at `ae844e7` (`scan/manifest.toml`, `dirty = false`). These are commits of the branch `worktree-wp-054-lwfa-resolution-scan`; a squash merge does not keep them reachable from `main` |
| Runs | ParticleMesh: six runs on CUDA, `lwfa-scan-r<N>/pm-vacuum` and `pm-plasma`. PIConGPU: six gpu-run jobs, `scan-r<N>-pic-vacuum` and `-plasma` (the job names as the coordinating session gave them; each run's start and end are in its `container.txt`). Smilei: one CPU box request, `runs/lwfa-scan-smilei` (request `406a48b`, result `ba94009`), six runs; its summaries and maps stay on that branch. The summaries of ParticleMesh and PIConGPU are on no branch and are in `wp-054-lwfa-resolution-scan/runs/`, with the maps of their plasma runs, which the off-axis rows read |
| Input hashes | each run's `manifest.toml` as `scan/r<N>/manifest-<run>.toml`; the box's `scan/box-status-smilei.toml`; the sha256 of the 18 summaries in `scan/manifest.toml`, and per grid of the summaries and of the three plasma runs' `maps.h5` in `scan/r<N>/manifest.toml` |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode, CUDA on an RTX 4080 SUPER. Smilei v5.1 (image `particlemesh-smilei:e4f0148b8952fc87`), 16 OpenMP threads on an AMD EPYC 9655. PIConGPU 0.8.0 (`6bb3470e`), CUDA, Float64, the same RTX 4080 SUPER |
| Figures | `wp-054-lwfa-resolution-scan/scan/scan_self.png` and `scan_distances.png`, and per grid `scan/r<N>/pair_distances.png` and `spectrum.png`, each with its JSON sidecar. The measured values below are keys of `scan/scan.json` (`self`, `distances`, `launch`) and of `scan/r<N>/comparison.json`; ranges, largest values, counts and ratios are computed from those keys. Four sets of numbers are computed here from those keys or from the maps and are in no JSON file: the comparison with the Yee dispersion relation (§4.1), the wake against the driver's lag (§4.2), the estimate of a centroid shift (§5.4) and the columns at the back of Smilei's window (§5.5); each gives its formula |
| Status | measured, not independently reviewed (a review follows) |

## 1. Setup

The case is that of `reports/lwfa-2d-a5.md` §1 in A1's units: a₀ = 2, waist 28, τ = 26, focus at x = 320; electrons of 0.01 n_c (λ_p = 62.8) behind 260 of vacuum and a 60 sin² ramp, 2 × 2 per cell, cold, ions immobile; a window moving at c from t = 256; Yee, Esirkepov, quadratic shapes, Boris, no current filter, Float64. The laser's launch is the matched one of that report: the cosine carrier from the envelope's peak with the Gouy phase and the wavefront envelope in all three, Smilei with its four corrections (`smilei_laser`), PIConGPU's launch term modelled and not removed.

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| Δx, Δy = 4Δx (c/ω₀) | 0.1963, 0.7854 | 0.1309, 0.5236 | 0.0982, 0.3927 |
| Δt, 0.95 of Yee's 2D limit | 0.18096 | 0.12064 | 0.09048 |
| window (cells) | 1536 × 256 | 2304 × 384 | 3072 × 512 |
| steps, and steps per output | 7176, 552 | 10 764, 828 | 14 352, 1104 |
| antenna (cells from the edge) | 40 | 60 | 80 |
| macro-particle weight at full density, n_c (c/ω₀)² | 3.855 × 10⁻⁴ | 1.713 × 10⁻⁴ | 9.638 × 10⁻⁵ |
| Smilei's plane-wave launch factor, divided out | 1.004474 | 1.001984 | 1.001115 |

Kept on every grid: the box in c/ω₀, the antenna's distance from the edge, the window's start, the cell's shape Δy/Δx = 4, the Courant number, 2 × 2 particles per cell, the spectrum's bins, the maps' spacing (λ₀/8 by λ₀/4, three outputs), and the output times, t = k · 99.891 for k = 0 to 13, the first before anything is in the box (`scan_same_case` refuses grids that differ in any of these, T-714). Changed with the grid besides Δ: the macro-particle weight (∝ Δ²), Smilei's injection node (two cells behind xmin, so 2Δx) and its plane-wave factor, and everything in PIConGPU that is counted in cells: its Huygens surface and its 12-cell absorber. Δx, Δy and Δt are refined together, so the scan gives one order for the three, and nothing about any of them alone.

The λ₀/48 case is not `scripts/lwfa/cases/lwfa-2d.toml`: that one has 829 steps per output, which no cadence on the other two grids matches, so its outputs are 0.12 to 1.6 time units later than here.

ParticleMesh ran on CUDA only in this scan. That its CPU and CUDA backends give one result on this case is WP-053's evidence, a pair at λ₀/48 whose lineouts differ by 10⁻¹¹ and less at the end (`reports/lwfa-2d-a5.md` §4.5, `lwfa-2d-a5/lwfa-a5-2-cuda-pair/pair.json`); it was not repeated at λ₀/32 or λ₀/64. All six ParticleMesh runs lost no particle (`lost = [0]` in the summaries).

## 2. Measures

**Between two codes on one grid**: `compare`'s, unchanged from `reports/lwfa-2d-a5.md` §2 and run per grid into `scan/r<N>/comparison.json`. A pair is written a–b (PM–S ParticleMesh–Smilei, PM–P ParticleMesh–PIConGPU, S–P Smilei–PIConGPU); the wake's window is 1.5 λ_p behind ParticleMesh's envelope peak; the maps leave out one laser wavelength at the back of the shared columns. `scan.json` has under `distances` the largest magnitude of each over a run, per pair and grid (`scan_distances`). A run's outputs count after `scan.json`'s `from`: t = 150 in vacuum (12 outputs, t = 200 to 1300); in the plasma runs t = 364, when the pulse's peak reaches the plasma (10 outputs, t = 400 to 1300, the ones WP-053 counts; before it a plasma run repeats its vacuum run); t = 750 for the bunch (6 outputs), from where the spectrum's highest group stands clear of the rest.

**One code across the grids** (`scan_self`, T-715, T-717): the primal nodes of the three lattices coincide every λ₀/16, so
- E_y on the axis, its analytic signal, the centroid of E_y² and the carrier phase against the finest grid are taken on the shared nodes, with nothing interpolated;
- E_x, which is on dual points that no two grids share, is brought to the primal nodes by the four-point midpoint formula (error 3/128 · Δx⁴ ∂⁴E_x) and compared against the finest grid over the 1.5 λ_p behind the finest grid's centroid; the wake's minimum and maximum are taken on each grid's own points, refined by a parabola;
- the axis lineout of E_y is the mean of the two rows at ∓Δy/2, a different place on every grid. In vacuum the beam's profile gives what that does to the amplitude, exp(−(Δy/2)²/w(ξ)²), and `amplitude_on_axis` has it divided out (0.02 %, 0.009 % and 0.005 % at focus on the three grids); in plasma it is left in.

**An order** (`observed_order`): three values f₃₂, f₄₈, f₆₄ with f = f∞ + C·Δᵖ have (f₃₂ − f₄₈)/(f₄₈ − f₆₄) = 2.00 for p = 1, 2.86 for p = 2, 4.11 for p = 3. An order and the extrapolated f∞ are given only when the two differences have one sign, exceed a floor and the ratio is that of an order between 0.5 and 4; a distance between codes, being a magnitude, also has none when it extrapolates below zero by more than half its finest value. Three grids give one equation for p and no check of it: an order below says that three values are compatible with one, and what it is worth is its constancy over the outputs.

**A launch difference** (`launch_fit`, T-718): from a pair's vacuum rows, the multiple c of the oblique term (a launch that multiplies the transverse spectrum by 1 − c·k_y²/4) and a focus offset δ, each fitted once from the amplitudes and once from the phases (`reports/lwfa-2d-a5.md` §3.3 has the shapes).

## 3. Two checks before the results

### 3.1 The λ₀/48 runs against WP-053's

The λ₀/48 runs here and the final set of WP-053 (`lwfa-2d-a5/lwfa-a5-4-compare/comparison.json`) are the same case but for the outputs' instants (828 against 829 steps apart) and ParticleMesh's side: CUDA at `1e36a43` here, the CPU at `6ead5bd` there. Largest over the run, PM–S / PM–P / S–P:

| | here | WP-053's final set |
|---|---|---|
| vacuum carrier phase (mrad) | 0.0518 / 0.696 / 0.650 | 0.0518 / 0.696 / 0.650 |
| vacuum centroid (10⁻³ c/ω₀) | 0.767 / 1.544 / 2.015 | 0.767 / 1.544 / 2.014 |
| vacuum amplitude, relative rms (%) | 0.00378 / 0.0657 / 0.0645 | 0.00378 / 0.0657 / 0.0646 |
| plasma carrier phase (mrad) | 0.0516 / 0.549 / 0.500 | 0.0516 / 0.549 / 0.500 |
| plasma centroid (10⁻³ c/ω₀) | 0.746 / 1.564 / 1.713 | 0.748 / 1.571 / 1.715 |
| plasma amplitude, relative rms (%) | **0.0281** / 0.0564 / 0.0562 | **0.0315** / 0.0563 / 0.0561 |
| wake E_x over 1.5 λ_p (%) | 0.154 / 0.123 / 0.151 | 0.155 / 0.123 / 0.152 |
| wake minimum (%) | **0.097** / 0.100 / 0.110 | **0.086** / 0.100 / 0.111 |
| wake maximum (%) | **0.108** / 0.043 / **0.082** | **0.073** / 0.042 / **0.060** |
| maps of E_y: all rows, flank, far wing, PM–S (%) | 0.058, 0.071, 0.281 | 0.058, 0.070, 0.290 |
| the same, PM–P (%) | 0.070, 0.129, 0.684 | 0.070, 0.129, 0.685 |
| the same, S–P (%) | 0.077, 0.140, 0.662 | 0.077, 0.141, 0.663 |

Reproduced to 0.5 % of the value or better: every laser row but one; to 1 %, the maps over all rows and the flank; to 3 %, the far wing. The wake's E_x agrees to 1 % in its run maximum and to 5 % per output. Not reproduced, in bold, all in pairs with Smilei: the plasma amplitude's relative rms for PM–S (11 % lower here), the wake minimum for PM–S (13 % higher) and the wake maximum for PM–S and S–P (48 % and 36 % higher). Nothing but the output instants differs for Smilei and PIConGPU between the two sets, so a run maximum of an extremum's ratio moves by up to half its value when the outputs are taken 0.4 to 1.6 time units earlier: those rows are known to a factor of 1.5, not to a digit. The highest occupied bin is the same in the three codes at all ten plasma outputs here (in WP-053's set PIConGPU's was one bin lower at t = 1100).

Because ParticleMesh's side changed between the sets, the table also says that the engine of `1e36a43` on CUDA and that of `6ead5bd` on the CPU give this case to the differences above. It is no test of either beyond that, since no output is at the same step.

### 3.2 PIConGPU's launch term, predicted per grid

WP-053 modelled PIConGPU's launch in 1D: B = k̂ × E on its Huygens surface, with its default 12-cell exponential absorber behind. Surface and absorber are counted in cells, so the model gives a different term on each grid, and T-716 fixed its three values before these runs (commit `a991f96`). Fitted from the runs (`scan.json`, `launch`; 12 vacuum outputs each; from the amplitude / from the phase):

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| oblique term, model (T-716) | 1.75 | 0.83 | 0.18 |
| fitted, PM–P | 1.727 / 1.735 | 0.818 / 0.820 | 0.175 / 0.175 |
| fitted, S–P | 1.743 / 1.741 | 0.824 / 0.823 | 0.177 / 0.177 |
| focus further out (c/ω₀), model | −0.15 | +0.44 | +0.12 |
| fitted, PM–P | −0.086 / −0.110 | +0.464 / +0.462 | +0.151 / +0.143 |
| fitted, S–P | −0.141 / −0.125 | +0.422 / +0.444 | +0.113 / +0.125 |
| fitted, PM–S: oblique term | −0.016 / −0.006 | −0.006 / −0.003 | −0.002 / −0.002 |
| fitted, PM–S: focus | +0.054 / +0.015 | +0.042 / +0.018 | +0.038 / +0.018 |

The prediction holds: the oblique term to 0.03 of its unit on every grid, with the sign change of the focus offset between λ₀/32 and λ₀/48, and the focus to 0.07 c/ω₀. The fits leave at most 0.002 % in amplitude and 0.003 mrad in phase. The λ₀/48 values were known when the model was built (WP-053 fitted 0.82 and a focus 0.42 to 0.47 further out there); the predictions are those for λ₀/32 and λ₀/64, runs the model had not seen, and both hold. This is a model confirmed on two more grids; it is still not a PIConGPU run with another absorber or another incident B, and PIConGPU's launch is not corrected.

Between ParticleMesh and Smilei the fit finds no oblique term on any grid, a focus offset that does not depend on the grid and is not the same from the amplitude (0.04 to 0.05 c/ω₀) and from the phase (0.015 to 0.018), so it is not a focus offset, and a phase constant of −0.098, −0.029, −0.013 mrad, whose differences have the ratio of a third order.

## 4. Each code against itself across the grids

`scan.json`, `self`; `scan_self.png`. The magnitude of the difference from the code's own λ₀/64 run, as a range over the outputs, and the orders the guard of §2 lets through: how many of the outputs have one, and their range.

| | ParticleMesh | Smilei | PIConGPU |
|---|---|---|---|
| **vacuum, 12 outputs** | | | |
| amplitude on the axis: λ₀/32 (%) | 0.008 to 0.232 | 0.009 to 0.230 | 0.042 to 0.261 |
| λ₀/48 (%) | 0.002 to 0.060 | 0.002 to 0.059 | 0.040 to 0.052 |
| order | 12 of 12: 1.81 to 2.31 | 12 of 12: 1.82 to 2.30 | 9 of 12: 1.40 to 3.10 |
| centroid: λ₀/32 (c/ω₀) | 0.052 to 0.655 | 0.052 to 0.655 | 0.050 to 0.653 |
| λ₀/48 (c/ω₀) | 0.013 to 0.169 | 0.013 to 0.169 | 0.011 to 0.170 |
| order | 12 of 12: 2.01 | 12 of 12: 2.01 | 12 of 12: 1.97 to 2.58 |
| carrier phase: λ₀/32 (mrad) | 17.1 to 218 | 17.2 to 218 | 16.4 to 219 |
| λ₀/48 (mrad) | 4.42 to 56.5 | 4.44 to 56.5 | 4.55 to 56.7 |
| order | 12 of 12: 2.01 | 12 of 12: 2.01 | 12 of 12: 1.74 to 2.00 |
| **plasma, 10 outputs** | | | |
| amplitude: λ₀/32 (%) | 0.007 to 0.316 | 0.008 to 0.308 | 0.104 to 0.390 |
| λ₀/48 (%) | 0.012 to 0.090 | 0.002 to 0.092 | 0.038 to 0.095 |
| order | 8 of 10: 1.03 to 2.94 | 7 of 10: 0.97 to 3.47 | 8 of 10: 1.85 to 2.78 |
| centroid: λ₀/32 (c/ω₀) | 0.160 to 0.620 | 0.160 to 0.620 | 0.155 to 0.620 |
| λ₀/48 (c/ω₀) | 0.041 to 0.159 | 0.041 to 0.160 | 0.039 to 0.158 |
| order | 10 of 10: 1.91 to 2.07 | 10 of 10: 1.92 to 2.06 | 10 of 10: 1.89 to 2.15 |
| carrier phase: λ₀/32 (mrad) | 52.3 to 185 | 52.4 to 185 | 51.9 to 184 |
| λ₀/48 (mrad) | 13.5 to 47.8 | 13.5 to 47.8 | 13.9 to 47.8 |
| order | 10 of 10: 1.99 to 2.01 | 10 of 10: 1.99 to 2.02 | 10 of 10: 1.88 to 1.99 |
| wake E_x lineout, relative rms: λ₀/32 (%) | 1.73 to 5.16 | 1.73 to 5.16 | 1.74 to 5.12 |
| λ₀/48 (%) | 0.48 to 1.41 | 0.48 to 1.41 | 0.51 to 1.39 |
| order | 9 of 10: 1.31 to 1.95 | 9 of 10: 1.31 to 1.95 | 9 of 10: 1.26 to 1.97 |
| wake minimum: λ₀/32 (%) | 0.92 to 1.78 | 0.92 to 1.94 | 1.04 to 1.83 |
| λ₀/48 (%) | 0.02 to 0.77 | 0.0002 to 0.80 | 0.03 to 0.78 |
| order | 6 of 10: 1.91 to 2.72 | 7 of 10: 1.93 to 2.46 | 7 of 10: 0.82 to 2.74 |
| wake maximum: λ₀/32 (%) | 1.66 to 5.91 | 1.66 to 5.93 | 1.69 to 6.06 |
| λ₀/48 (%) | 0.31 to 1.96 | 0.34 to 1.84 | 0.32 to 1.94 |
| order | 10 of 10: 0.91 to 3.60 | 10 of 10: 0.77 to 3.37 | 10 of 10: 0.84 to 3.57 |
| places of the two extrema (c/ω₀) | 0.01 to 0.91, 1 order in 19 | 0.007 to 1.01, 1 in 19 | 0.01 to 0.75, none |
| **electrons, 6 outputs from t = 800** | | | |
| bunch weight: λ₀/32 (%) | 8.0 to 8.7 | 9.6 to 10.7 | 8.5 to 9.2 |
| λ₀/48 (%) | 0.26 to 0.36 | 0.14 to 1.23 | 0.04 to 1.29 |
| order | none | none | 1 of 6 |

### 4.1 The laser: second order, and the Yee dispersion relation's

The carrier phase and the centroid have the order 2.01 at every vacuum output in ParticleMesh and in Smilei, and 1.99 to 2.02 and 1.91 to 2.07 in plasma. That is as constant as an order from three values can be, and the size is the scheme's own: on the axis the Yee grid carries the wave with k = (2/Δx)·asin((Δx/Δt)·sin(Δt/2)) and the group velocity cos(kΔx/2)/cos(Δt/2), which gives k − 1 = 2.429, 1.077, 0.605 × 10⁻⁴ and 1 − v_g = 7.30, 3.23, 1.82 × 10⁻⁴ on the three grids. The measured phase difference from the finest grid over (k_N − k₆₄)·x_c, and the measured centroid difference over (v_g,N − v_g,₆₄)·x_c/v_g,₆₄, with x_c the finest grid's centroid, at the twelve vacuum outputs and both coarser grids:

| measured over predicted | ParticleMesh | Smilei | PIConGPU |
|---|---|---|---|
| carrier phase | 0.975 to 1.003 | 0.976 to 1.003 | 0.938 to 1.007 |
| centroid | 0.980 to 1.000 | 0.988 to 1.000 | 0.817 to 1.002 |

At the last output ParticleMesh's phase differences are 218.4 and 56.5 mrad against 217.8 and 56.3 predicted, its centroid differences −0.6550 and −0.1693 c/ω₀ against −0.6549 and −0.1693. The prediction is for a plane wave on the axis; a focused beam's oblique components and the axis lineout's rows are not in it, and the ratios are furthest from one at the first outputs, where the differences are smallest. PIConGPU's are further off there because its launch term, which differs between its grids (§3.2), is a larger part of a small difference.

What the finest grid still lacks follows from the same two numbers: by the formula the λ₀/64 run is 72.3 mrad and 0.217 c/ω₀ behind the continuum at the last output, and the extrapolation from the three runs gives 72.3 mrad and 0.216 c/ω₀ (ParticleMesh; Smilei's limits differ from ParticleMesh's by 0.002 mrad and 0.0004 c/ω₀). The two agree because both are the second-order term; neither is a measurement on a finer grid.

The amplitude on the axis is second order from t = 500 (1.97 to 2.03 in ParticleMesh and Smilei) and has other ratios before, where the difference from the finest grid is 0.002 % to 0.025 % and smallest at t = 400 (orders 1.81 to 2.31 at those three outputs).

### 4.2 The wake: converging, no constant order

The wake's E_x lineout at λ₀/48 is 0.48 % to 1.41 % from the finest grid's, in every code, with orders 1.3 to 2.0 at nine of ten outputs. Most of that is the driver's place: the coarser grid's pulse is behind the finest grid's by the centroid differences of the table, and a wake of wavenumber k_p = 0.1 displaced by that lag differs by k_p times it in relative rms. For ParticleMesh the measured lineout difference is 0.83 to 1.64 times k_p · lag over the ten outputs and both coarser grids (at the end 1.36 % against 1.59 % at λ₀/48). This is an estimate for a sinusoid, computed here; the wake was not shifted and compared again.

The extrema, which do not depend on the wake's place, converge without one order: the maximum at λ₀/48 is 0.3 % to 2.0 % below the finest grid's and the minimum 0.02 % to 0.8 % shallower, with orders from 0.8 to 3.6 over the outputs. The maximum is furthest from the finest grid's at t = 700 and 800 (1.2 % and 2.0 % low at λ₀/48, 4.7 % and 5.9 % at λ₀/32). The three codes show the same pattern at the same outputs, so it is the discretized problem's and not a code's.

### 4.3 The electrons

The bunch at λ₀/32 is 8 % to 11 % heavier than at λ₀/64 and at λ₀/48 within 1.3 % of it, in every code: no order, and none expected, since the bunch is 3.5, 7.3 and 12.9 full-density macro-particle weights on the three grids. At λ₀/64 the rule that defines the bunch (the occupied bins above the widest empty gap) cuts at different places in the three codes at t = 800 and 900 (from 14.8, 13.2 and 18.4 mₑc² at t = 800, where the widest gaps are 4.0, 2.2 and 0.2 wide): the weights are 1.2503, 1.2579 and 1.2329 × 10⁻³ at t = 800 against 1.2426, 1.2444 and 1.2437 × 10⁻³ at the end. The highest occupied bin at the end is 65.8, 67.2 and 67.6 mₑc² on the three grids, in all three codes: it rises by 1.4 and then 0.4 with the grid, which is a second-order ratio as far as bins of 0.2 can say.

## 5. The codes against each other on each grid

`scan/r<N>/comparison.json`; `scan.json`, `distances`; `scan_distances.png`. The largest magnitude over the run on λ₀/32, λ₀/48, λ₀/64:

| | PM–S | PM–P | S–P |
|---|---|---|---|
| **vacuum, 12 outputs** | | | |
| carrier phase (mrad) | 0.118, 0.052, 0.036 | 0.58, 0.70, 0.20 | 0.70, 0.65, 0.17 |
| centroid (10⁻³ c/ω₀) | 0.61, 0.77, 0.79 | 4.41, 1.54, 1.65 | 3.97, 2.01, 2.42 |
| amplitude, relative rms (%) | 0.0040, 0.0038, 0.0037 | 0.111, 0.066, 0.016 | 0.112, 0.065, 0.015 |
| **plasma, 10 outputs** | | | |
| carrier phase (mrad) | 0.113, 0.052, 0.038 | 1.05, 0.55, 0.17 | 1.06, 0.50, 0.14 |
| centroid (10⁻³ c/ω₀) | 0.61, 0.75, 0.77 | 4.58, 1.56, 1.74 | 3.96, 1.71, 2.41 |
| amplitude, relative rms (%) | 0.037, 0.028, 0.029 | 0.114, 0.056, 0.013 | 0.116, 0.056, 0.029 |
| wake E_x over 1.5 λ_p, relative rms (%) | 0.182, 0.154, 0.136 | 0.395, 0.123, 0.022 | 0.400, 0.151, 0.138 |
| wake minimum, a/b − 1 (%) | 0.123, 0.097, 0.117 | 0.160, 0.100, 0.037 | 0.171, 0.110, 0.118 |
| wake maximum, a/b − 1 (%) | 0.056, 0.108, 0.055 | 0.273, 0.043, 0.011 | 0.243, 0.082, 0.058 |
| **maps of E_y, 3 outputs** | | | |
| all rows (%) | 0.075, 0.058, 0.050 | 0.152, 0.070, 0.020 | 0.162, 0.077, 0.052 |
| flank (%) | 0.093, 0.071, 0.064 | 0.228, 0.129, 0.034 | 0.242, 0.140, 0.071 |
| far wing (%) | 0.34, 0.28, 0.24 | 0.96, 0.68, 0.20 | 0.94, 0.66, 0.27 |
| **electrons** | | | |
| bunch weight, 6 outputs (%) | 2.01, 1.35, 0.60 | 0.92, 0.14, 1.41 | 2.98, 1.23, 2.03 |
| highest occupied bin, 10 outputs (mₑc²) | 0, 0, 0 | 0.2, 0, 0 | 0.2, 0, 0 |

The extremum rows are known to a factor of 1.5 (§3.1). PIConGPU's spectrum covers its whole simulated domain and the other two a common region, as in WP-053, so its bunch rows compare different regions; at λ₀/64 the bunch rows of every pair also depend on where the rule cuts (§4.3).

Orders the guard lets through, of the 42 rows of `scan.json`'s `distances` (14 per pair): PM–S vacuum carrier phase 3.00 (towards 0.024 mrad), PM–S plasma carrier phase 3.37 (towards 0.031 mrad), PM–S vacuum amplitude 2.25 (towards 0.0036 %, that is, a fall of 7 % in all), PM–S maps over all rows 1.18 and flank 2.44 (towards 0.030 % and 0.057 %), S–P maps over all rows 2.42 (towards 0.027 %). No other row has the ratio of an order between 0.5 and 4 towards a distance that is not negative: they stay, rise, fall and rise, or fall faster at the fine end than a power of Δ does (the wake's E_x for PM–P: 0.395, 0.123, 0.022 %).

### 5.1 What the table says

**ParticleMesh–Smilei: one row falls, the others stay.** The carrier phase falls by a factor 3.3 from λ₀/32 to λ₀/64, at about third order and towards 0.02 to 0.03 mrad; with it goes the fitted phase constant (§3.2). Of the other rows, the vacuum amplitude (0.004 %), the centroid (0.0006 to 0.0008 c/ω₀, growing slightly with the grid), the plasma amplitude (0.03 % to 0.04 %), the wake's E_x (0.14 % to 0.18 %) and its minimum are within a factor 1.35 across the three grids; the wake's maximum is within the factor of 1.5 that an extremum row is known to; the maps fall by a third over all rows and the flank, by the guard's orders towards 0.03 % and 0.06 %, and by 28 % in the far wing. A distance that does not fall when the cell shrinks by half is not a truncation error that the two codes commit differently: it is a difference in what they launch or solve that does not depend on the grid. What it is has not been found (§5.2 to §5.6 narrow it), and the scan shows only that refining does not remove it.

**ParticleMesh–PIConGPU: the launch term.** In vacuum the amplitude distance is 0.111, 0.066, 0.016 %, in the ratio 6.8 : 4.0 : 1; the fitted oblique term is 1.73, 0.82, 0.175, in the ratio 9.9 : 4.7 : 1, and with the fitted focus offset it leaves 0.001 % or less of the peak amplitude's ratio at every output (§3.2). The vacuum phase is largest at λ₀/48, where the fitted focus offset is largest (+0.46 c/ω₀). So these rows are not a convergence of two codes towards each other: they are one launch term whose size depends on the grid through PIConGPU's cell counts, which the model has small at λ₀/64; which part of the surface and the absorber makes it so was not analysed. Where it is small the pair agrees closely: at λ₀/64, 0.016 % in the vacuum amplitude, 0.20 mrad, 0.022 % in the wake's E_x, 0.011 % and 0.037 % in its extrema, 0.020 % in the maps, 0.034 % in the flank. Whether the wake rows at λ₀/32 and λ₀/48 are that term alone was not tested: no run has the term removed.

**Smilei–PIConGPU: both.** At λ₀/32 and λ₀/48 it follows PM–P; at λ₀/64 its plasma rows are PM–S's (wake 0.138 % against 0.136 %, maps 0.052 % against 0.050 %) while its vacuum rows are PM–P's. On the finest grid, then, the three codes are two that agree to 0.02 % in the wake's E_x and in the maps, and a third 0.14 % and 0.05 % from both.

### 5.2 Smilei's wake difference

WP-053 found Smilei's wake apart from both other codes from t = 700 at λ₀/48. The wake's E_x per output (relative rms over 1.5 λ_p, %; PM–S / PM–P / S–P):

| t | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| 400 | 0.049 / 0.104 / 0.103 | 0.033 / 0.062 / 0.066 | 0.024 / 0.022 / 0.029 |
| 500 | 0.077 / 0.320 / 0.321 | 0.062 / 0.123 / 0.130 | 0.053 / 0.014 / 0.055 |
| 600 | 0.101 / 0.395 / 0.400 | 0.081 / 0.108 / 0.130 | 0.069 / 0.017 / 0.071 |
| 700 | 0.135 / 0.362 / 0.389 | 0.121 / 0.066 / 0.144 | 0.114 / 0.018 / 0.116 |
| 800 | 0.182 / 0.325 / 0.367 | 0.154 / 0.043 / 0.151 | 0.136 / 0.017 / 0.138 |
| 900 | 0.170 / 0.244 / 0.273 | 0.121 / 0.037 / 0.127 | 0.136 / 0.016 / 0.137 |
| 1000 | 0.126 / 0.177 / 0.212 | 0.100 / 0.023 / 0.105 | 0.097 / 0.012 / 0.098 |
| 1100 | 0.124 / 0.132 / 0.178 | 0.085 / 0.021 / 0.085 | 0.074 / 0.009 / 0.075 |
| 1200 | 0.093 / 0.109 / 0.137 | 0.069 / 0.026 / 0.073 | 0.056 / 0.006 / 0.057 |
| 1300 | 0.080 / 0.094 / 0.118 | 0.068 / 0.030 / 0.073 | 0.056 / 0.004 / 0.058 |

The difference is on every grid, with one course in time: it grows from the pulse's entry into the plasma, is largest at t = 800 or 900 (0.182, 0.154, 0.136 %) and falls to 0.06 % to 0.08 % at the end. From λ₀/32 to λ₀/64 its maximum falls by a quarter; at t = 900 it is 0.170, 0.121, 0.136 %, not monotone. At λ₀/64, where PIConGPU's launch term is small, it is visibly Smilei's alone from t = 500: S–P is within 0.003 % of PM–S at every output from there, and PM–P is 4 to 14 times smaller. "From t = 700" was the λ₀/48 picture, where PIConGPU's launch term hid it before. The plasma amplitude's relative rms has the same course for PM–S (largest at t = 800 to 900: 0.037, 0.028, 0.029 %) and at λ₀/64 the same assignment (S–P 0.029 %, PM–P 0.004 % at t = 900).

It is not explained. Two things it is not: a discretization difference (it does not fall with the grid), and an effect of Smilei's heavier bunch in proportion (the bunch excess falls from 2.0 % to 0.6 % of the bunch, §5.3, while the wake difference falls by a quarter). Not looked at: where along the lineout the difference sits, and Smilei's plasma load against ParticleMesh's particle by particle.

### 5.3 Smilei's bunch

Smilei's bunch is heavier than ParticleMesh's on every grid. Over the common region, from the bunch rule and from the weight above 20 mₑc² (from t = 900, where the whole bunch is above it):

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| bunch, Smilei over ParticleMesh (%) | +2.05 at all six outputs | +1.02, and +1.37 at t = 900 | +0.61 at t = 800 and 900, +0.15 after |
| the same in macro-particle weights | 0.072 | 0.074, 0.100 | 0.079, 0.019 |
| weight above 20 mₑc² (%) | +2.05 | +1.02, and +1.37 at t = 1000 | +0.39 |
| the same in macro-particle weights | 0.072 | 0.074, 0.100 | 0.050 |
| weight above 10 mₑc² (%) | +2.04 to +2.49 | +1.53 to +2.12 | +0.50 to +0.73 |

The excess is a fraction of one macro-particle on every grid, 0.02 to 0.10 of a full-density weight, and between λ₀/32 and λ₀/48 the same fraction (0.072 and 0.074): there it follows the macro-particle weight, that is Δ². At λ₀/64 it is 0.05 to 0.08 of a weight by two of the three measures and 0.02 by the bunch rule after t = 950, where that rule cuts ParticleMesh's and Smilei's spectra 1.0 to 1.2 mₑc² apart. So the relative excess shrinks with the grid, from 2.0 % to 2.5 % at λ₀/32 to 0.15 % to 0.73 % at λ₀/64, and is not a difference in the trapped charge that stays; a difference of a fixed fraction of one particle's weight points at how single particles are weighted or placed when loaded. That reading is inferred from three numbers. The loads were not compared, and Smilei's rule at the ramp's foot, WP-053's candidate, is still untested.

ParticleMesh's bunch against PIConGPU's is 0.92 % and 0.56 % apart at λ₀/32, 0.14 % at λ₀/48 and 1.41 %, 0.08 %, 0.54 % and 0.09 % at λ₀/64, where the rule's cut moves between outputs in both codes; in macro-particle weights at most 0.03, 0.01 and 0.18.

### 5.4 The centroid distances to PIConGPU

They follow neither the grid nor the fitted launch term: 4.4, 1.5, 1.7 × 10⁻³ c/ω₀ for PM–P in vacuum and 4.0, 2.0, 2.4 × 10⁻³ for S–P, so λ₀/64, with the smallest term, does not have the smallest distance. The signed series differ in kind between the grids: PM–P runs from −0.7 to −4.4 and back to −1.3 × 10⁻³ at λ₀/32, from −1.5 through zero to +1.3 and +1.0 at λ₀/48, and from +0.7 to +1.7 and back to +0.7 at λ₀/64.

One mechanism was estimated and does not account for it: an amplitude ratio r(ξ) that varies along the pulse moves the centroid of E_y² by (τ²/4) · d ln r²/dξ. With the fitted oblique term and focus offset of §3.2 that is at most 0.7, 0.6 and 0.2 × 10⁻³ c/ω₀ on the three grids, a seventh, two fifths and a ninth of what is measured. The estimate is computed here from the fit's two shapes for a Gaussian envelope; it is not a fit to the centroid. What else PIConGPU's launch does to the pulse's timing (the model's response as a function of frequency was not computed) is open.

ParticleMesh–Smilei's centroid difference, 0.6 to 0.8 × 10⁻³ c/ω₀, has one course on the three grids: from +0.46, +0.14, +0.02 × 10⁻³ at t = 200 to −0.61, −0.77, −0.79 at t = 700 and −0.34, −0.49, −0.50 at the end. Its value at t = 200 falls with the grid at a ratio of 2.9, that of a second order; the excursion of about −0.8 × 10⁻³ after it does not fall.

### 5.5 Smilei's E_y at the back of its window

WP-053 left out the last laser wavelength of the shared columns because Smilei's E_y differs from both other codes' in two or three columns there. On every grid here the maps' shared columns begin exactly at Smilei's window back (x = 447.68 at t = 699.2 and 947.19 at t = 1198.7 on all three), and the columns are 0.785 c/ω₀ apart. The norm over all rows of the difference of two codes' E_y in one column, over the norm of ParticleMesh's column, computed here from the plasma runs' maps:

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| column at Smilei's back, t = 700, PM–S | 0.227 | 0.229 | 0.461 |
| the same, t = 1200 | 0.265 | 0.272 | 0.347 |
| next column, t = 700 and 1200, PM–S | 0.027, 0.070 | 0.019, 0.047 | 0.011, 0.040 |
| the four columns after it, PM–S | 0.013 to 0.048 | 0.013 to 0.038 | 0.010 to 0.039 |
| column at Smilei's back, both outputs, PM–P | 0.021, 0.042 | 0.008, 0.011 | 0.002, 0.004 |

On every grid one column is off, the one at Smilei's back, by a quarter to a half of the field there; it does not shrink with the grid (it is larger at λ₀/64); and the next column is at most 1.5 times the largest of the four after it at the same output. With the back kept, the far wing's PM–S number at t = 700 is 0.79, 0.80 and 4.17 % against 0.33, 0.28 and 0.24 % without, and the number over all rows 0.096, 0.084, 0.142 % against 0.070, 0.050, 0.046 %.

**The depth of the region is c times the time since Smilei's window last moved.** WP-053's maps at λ₀/48 have the same window back (the same lattice column) and are 0.8 and 1.4 time units later than these, and more columns are off in them: by the same measure 0.398, 0.276, then 0.021 at t = 700.1, and 0.384, 0.398, 0.200, then 0.041 at t = 1200.1. Smilei's window moves by a whole patch (4.71 c/ω₀) at the end of the iteration whose half-step time passes what it has moved (`smilei_window_back`, from its source): before these outputs at t = 698.9 to 699.0 and at t = 1198.5 (the iteration depends on the grid's Δt). A disturbance that starts at the new back at that moment and runs forward at c has reached

| output | time since the move | reach (c/ω₀) | columns within reach | columns off |
|---|---|---|---|---|
| t = 699.2, the three grids of this scan | 2, 2, 3 steps: 0.24 to 0.36 | 0.24 to 0.36 | 1 | 1 |
| t = 700.1, WP-053's set | 9 steps: 1.09 | 1.09 | 2 | 2 |
| t = 1198.7, the three grids | 1, 2, 2 steps: 0.18 to 0.24 | 0.18 to 0.24 | 1 | 1 |
| t = 1200.1, WP-053's set | 14 steps: 1.69 | 1.69 | 3 (the third at 1.57) | 3, the third at half the size |

Eight maps, no free parameter; the instant of a move is known to a step (0.12 c/ω₀ at λ₀/48), which matters only for the third column of the last row. This supports the cause WP-053 proposed, Smilei's boundary acting on the wake's field, in a more specific form: something is set off at Smilei's back each time its window jumps, and it travels into the window. It is a reading of eight maps and Smilei's moving rule; no Smilei run with another patch size or another boundary has tested it, what the boundary does there was not read in its source, and why the column's difference is larger at λ₀/64 is open. The window's next jump, 4.71 time units later, takes the back past everything the disturbance has reached, which would keep it within one patch of the back; that is an inference, and the margin of one laser wavelength (6.28 c/ω₀) covers a patch.

### 5.6 The far wing

Between ParticleMesh and Smilei the far wing (2.15 to 2.85 waists from the axis) is 0.049, 0.049, 0.050 % before the plasma (t = 300) on the three grids, and 0.33 to 0.34, 0.27 to 0.28 and 0.24 to 0.25 % in it (t = 700 and 1200). The value before the plasma does not depend on the grid at all; the value in the plasma falls slowly, by 28 % from λ₀/32 to λ₀/64, with no order. Against PIConGPU it is 0.96, 0.68, 0.20 % before the plasma, the launch term again (the oblique term acts most where the beam is widest), and at λ₀/64 falls to 0.12 % and 0.08 % in the plasma for PM–P while S–P stays at 0.26 %. So on the finest grid the far wing in plasma is, like the wake, Smilei against the other two.

## 6. Cost, as measured

| | λ₀/32 | λ₀/48 | λ₀/64 | What the time covers | Hardware |
|---|---|---|---|---|---|
| ParticleMesh vacuum / plasma | 16.5 s / 112.3 s | 35.4 s / 378.7 s | 69.2 s / 864.1 s | the stepping loop with its outputs (`run_seconds`), the diagnostics on host copies | RTX 4080 SUPER |
| PIConGPU vacuum / plasma | 37.5 s / 61.0 s | 61.2 s / 136.0 s | 93.8 s / 263.8 s | the `picongpu` run after compilation; the jobs with compilation took 105 / 127, 128 / 202 and 160 / 329 s | RTX 4080 SUPER |
| Smilei vacuum / plasma | 16.7 s / 115.3 s | 45.8 s / 385.4 s | 113.0 s / 916.6 s | the `smilei` process; the box's steps 17 / 115, 46 / 386 and 113 / 917 s | AMD EPYC 9655, 16 OpenMP threads |

Single runs; the jobs' times with compilation are as the coordinating session reported them, the others from the runs' summaries and manifests. No performance conclusion is drawn: ParticleMesh ran in Reference mode, which sorts every step, in Float64, with no optimization round; the codes ran on different hardware with different diagnostics; A5's performance comparison is not this.

## 7. For A7: what a threshold on this case could rest on

A7 on `main` has no LWFA threshold. This section lists what the three grids offer one, with what each choice passes and fails. It proposes none and accepts none.

**What the scan fixes first.** A threshold on a code-to-code distance and a threshold on accuracy are different things on this case, by two to three orders of magnitude in the laser: at λ₀/48 the three codes agree with each other in the carrier phase to 0.05 to 0.7 mrad while each is 56 mrad from its own λ₀/64 run and, by extrapolation, 129 mrad from the limit. A code-to-code bound says that ParticleMesh solves the same discrete problem as another Yee code with the same launch. It says nothing about whether λ₀/48 is fine enough.

**Option 1: fixed bounds against Smilei, the pair whose launches are matched.** For illustration, bounds at 1.6 to 2.5 times the largest PM–S value over the three grids, rounded: carrier phase 0.2 mrad, centroid 0.0015 c/ω₀, vacuum amplitude 0.01 %, plasma amplitude 0.06 %, wake E_x 0.3 %, wake extrema 0.2 %, maps over all rows 0.12 % and flank 0.2 %, highest bin within one bin: twelve rows with the vacuum and plasma ones counted apart.

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| PM–S | passes 12 of 12, at most 0.63 of a bound | passes 12, at most 0.54 | passes 12, at most 0.58 |
| PM–P | fails 10: every row but the wake minimum and the bin (up to 11 times the bound) | fails 5: vacuum phase 3.5, vacuum amplitude 6.6, plasma phase 2.75, both centroids 1.03 and 1.04 | fails 3: vacuum amplitude 1.6, both centroids 1.10 and 1.16; vacuum phase passes at 0.996 |
| S–P | fails 10 | fails 5 | fails 3 |

Such bounds do not depend on the grid, because the PM–S distances do not (§5.1), and they rest on one reference code whose remaining differences from ParticleMesh are unexplained in four places (§8). They would not admit PIConGPU as a second reference on any grid as its launch stands. The extremum rows need their factor of 1.5 (§3.1).

**Option 2: a bound relative to the two reference codes' own distance on the same grid.** ParticleMesh's distance to each reference no larger than f times Smilei–PIConGPU's, row by row. The largest ratio over the rows of §5 without the far wing is 1.16 at λ₀/32 (plasma centroid, PM–P), 1.33 at λ₀/48 (wake maximum, PM–S, an extremum row) and 1.25 at λ₀/64 (plasma phase, PM–P); the far wing is at most 1.03. With f = 1.5 every row passes on every grid; with f = 1 three rows fail at λ₀/32, seven at λ₀/48 and three at λ₀/64. The bar is not a fixed quantity: at λ₀/32 and λ₀/48 it is PIConGPU's launch term (PM–S passes at 0.04 to 0.5 of it in the laser rows), at λ₀/64 it is Smilei's unexplained difference in the plasma rows (PM–S is 0.90 to 0.99 of S–P in the plasma amplitude, the wake and the maps there). It says that ParticleMesh is no outlier among three, with no absolute scale.

**Option 3: a bound across grids against the scheme's dispersion relation, in vacuum.** The measured phase and centroid differences between two grids over the axial Yee prediction are within 2.5 % of one for ParticleMesh at all twelve outputs and both grid pairs (§4.1). A bound of 5 % passes ParticleMesh and Smilei on both pairs and fails PIConGPU at its first outputs (6 % in the phase, 18 % in the centroid). This is the only option here that compares with a closed form instead of another code, and the only one that is about accuracy; it needs two grids per test, covers the laser in vacuum and not the wake or the electrons, and its 2.5 % includes what a plane-wave prediction leaves out of a focused beam. An order criterion in plasma would have the carrier phase (1.99 to 2.01) and the centroid (1.91 to 2.07) and no more: the wake's orders run from 0.8 to 3.6.

**Option 4: no threshold on the electrons from this case, or one in macro-particle weights.** The bunch is 3.5 to 12.9 macro-particles, its weight moves by 8 % to 11 % between λ₀/32 and λ₀/64, and at λ₀/64 the rule that defines it is unstable where the gap is narrow. In percent, a bound of 2.5 % passes PM–S on all three grids and PM–P too, and fails S–P at λ₀/32 (3.0 %); in macro-particle weights, 0.3 of a weight passes every pair on every grid (largest 0.26, S–P at λ₀/64, where the rule's cut differs), and one weight is 29 %, 14 % and 7.8 % of the bunch on the three grids. The highest occupied bin is the same in the three codes at every plasma output on λ₀/48 and λ₀/64 and one bin apart once on λ₀/32.

**What none of them can rest on yet.** A PIConGPU column in a fixed bound needs its launch term removed or tested by a run; until then PM–P is small only where the term happens to be (λ₀/64). The plasma rows against Smilei rest on a difference that is Smilei's alone on the finest grid and has no explanation, so a bound set from PM–S in the wake (0.14 % to 0.18 %) is set by that difference, while the pair without it agrees to 0.02 %. And three grids support an order only as one equation: a bound that uses an extrapolated limit would want a fourth.

## 8. Verification status and open issues

- **Not independently reviewed.** A review follows this commit. One run per code and grid; nothing here is a validation claim, and nothing is an A7 pass.
- **Measured.** Every number in the tables of §3 to §6, from the keys named there.
- **Inferred, and said so where it stands.** That the laser's change with the grid is the Yee dispersion relation's (a plane-wave formula matched to 2.5 %); the λ₀/64 run's distance from the continuum (formula and three-point extrapolation, which agree and are not independent); that the wake lineout's change is mostly the driver's lag (an estimate for a sinusoid); that Smilei's bunch excess is a load granularity (three numbers); that an extremum row is known to a factor of 1.5 (two λ₀/48 sets with outputs at different instants); that the region at Smilei's back is set off by its window's moves (§5.5).
- **Confirmed by a prediction.** The 1D model of PIConGPU's launch, on the two grids it had not been built on, in the amplitude and in the phase (§3.2). Still no PIConGPU run tests it, and PIConGPU's launch is not corrected.
- **Narrowed, not tested by a run.** Smilei's E_y at the back of its window: one column at its back on every grid, and a depth that follows the time since its window moved in eight maps (§5.5).
- **Unexplained**, each with its numbers above:
  - ParticleMesh against Smilei in vacuum, independent of the grid: 0.004 % in amplitude, a centroid excursion of −0.8 × 10⁻³ c/ω₀, and fitted focus offsets that differ between amplitude and phase (§3.2, §5.4).
  - Smilei's wake and plasma-amplitude difference from both other codes, on every grid (§5.2).
  - Smilei's bunch excess of 0.02 to 0.10 macro-particle weights (§5.3).
  - The far wing against Smilei in plasma, 0.24 % to 0.34 % (§5.6).
  - The centroid distances to PIConGPU (§5.4).
  - Why the wake's maximum is furthest from converged at t = 700 and 800 in all three codes (§4.2).
- **Not shown.**
  - ParticleMesh on the CPU on these grids, in Float32, or in its Fast mode.
  - The engine on `main`: ParticleMesh ran the branch's code of `1e36a43`, which does not have what `main` merged after the branch point (#410, #411).
  - A fourth grid: every order is one equation; and Δy, Δt and the particles per cell were not varied apart from Δx, so the macro-particle weight and the cell size change together.
  - PIConGPU's spectrum over the common region (#412).
  - A repeat of any run; Smilei's and PIConGPU's determinism is assumed.
  - Where along the lineout Smilei's wake difference sits, and a particle-by-particle comparison of the loads.
  - A Smilei run with another patch size or boundary, which would test §5.5.
- **The tests.** `scripts/lwfa/test/runtests.jl`, 1455 assertions, among them T-714 (the three grids are one case), T-715 (the measures across grids), T-716 (PIConGPU's launch term per grid), T-717 (`compare_scan` end to end on synthetic runs with planted orders, and what `scan_distances` counts) and T-718 (`launch_fit`). They test the generators, the readers and the measures on synthetic data, not the physics of the case. Six defects planted in `scan_distances` at `ae844e7`, one at a time: the plasma rows counted from t = 150, the bunch counted from the plasma's start, the plasma's start taken as the laser's delay, two map bands swapped, the guard against a negative limit removed, and the highest bin's difference made a sum. Each made T-717 fail, and the sources were restored.
