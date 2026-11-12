# LWFA 2D on three grids: A5's first case at λ₀/32, λ₀/48 and λ₀/64 in ParticleMesh, Smilei and PIConGPU

**Summary.**
- The scan: A5's first 2D laser-wakefield case on Δx = λ₀/32, λ₀/48 and λ₀/64 (Δy = 4Δx, Δt ∝ Δx, the outputs at the same times) in ParticleMesh (CUDA), Smilei v5.1 and PIConGPU 0.8.0: 18 runs, compared on the lattice points the three grids share.
- Converges, in each code: the laser, at second order. Carrier phase and centroid differences between grids have the order 2.01 at all twelve vacuum outputs in ParticleMesh and in Smilei (1.74 to 2.58 in PIConGPU), and in those two they are the axial Yee dispersion relation's to 2.5 % (PIConGPU's are 6 % and 18 % off it at its first outputs). At the end λ₀/48 is 56 mrad and 0.17 c/ω₀ from λ₀/64, and λ₀/64 by extrapolation another 72 mrad and 0.22 c/ω₀ from the limit.
- Converges without a constant order: the wake. Its E_x at λ₀/48 is 0.5 % to 1.4 % (relative rms over 1.5 λ_p) from λ₀/64, its maximum 0.3 % to 2.0 %. From t = 700 a displacement of the wake by about two thirds of the laser's lag, concentrated at the back of the bucket, accounts for 43 % to 89 % of the squared difference; at t = 400 and 500 it is not a displacement. The trapped bunch, about 140, 260 and 470 macro-particles on the three grids, is 8 % to 11 % heavier at λ₀/32 than at λ₀/64 and within 1.3 % at λ₀/48.
- By each row's largest value over the run, a code at λ₀/48 is 81 to 1095 times further from its own λ₀/64 run than from another code in the vacuum carrier phase, 84 to 221 times in the centroid and 9.0 to 11.5 times in the wake's E_x (at single outputs at least 19, 6.7 and 5.5 times): the three share the scheme's error, and a code-to-code distance measures whether two implementations solve one discrete problem, not how accurately.
- Not shown to converge: ParticleMesh against Smilei. Only their vacuum carrier phase falls at an order (third, at all twelve outputs: 0.118, 0.052, 0.036 mrad). The largest values of the wake's E_x (0.182, 0.154, 0.136 %), the amplitudes and the maps of E_y fall by 7 % to 34 % from λ₀/32 to λ₀/64, slower than a first order towards zero would; the centroid and the wake's extrema do not fall. That rules out a second-order difference and a first-order one towards zero; three grids do not separate a grid-independent part from a first-order one, and a fourth grid for the two plasma runs would. What the difference is, is not explained.
- The vacuum amplitude and phase distances to PIConGPU are its launch term, which a 1D model built on the λ₀/48 runs gave for the other two grids before they ran: 1.75 and 0.18 of the oblique term predicted at λ₀/32 and λ₀/64, 1.73 and 0.175 fitted (0.83 and 0.82 at λ₀/48); λ₀/32 confirms the model's layout sharply, λ₀/64 only that the term is small. The wake and the maps fall with the term, untested whether by it; the centroid distances (0.0015 to 0.0046 c/ω₀) and the bunch follow neither the grid nor the term. At λ₀/64 ParticleMesh and PIConGPU are 0.016 % (vacuum amplitude), 0.20 mrad, 0.022 % (wake E_x) and 0.020 % (maps) apart.
- What WP-053 left open: Smilei's wake difference is on every grid (0.11 % to 0.18 % at t = 700 to 900) and at λ₀/64 it is Smilei against two codes whose wakes agree to 0.02 %; Smilei's bunch is heavier by 2.0 %, 1.4 % and 0.6 %, which is one to four of the bunch's own macro-particles; its E_y at the back of its window is off by 23 % to 46 % in the first kept column on every grid, to a depth that in twelve maps is consistent with c times the time since Smilei's window last moved; the far wing against Smilei falls slowly (0.34, 0.28, 0.24 %). The first two and the last are not explained.
- For a threshold on this case (§7, four options, none proposed): fixed bounds at 1.6 to 2.5 times the ParticleMesh–Smilei distances pass that pair on all three grids and fail ParticleMesh–PIConGPU in 10, 5 and 3 of 12 rows; a bound relative to Smilei–PIConGPU passes at a factor 1.5 and fails at 1; neither bounds the error of the solution, which only a bound across grids does.
- Not shown: ParticleMesh on the CPU (CUDA only here), Float32 and its Fast mode; a fourth grid, so every order is one equation with no check; Δy, Δt or the particles per cell varied alone; a PIConGPU run that tests the model of its launch; PIConGPU's spectrum over the common region; a repeat of any run. This is not an A7 pass.

| | |
|---|---|
| Work package | WP-054 (`trace/wp/WP-054-lwfa-resolution-scan.md`); goals A5, A7. Follows WP-053 (`reports/lwfa-2d-a5.md`, PR #413) |
| Cases | `scripts/lwfa/cases/lwfa-2d-r32.toml` (sha256 `9be9531b…`), `lwfa-2d-r48.toml` (`51b1d5fc…`), `lwfa-2d-r64.toml` (`0b0dc953…`); each run's manifest has the hash of its copy |
| Commits | ParticleMesh ran the code of `1e36a43` (its six manifests, `dirty = false`). Smilei's and PIConGPU's inputs were generated at `a991f96` (their manifests); `1e36a43` differs from it in the comparison code and its tests only (`git diff --stat a991f96 1e36a43`: `scripts/lwfa/lwfa.jl`, `scripts/lwfa/test/runtests.jl`). The comparison ran at `16d943d` (`scan/manifest.toml`, `dirty = false`); the per-grid `comparison.json` files are byte for byte those of the reviewed commit `fb2015d`. These are commits of the branch `worktree-wp-054-lwfa-resolution-scan`; a squash merge does not keep them reachable from `main` |
| Runs | ParticleMesh: six runs on CUDA, `lwfa-scan-r<N>/pm-vacuum` and `pm-plasma`. PIConGPU: six gpu-run jobs, `scan-r<N>-pic-vacuum` and `-plasma` (the job names as the coordinating session gave them; each run's start and end are in its `container.txt`). Smilei: one CPU box request, `runs/lwfa-scan-smilei` (request `406a48b`, result `ba94009`), six runs; its summaries and maps stay on that branch. The summaries of ParticleMesh and PIConGPU are on no branch and are in `wp-054-lwfa-resolution-scan/runs/`, with the maps of their plasma runs, which the off-axis rows read |
| Input hashes | each run's `manifest.toml` as `scan/r<N>/manifest-<run>.toml`; the box's `scan/box-status-smilei.toml`; the sha256 of the 18 summaries in `scan/manifest.toml`, and per grid of the summaries and of the three plasma runs' `maps.h5` in `scan/r<N>/manifest.toml` |
| What the counted rows read | from each `summary.json` the lineouts `Ey_axis` and `Ex_axis` with their offsets and the `spectrum` with its region; from each plasma run's `maps.h5` the maps of E_y. The committed summaries and maps also hold `Ey_envelope`, `rho_axis`, `spectrum_window`, `spectrum_plugin` and the maps' `rho`, which no counted row needs (`compare` computes the envelope again from `Ey_axis` and reads `rho_axis` only for a row kept for the record); that is about 18 of the 40 MB by the review's count, kept because the manifests' hashes are of the files as the runs wrote them |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode, CUDA on an RTX 4080 SUPER. Smilei v5.1 (image `particlemesh-smilei:e4f0148b8952fc87`), 16 OpenMP threads on an AMD EPYC 9655. PIConGPU 0.8.0 (`6bb3470e`), CUDA, Float64, the same RTX 4080 SUPER |
| Figures | `wp-054-lwfa-resolution-scan/scan/scan_self.png` and `scan_distances.png`, and per grid `scan/r<N>/pair_distances.png` and `spectrum.png`, each with its JSON sidecar. The measured values below are keys of `scan/scan.json` (`self`, `distances`, `launch`, `yee`, `back_columns`, `bunch_bins`), of `scan/r<N>/comparison.json` and of `scan/back_columns_wp053.json`; ranges, largest values, counts and ratios are computed from those keys |
| Status | measured; reviewed independently once, a second pass follows this commit |
| Review | One independent review by a fresh numerics-reviewer subagent (claude-fable-5-2, 2026-11-12) on `fb2015d`; its findings are in `wp-054-lwfa-resolution-scan/review.md` and are applied in this text. It reproduced every table and all of `scan.json`, confirmed from the commit stamps that the launch prediction preceded the runs, and found eight claims that were wrong or stronger than the data and tests that let wrong definitions pass (§8). A short second pass follows |

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

**Between two codes on one grid**: `compare`'s, unchanged from `reports/lwfa-2d-a5.md` §2 and run per grid into `scan/r<N>/comparison.json`. A pair is written a–b (PM–S ParticleMesh–Smilei, PM–P ParticleMesh–PIConGPU, S–P Smilei–PIConGPU); the wake's window is 1.5 λ_p behind ParticleMesh's envelope peak; the maps leave out one laser wavelength at the back of the shared columns. A run's outputs count after `scan.json`'s `from`: t = 150 in vacuum (12 outputs, t = 200 to 1300); in the plasma runs t = 364, when the pulse's peak reaches the plasma (10 outputs, t = 400 to 1300, the ones WP-053 counts; before it a plasma run repeats its vacuum run); t = 750 for the bunch (6 outputs), from where the spectrum's highest group stands clear of the rest. `scan.json` has under `distances`, per pair, observable and grid (`scan_distances`): the largest magnitude over those outputs (`max`) and the output it is at (`t_max`); the three grids' values at each output with the order of those three (`per_output`); and how many outputs have an order, with the ranges of the orders and of the limits (`orders`). The three largest values get no order of their own: they are at different outputs on different grids.

The window's anchor, ParticleMesh's envelope peak, snaps to a crest of the carrier, so it moves in steps of half a laser wavelength, and at t = 800 it is one such step further forward at λ₀/64 (x = 690.27) than at λ₀/32 and λ₀/48 (686.83, 686.96). `wake_anchor_sensitivity` in each `comparison.json` has the wake rows with the window half a wavelength either way: the PM–S run maxima are 0.178 to 0.187, 0.144 to 0.170 and 0.131 to 0.146 % over the three placements, so a wake row is known to about ±10 %.

**One code across the grids** (`scan_self`, T-715, T-717): the primal nodes of the three lattices coincide every λ₀/16, so
- E_y on the axis, its analytic signal, the centroid of E_y² and the carrier phase against the finest grid are taken on the shared nodes, with nothing interpolated;
- E_x, which is on dual points that no two grids share, is brought to the primal nodes by the four-point midpoint formula (error 3/128 · Δx⁴ ∂⁴E_x) and compared against the finest grid over the 1.5 λ_p behind the finest grid's centroid; the wake's minimum and maximum are taken on each grid's own points, refined by a parabola, and only where each grid has an extremum inside the window (a lineout still rising at the window's edge has its largest sample there, at a different place on every grid);
- `wake_Ex_displacement`: the finest grid's lineout displaced rigidly by s, by the cubic through its four nearest points, against the coarser grid's, for s from −1.5 to 1.5 c/ω₀ in steps of 0.01: the best s, what it leaves, what a displacement by the centroid lag leaves, and the part of the squared difference at the steepest tenth of the points (`wake_displacement`);
- the axis lineout of E_y is the mean of the two rows at ∓Δy/2, a different place on every grid. In vacuum the beam's profile gives what that does to the amplitude, exp(−(Δy/2)²/w(ξ)²), and `amplitude_on_axis` has it divided out (0.02 %, 0.009 % and 0.005 % at focus on the three grids); in plasma it is left in.

**An order** (`observed_order`): three values f₃₂, f₄₈, f₆₄ with f = f∞ + C·Δᵖ have (f₃₂ − f₄₈)/(f₄₈ − f₆₄) = 2.00 for p = 1, 2.86 for p = 2, 4.11 for p = 3. An order and the extrapolated f∞ are given only when the two differences have one sign, exceed a floor and the ratio is that of an order between 0.5 and 4; a distance between codes, being a magnitude, also has none when it extrapolates below zero by more than half its finest value (`distance_order`). Three grids give one equation for p and no check of it: an order below says that three values are compatible with one, and what it is worth is its constancy over the outputs.

**A launch difference** (`launch_fit`, T-718): from a pair's vacuum rows, the multiple c of the oblique term (a launch that multiplies the transverse spectrum by 1 − c·k_y²/4) and a focus offset δ, each fitted once from the amplitudes and once from the phases (`reports/lwfa-2d-a5.md` §3.3 has the shapes); and `centroid_shift`, what those two terms would do to the centroid of E_y² (§5.4).

**Also in `scan.json`**: `yee`, the axial Yee wavenumber and group velocity of each grid, and in each vacuum row of `self` the measured differences over what they predict (§4.1); `back_columns`, the maps column by column from the back of the shared ones, with Smilei's last window move (§5.5; `scan/back_columns_wp053.json` is the same for WP-053's final runs, written by `lwfa.jl back-columns`); `bunch_bins`, the bunch bin by bin at the last output (§4.3).

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

Reproduced to 0.5 % of the value or better: every laser row but one; to 1 %, the maps over all rows and the flank; to 3 %, the far wing. The wake's E_x agrees to 1 % in its run maximum and to 5 % per output. Not reproduced, in bold, all in pairs with Smilei: the plasma amplitude's relative rms for PM–S (11 % lower here), the wake minimum for PM–S (13 % higher) and the wake maximum for PM–S and S–P (48 % and 36 % higher). Nothing but the output instants differs for Smilei and PIConGPU between the two sets, so a run maximum of an extremum's ratio moves by up to half its value when the outputs are taken 0.4 to 1.6 time units earlier: those rows are known to a factor of 1.5 at best, not to a digit. The highest occupied bin is the same in the three codes at all ten plasma outputs here (in WP-053's set PIConGPU's was one bin lower at t = 1100).

Because ParticleMesh's side changed between the sets, the table also says that the engine of `1e36a43` on CUDA and that of `6ead5bd` on the CPU give this case to the differences above. It is no test of either beyond that, since no output is at the same step.

### 3.2 PIConGPU's launch term, predicted per grid

WP-053 modelled PIConGPU's launch in 1D: B = k̂ × E on its Huygens surface, with its default 12-cell exponential absorber behind. Surface and absorber are counted in cells, so the model gives a different term on each grid, and T-716 fixed its three values before these runs (commit `a991f96`). Fitted from the runs (`scan.json`, `launch`; 12 vacuum outputs each; from the amplitude / from the phase):

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| oblique term, model (T-716) | 1.748 | 0.832 | 0.183 |
| fitted, PM–P | 1.727 / 1.735 | 0.818 / 0.820 | 0.175 / 0.175 |
| fitted, S–P | 1.743 / 1.741 | 0.824 / 0.823 | 0.177 / 0.177 |
| the model with its surface a cell nearer the edge, a cell further | 1.843, 1.540 | 0.625, 1.051 | 0.160, 0.238 |
| with its absorber a cell shorter or longer, a fifth weaker or stronger | 1.72 to 1.77 | 0.83 to 0.84 | 0.154 to 0.211 |
| focus further out (c/ω₀), model | −0.149 | +0.445 | +0.124 |
| fitted, PM–P | −0.086 / −0.110 | +0.464 / +0.462 | +0.151 / +0.143 |
| fitted, S–P | −0.141 / −0.125 | +0.422 / +0.444 | +0.113 / +0.125 |
| fitted, PM–S: oblique term | −0.016 / −0.006 | −0.006 / −0.003 | −0.002 / −0.002 |
| fitted, PM–S: focus | +0.054 / +0.015 | +0.042 / +0.018 | +0.038 / +0.018 |

The λ₀/48 values were known when the model was built (WP-053 fitted 0.82 and a focus 0.42 to 0.47 further out there); the predictions are those for λ₀/32 and λ₀/64, runs the model had not seen, and both hold: the oblique term to 0.02 of its unit, with the sign change of the focus offset between λ₀/32 and λ₀/48. The fits leave at most 0.002 % in amplitude and 0.003 mrad in phase.

How sharp that is differs between the two grids (the fourth and fifth rows, all in T-716). At λ₀/32 the fitted 1.73 to 1.74 is the model's 1.748 and neither neighbour's 1.84 or 1.54: it confirms the model's layout, surface on the seventeenth cell and twelve absorber cells, sharply. At λ₀/64 the fitted 0.175 to 0.177 lies inside what a cell of the absorber or a fifth of its strength does to the model (0.154 to 0.211) and beside what a cell of the surface does (0.160, 0.238): it confirms that the term is small there, not its value. The focus is predicted to 0.07 c/ω₀, which at λ₀/32 is up to 42 % of the predicted −0.149 (−0.086 fitted from the amplitudes of PM–P), and the fit's own floor for a focus is 0.04 to 0.05: that is the "focus" it finds from the amplitudes of ParticleMesh and Smilei, which launch none.

This is a model confirmed on two more grids; it is still not a PIConGPU run with another absorber or another incident B, and PIConGPU's launch is not corrected.

Between ParticleMesh and Smilei the fit finds no oblique term on any grid; a focus offset from the amplitude that falls slowly with the grid (0.054, 0.042, 0.038 c/ω₀) and one from the phase that does not (0.015, 0.018, 0.018), so the two disagree and it is not a focus offset; and a phase constant of −0.098, −0.029, −0.013 mrad, whose differences have the ratio of a third order.

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
| wake maximum, 9 outputs from t = 500: λ₀/32 (%) | 1.66 to 5.91 | 1.66 to 5.93 | 1.69 to 6.06 |
| λ₀/48 (%) | 0.31 to 1.96 | 0.34 to 1.84 | 0.32 to 1.94 |
| order | 9 of 9: 0.91 to 3.60 | 9 of 9: 0.77 to 3.37 | 9 of 9: 0.96 to 3.57 |
| places of the two extrema, 19 values (c/ω₀) | 0.01 to 0.91, 1 order | 0.007 to 1.01, 1 order | 0.01 to 0.75, none |
| **electrons, 6 outputs from t = 800** | | | |
| bunch weight: λ₀/32 (%) | 8.0 to 8.7 | 9.6 to 10.7 | 8.5 to 9.2 |
| λ₀/48 (%) | 0.26 to 0.36 | 0.14 to 1.23 | 0.04 to 1.29 |
| order | none | none | 1 of 6 |

At t = 400 the wake has no maximum inside the window: E_x still rises towards the anchor, so the largest sample is the window's edge, a different place on every grid. That output has no wake maximum in the table.

### 4.1 The laser: second order, and the Yee dispersion relation's

The carrier phase and the centroid have the order 2.01 at every vacuum output in ParticleMesh and in Smilei, and 1.99 to 2.02 and 1.91 to 2.07 in plasma. That is as constant as an order from three values can be, and the size is the scheme's own: on the axis the Yee grid carries the wave with k = (2/Δx)·asin((Δx/Δt)·sin(Δt/2)) and the group velocity cos(kΔx/2)/cos(Δt/2), which gives k − 1 = 2.429, 1.077, 0.605 × 10⁻⁴ and 1 − v_g = 7.30, 3.23, 1.82 × 10⁻⁴ on the three grids (`scan.json`, `yee`; T-715 checks the two formulas against the dispersion relation and its derivative). The measured phase difference from the finest grid over (k_N − k₆₄)·x_c, and the measured centroid difference over (v_g,N − v_g,₆₄)·x_c/v_g,₆₄, with x_c the finest grid's centroid, at the twelve vacuum outputs and both coarser grids (the `yee` key of each vacuum row):

| measured over predicted | ParticleMesh | Smilei | PIConGPU |
|---|---|---|---|
| carrier phase | 0.975 to 1.003 | 0.976 to 1.003 | 0.938 to 1.007 |
| centroid | 0.980 to 1.000 | 0.988 to 1.000 | 0.817 to 1.002 |

At the last output ParticleMesh's phase differences are 218.4 and 56.5 mrad against 217.8 and 56.3 predicted, its centroid differences −0.6550 and −0.1693 c/ω₀ against −0.6549 and −0.1693. The prediction is for a plane wave on the axis; a focused beam's oblique components and the axis lineout's rows are not in it, and the ratios are furthest from one at the first outputs, where the differences are smallest. PIConGPU's are further off there because its launch term, which differs between its grids (§3.2), is a larger part of a small difference.

What the finest grid still lacks follows from the same two numbers: by the formula the λ₀/64 run is 72.3 mrad and 0.217 c/ω₀ behind the continuum at the last output (`yee.finest`), and the extrapolation from the three runs gives 72.3 mrad and 0.216 c/ω₀ (ParticleMesh; Smilei's limits differ from ParticleMesh's by 0.002 mrad and 0.0004 c/ω₀). The two agree because both are the second-order term; neither is a measurement on a finer grid.

The amplitude on the axis is second order from t = 500 (1.97 to 2.03 in ParticleMesh and Smilei) and has other ratios before, where the difference from the finest grid is 0.002 % to 0.025 % and smallest at t = 400 (orders 1.81 to 2.31 at those three outputs).

### 4.2 The wake: converging, no constant order

The wake's E_x lineout at λ₀/48 is 0.48 % to 1.41 % from the finest grid's, in every code, with orders 1.3 to 2.0 at nine of ten outputs. The coarser grid's pulse is behind the finest grid's by the centroid differences of the table, and a wake that rode with its driver would differ by that displacement. Tested, it is not that simple (`wake_Ex_displacement`; ParticleMesh, the lag and the best displacement in c/ω₀, the difference before and after it in percent, and the part of the squared difference it removes):

| t | λ₀/32: lag, best | before, after | removed | λ₀/48: lag, best | before, after | removed |
|---|---|---|---|---|---|---|
| 400 | −0.160, +0.07 | 1.73, 1.64 | 10 % | −0.041, +0.01 | 0.48, 0.46 | 6 % |
| 500 | −0.215, −0.09 | 1.82, 1.46 | 36 % | −0.056, −0.01 | 0.73, 0.70 | 8 % |
| 600 | −0.261, −0.13 | 2.62, 1.68 | 59 % | −0.067, −0.03 | 0.80, 0.64 | 36 % |
| 700 | −0.306, −0.20 | 4.21, 2.85 | 54 % | −0.079, −0.06 | 1.30, 0.98 | 44 % |
| 800 | −0.356, −0.22 | 4.68, 3.25 | 52 % | −0.094, −0.07 | 1.41, 1.07 | 43 % |
| 900 | −0.409, −0.23 | 4.09, 2.29 | 69 % | −0.105, −0.07 | 1.17, 0.80 | 53 % |
| 1000 | −0.463, −0.28 | 4.09, 1.98 | 76 % | −0.122, −0.08 | 1.16, 0.69 | 65 % |
| 1100 | −0.516, −0.31 | 4.35, 1.84 | 82 % | −0.131, −0.08 | 1.20, 0.60 | 75 % |
| 1200 | −0.569, −0.36 | 4.71, 1.76 | 86 % | −0.145, −0.09 | 1.24, 0.54 | 81 % |
| 1300 | −0.620, −0.41 | 5.16, 1.67 | 89 % | −0.159, −0.10 | 1.36, 0.49 | 87 % |

- Displacing the finest lineout by the full centroid lag makes the difference larger at t = 400 and 500 on both grids (1.73 to 2.59 % and 1.82 to 2.33 % at λ₀/32) and from t = 700 removes 35 % to 64 % of the squared difference at λ₀/32 and 29 % to 61 % at λ₀/48: the wake does not simply ride with its driver.
- From t = 700 the best rigid displacement is 0.56 to 0.76 of the lag, about two thirds, and removes 43 % to 89 % of the squared difference, growing through the run; it leaves 0.49 % to 1.07 % at λ₀/48 and 1.67 % to 3.25 % at λ₀/32.
- From t = 700, 57 % to 70 % of the squared difference is at the steepest tenth of the points (`steep_share`), which in this wake is the jump at the back of the bucket (the review, which split the window at λ₀/48, has 81 % to 93 % of it more than 40 c/ω₀ behind the anchor; that split is in its files, not in `scan.json`): the displacement is concentrated there.
- At t = 400 and 500 it is not a displacement: the best one has the other sign at t = 400 and removes 6 % to 36 %.
- Smilei gives the same numbers to 0.01 c/ω₀ and one point of percent, PIConGPU to 0.02 and eight points.

The extrema, which do not depend on the wake's place, converge without one order: the maximum at λ₀/48 is 0.3 % to 2.0 % below the finest grid's and the minimum 0.02 % to 0.8 % shallower, with orders from 0.8 to 3.6 over the outputs. The maximum is furthest from the finest grid's at t = 700 and 800 (1.2 % and 2.0 % low at λ₀/48, 4.7 % and 5.9 % at λ₀/32). The three codes show the same pattern at the same outputs, so it is the discretized problem's and not a code's.

### 4.3 The electrons

The bunch at λ₀/32 is 8 % to 11 % heavier than at λ₀/64 and at λ₀/48 within 1.3 % of it, in every code, with no order.

The bunch is 3.5, 7.3 and 12.9 full-density macro-particle weights on the three grids, but its particles are not full-density ones (`scan.json`, `bunch_bins`: the bunch bin by bin at the last output). Apart from a few bins of 0.0001 to 0.016, the lightest bins weigh 0.021 to 0.031 of a full-density weight, 9 to 20 of them per code and grid; read as holding one particle each (the summaries have histograms, not particles), they say that the trapped particles were loaded low on the density ramp. At the mean of those bins, 0.025 to 0.028, the bunch is about 140, 260 and 470 macro-particles (132 to 140, 261 to 264 and 461 to 484 over the three codes), in 32, 35 and 50 occupied bins for ParticleMesh, and one of them is 0.7 %, 0.4 % and 0.2 % of it. So the 8 % to 11 % between λ₀/32 and λ₀/64 is ten and more of the coarsest grid's bunch particles, not the granularity of a few: the trapped charge depends on the grid, and at these three values not as a power of Δ. (The summary of `reports/lwfa-2d-a5.md` has "the bunch is seven macro-particle weights", which is its weight and not a count; its §4.4 has "at least about 35 macro-particles, most of them light", and by the bins here it is about 260 at λ₀/48.)

At λ₀/64 the rule that defines the bunch (the occupied bins above the widest empty gap) cuts at different places in the three codes at t = 800 and 900 (from 14.8, 13.2 and 18.4 mₑc² at t = 800, where the widest gaps are 4.0, 2.2 and 0.2 wide): the weights are 1.2503, 1.2579 and 1.2329 × 10⁻³ at t = 800 against 1.2426, 1.2444 and 1.2437 × 10⁻³ at the end. The highest occupied bin at the end is 65.8, 67.2 and 67.6 mₑc² on the three grids, in all three codes: it rises by 1.4 and then 0.4 with the grid, which is a second-order ratio as far as bins of 0.2 can say.

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

The three values of a row are each grid's largest and need not be at one output (the vacuum amplitude's for PM–S are at t = 800, 800 and 700). The wake rows are known to ±10 % from the anchor's step (§2), the extremum rows to a factor of 1.5 at best (§3.1). PIConGPU's spectrum covers its whole simulated domain and the other two a common region, as in WP-053, so its bunch rows compare different regions; at λ₀/64 the bunch rows of every pair also depend on where the rule cuts (§4.3).

**Orders, output by output** (`distances`, `per_output` and `orders`). For ParticleMesh against Smilei, how many outputs have an order by the guard of §2, the orders, and the fall from λ₀/32 to λ₀/64 at the single outputs:

| PM–S | outputs with an order | orders | from λ₀/32 to λ₀/64, output by output |
|---|---|---|---|
| vacuum carrier phase | 12 of 12 | 2.99 to 3.04, towards 0.004 to 0.024 mrad | falls at 12 of 12, by 70 % to 84 % |
| vacuum centroid | 3 of 12 (t = 400 to 600) | 2.24, 2.71, 3.25, rising | grows at 10 of 12 |
| vacuum amplitude | 9 of 12 | 0.79 to 3.02 | falls at 8 of 12, by at most 22 % |
| plasma carrier phase | 3 of 10 (t = 400 to 600) | 3.07, 3.21, 3.70 | falls at 7 of 10; grows at the last three (0.007, 0.014, 0.022 mrad at t = 1300) |
| plasma centroid | 3 of 10 | 2.22, 2.64, 2.30, rising | grows at 10 of 10 |
| plasma amplitude | 4 of 10 | 1.67 to 2.96 | falls at 9 of 10, by at most 44 % |
| wake E_x | 4 of 10 | 0.63, 0.89, 2.70, 0.80 | falls at 10 of 10, by 16 % to 51 % |
| wake minimum, maximum | 3 and 0 of 10 | 2.80 to 3.79 | falls at 5 and 7 of 10 |
| maps, all rows | 3 of 3 | 2.22, 3.32, 1.18 | falls at 3 of 3, by 34 % to 72 % |
| maps, flank | 2 of 3 | 3.48, 2.44 | falls at 3 of 3, by 31 % to 73 % |
| maps, far wing | 1 of 3 | 1.48 | falls at 2 of 3, by at most 28 % |
| bunch weight | 1 of 6 | 1.63 | falls at 6 of 6, by 70 % to 93 % |

One row has an order at every output and the same one: the vacuum carrier phase, third order. The maps over all rows have one at each of their three outputs, and three different ones; their 2.22, like the flank's 3.48, is the output before the plasma, where the difference over all rows is 0.037, 0.017, 0.010 %. The plasma carrier phase has an order only up to t = 600 and stops falling after: at the last three outputs it is larger at λ₀/64 than at λ₀/32. For the pairs with PIConGPU 4 of 121 output rows (PM–P) and 15 of 121 (S–P) have an order, none of them at more than three outputs of a row. An order of the three run maxima would mislead. They can sit at different outputs (the plasma carrier phase's at t = 400, 500, 500, the vacuum amplitude's at 800, 800, 700), and then it is no output's order; and where they sit at one, as for the maps, it is that output's order in a row whose other outputs have other orders or none.

### 5.1 What the table says

**ParticleMesh–Smilei: not shown to converge.** The vacuum carrier phase falls by a factor 3.3 from λ₀/32 to λ₀/64, at third order at every output; with it goes the fitted phase constant (§3.2). Of the other rows:
- the wake's E_x falls at every output, by 16 % to 51 % from λ₀/32 to λ₀/64, and by 21 % to 25 % in its run maximum (0.182, 0.154, 0.136 %; 25 % as it stands, 21 % with the λ₀/64 window on the other grids' anchor, 0.144 %);
- the maps fall by 34 % over all rows and 31 % in the flank in their largest values, the far wing by 28 %, the plasma amplitude by 21 %, the vacuum amplitude by 7 %;
- the centroid grows with the grid at nearly every output (0.61 to 0.79 × 10⁻³ c/ω₀ in its largest value);
- the wake's minimum is within a factor 1.3 on the three grids and its maximum spans a factor 1.95 (0.056, 0.108, 0.055 %), which is beyond the factor of 1.5 that the output instants alone give an extremum row, and has no direction.

A fall of 7 % to 34 % when the cell halves is not a second-order truncation error that the two codes commit differently (75 %), and not a first-order one towards zero (50 %). It is what a first-order part on top of a grid-independent one would give as well as what a slower fall towards zero would: 0.090 % + 0.046 % · (64/N), for one, has the three wake maxima to 2 %. Three grids do not separate the two. A fourth, finer grid for the two plasma runs would (λ₀/96 has its nodes on the shared lattice): a first-order part goes on falling in proportion to the cell, a constant does not. Until then the scan shows that these differences are not removed by a grid twice as fine, and not what they are (§5.2 to §5.6 narrow it).

**ParticleMesh–PIConGPU: the launch term, in the vacuum amplitude and phase.** In vacuum the amplitude distance is 0.111, 0.066, 0.016 %, in the ratio 6.8 : 4.0 : 1; the fitted oblique term is 1.73, 0.82, 0.175, in the ratio 9.9 : 4.7 : 1, and with the fitted focus offset it leaves 0.001 % or less of the peak amplitude's ratio at every output (§3.2). The vacuum phase is largest at λ₀/48, where the fitted focus offset is largest (+0.46 c/ω₀), and the fit leaves 0.003 mrad of it. For these two observables the distances are that one term, whose size depends on the grid through PIConGPU's cell counts and which the model has small at λ₀/64; which part of the surface and the absorber makes it so was not analysed.

For the other rows that is not shown. The wake's E_x (0.395, 0.123, 0.022 %, as 18 : 5.6 : 1) and the maps (7.8 : 3.6 : 1) fall with the term (9.9 : 4.7 : 1), the first faster than it and the second slower; whether by it was not tested, since no run has the term removed. The centroid (§5.4) and the bunch do not follow it. Where the term is small the pair agrees closely: at λ₀/64, 0.016 % in the vacuum amplitude, 0.20 mrad, 0.022 % in the wake's E_x, 0.011 % and 0.037 % in its extrema, 0.020 % in the maps, 0.034 % in the flank.

**Smilei–PIConGPU: both.** At λ₀/32 and λ₀/48 it follows PM–P; at λ₀/64 its plasma rows are PM–S's (wake 0.138 % against 0.136 %, maps 0.052 % against 0.050 %) while its vacuum rows are PM–P's. On the finest grid, then, the three codes are two that agree to 0.02 % in the wake's E_x and in the maps, and a third 0.14 % and 0.05 % from both.

### 5.2 Smilei's wake difference

WP-053 found Smilei's wake apart from both other codes from t = 700 at λ₀/48. The wake's E_x per output (relative rms over 1.5 λ_p, %; PM–S / PM–P / S–P):

| t | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| 400 | 0.048 / 0.103 / 0.103 | 0.033 / 0.062 / 0.066 | 0.024 / 0.022 / 0.029 |
| 500 | 0.077 / 0.320 / 0.321 | 0.062 / 0.123 / 0.130 | 0.053 / 0.014 / 0.055 |
| 600 | 0.101 / 0.395 / 0.400 | 0.081 / 0.108 / 0.130 | 0.069 / 0.017 / 0.071 |
| 700 | 0.135 / 0.362 / 0.389 | 0.121 / 0.066 / 0.144 | 0.114 / 0.018 / 0.116 |
| 800 | 0.182 / 0.325 / 0.367 | 0.154 / 0.043 / 0.151 | 0.136 / 0.017 / 0.138 |
| 900 | 0.170 / 0.244 / 0.273 | 0.121 / 0.037 / 0.127 | 0.136 / 0.016 / 0.137 |
| 1000 | 0.126 / 0.177 / 0.211 | 0.100 / 0.023 / 0.105 | 0.097 / 0.012 / 0.098 |
| 1100 | 0.124 / 0.132 / 0.178 | 0.084 / 0.021 / 0.085 | 0.074 / 0.009 / 0.075 |
| 1200 | 0.093 / 0.109 / 0.137 | 0.069 / 0.026 / 0.073 | 0.056 / 0.006 / 0.057 |
| 1300 | 0.079 / 0.094 / 0.118 | 0.068 / 0.030 / 0.073 | 0.057 / 0.004 / 0.058 |

The difference is on every grid, with one course in time: it grows from the pulse's entry into the plasma, is largest at t = 800 or 900 (0.182, 0.154, 0.136 %) and falls to 0.06 % to 0.08 % at the end. From λ₀/32 to λ₀/64 its maximum falls by 21 % to 25 % (§5.1); at t = 900 it is 0.170, 0.121, 0.136 %, not monotone. At λ₀/64, where PIConGPU's launch term is small, it is visibly Smilei's alone from t = 500: S–P is within 0.003 % of PM–S at every output from there, and PM–P is 4 to 13 times smaller. "From t = 700" was the λ₀/48 picture, where PIConGPU's launch term hid it before. The plasma amplitude's relative rms has the same course for PM–S (largest at t = 800 to 900: 0.037, 0.028, 0.029 %) and at λ₀/64 the same assignment (S–P 0.029 %, PM–P 0.004 % at t = 900).

It is not explained. Two things it is not: a second-order discretization difference (§5.1), and an effect of Smilei's heavier bunch in proportion (the bunch excess falls from 2.0 % to 0.6 % of the bunch, §5.3, while the wake difference falls by a quarter or less). Not looked at: where along the lineout the difference sits, and Smilei's plasma load against ParticleMesh's particle by particle.

### 5.3 Smilei's bunch

Smilei's bunch is heavier than ParticleMesh's on every grid. Over the common region, from the bunch rule and from the weight above 20 mₑc² (from t = 900, where the whole bunch is above it):

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| bunch, Smilei over ParticleMesh (%) | +2.05 at all six outputs | +1.02, and +1.37 at t = 900 | +0.61 at t = 800 and 900, +0.15 after |
| the same in full-density macro-particle weights | 0.072 | 0.074, 0.100 | 0.079, 0.019 |
| the same in the bunch's own particles (§4.3) | 2.7 | 2.6, 3.6 | 2.8, 0.7 |
| weight above 20 mₑc² (%) | +2.05 | +1.02, and +1.37 at t = 1000 | +0.39 |
| weight above 10 mₑc² (%) | +2.04 to +2.49 | +1.53 to +2.12 | +0.50 to +0.73 |

The relative excess shrinks with the grid, from 2.0 % to 2.5 % at λ₀/32 to 0.15 % to 0.73 % at λ₀/64. In the bunch's own particles it is one to four on every grid (0.7 to 3.6 at 0.026 to 0.028 of a full-density weight each), of about 140, 260 and 470. The excess is a fraction of one full-density weight (0.02 to 0.10) but several of the bunch's particles, so its size does not point at how a single particle is weighted when loaded. At the last output the two codes' spectra differ in 35 of 39, 42 of 44 and 49 of 55 of the bunch's occupied bins on the three grids (`bunch_bins`), so three more trapped particles and particles 2 % heavier are not told apart by these histograms. What the excess is, is open; Smilei's rule at the ramp's foot, WP-053's candidate, is untested, and the particles concerned are loaded there.

ParticleMesh's bunch against PIConGPU's is 0.92 % and 0.56 % apart at λ₀/32, 0.14 % at λ₀/48 and 1.41 %, 0.08 %, 0.54 % and 0.09 % at λ₀/64, where the rule's cut moves between outputs in both codes.

### 5.4 The centroid distances to PIConGPU

They follow neither the grid nor the fitted launch term: 4.4, 1.5, 1.7 × 10⁻³ c/ω₀ for PM–P in vacuum and 4.0, 2.0, 2.4 × 10⁻³ for S–P, so λ₀/64, with the smallest term, does not have the smallest distance. The signed series differ in kind between the grids: PM–P runs from −0.7 to −4.4 and back to −1.3 × 10⁻³ at λ₀/32, from −1.5 through zero to +1.3 and +1.0 at λ₀/48, and from +0.7 to +1.7 and back to +0.7 at λ₀/64.

One mechanism was estimated and does not account for it: an amplitude ratio r(ξ) that varies along the pulse moves the centroid of E_y² by (τ²/4) · d ln r²/dξ. With the fitted oblique term and focus offset of §3.2 that is at most 0.65 to 0.67, 0.62 and 0.18 × 10⁻³ c/ω₀ on the three grids (`launch`, `centroid_shift`, from the two fits; T-718 checks the formula against a pulse summed on a grid), a seventh, two fifths and a ninth of what is measured. What else PIConGPU's launch does to the pulse's timing (the model's response as a function of frequency was not computed) is open.

ParticleMesh–Smilei's centroid difference, 0.6 to 0.8 × 10⁻³ c/ω₀, has one course on the three grids: from +0.46, +0.14, +0.02 × 10⁻³ at t = 200 to −0.61, −0.77, −0.79 at t = 700 and −0.34, −0.49, −0.50 at the end. Its value at t = 200 falls with the grid at a ratio of 2.9, that of a second order; the excursion of about −0.8 × 10⁻³ after it does not fall. The same estimate for the terms fitted to this pair gives 0.02 to 0.06 × 10⁻³.

### 5.5 Smilei's E_y at the back of its window

WP-053 left out the last laser wavelength of the shared columns because Smilei's E_y differs from both other codes' in two or three columns there. On every grid here the maps' shared columns begin exactly at Smilei's window back (x = 447.68 at t = 699.2 and 947.19 at t = 1198.7 on all three), and the columns are 0.785 c/ω₀ apart. The norm over all rows of the difference of two codes' E_y in one column, over the norm of ParticleMesh's column (`scan.json`, `back_columns`; T-711):

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| column at Smilei's back, t = 700, PM–S | 0.227 | 0.229 | 0.461 |
| the same, t = 1200 | 0.265 | 0.272 | 0.347 |
| next column, t = 700 and 1200, PM–S | 0.027, 0.070 | 0.019, 0.047 | 0.011, 0.040 |
| the four columns after it, PM–S | 0.013 to 0.048 | 0.013 to 0.038 | 0.010 to 0.039 |
| column at Smilei's back, both outputs, PM–P | 0.021, 0.042 | 0.008, 0.011 | 0.002, 0.004 |

On every grid one column is off, the one at Smilei's back, by a quarter to a half of the field there; it does not shrink with the grid (it is larger at λ₀/64); and the next column is at most 1.5 times the largest of the four after it at the same output. With the back kept, the far wing's PM–S number at t = 700 is 0.79, 0.80 and 4.17 % against 0.33, 0.28 and 0.24 % without, and the number over all rows 0.096, 0.084, 0.142 % against 0.070, 0.050, 0.046 %.

**The depth of the region is consistent with c times the time since Smilei's window last moved.** WP-053's maps at λ₀/48 (`scan/back_columns_wp053.json`) have the same window back, the same lattice column, and are 0.8 and 1.4 time units later than these, and more columns are off in them: by the same measure 0.398, 0.276, then 0.021 at t = 700.1, and 0.384, 0.398, 0.200, then 0.041 at t = 1200.1. Smilei's window moves by a whole patch (4.71 c/ω₀) at the end of the iteration whose half-step time passes what it has moved (`smilei_window_back`, from its source; `smilei_last_move`, T-719). A disturbance that starts at the new back at that moment and runs forward at c has reached

| output | steps since the move, and the reach (c/ω₀) | columns within reach | columns off |
|---|---|---|---|
| t = 300, the three grids | 7, 10, 14: 1.21 to 1.27 | 2 | 2 (1.3 to 2.1 against 0.5 to 1.1 in the four after) |
| t = 300, WP-053's set | 13: 1.568 | 2 (the third at 1.571) | 2 (1.8, 1.6 against 0.6 to 1.0) |
| t = 699.2, the three grids | 2, 2, 3: 0.24 to 0.36 | 1 | 1 |
| t = 700.1, WP-053's set | 9: 1.09 | 2 | 2 |
| t = 1198.7, the three grids | 1, 2, 2: 0.18 to 0.24 | 1 | 1 |
| t = 1200.1, WP-053's set | 14: 1.69 | 3 (the third at 1.57) | 3, the third at half the size |

Twelve maps, no free parameter; the instant of a move is known to a step (0.09 to 0.18 c/ω₀). At t = 300 the pulse has not reached the back of the window, the field there is small and every column differs by its own size between the codes; the first two still stand out by a factor of 2. The columns bound the speed of whatever travels: at least 0.93 c (the third column, 1.571 from the back, reached at 1.69), and less than 1.45 c from the maps with a wake at the back (the third not reached at 1.09). If the maps at t = 300 are counted, no more than 1.002 c (WP-053's, the third not reached at 1.568), which with the move's instant known to a step is less than 1.09 c. So the depth is consistent with c; it is not a measurement of the speed.

This supports the cause WP-053 proposed, Smilei's boundary acting on the wake's field, in a more specific form: something is set off at Smilei's back each time its window jumps, and it travels into the window. It is a reading of twelve maps and Smilei's moving rule; no Smilei run with another patch size or another boundary has tested it, what the boundary does there was not read in its source, and why the column's difference is larger at λ₀/64 is open. The window's next jump, 4.71 time units later, takes the back past everything the disturbance has reached, which would keep it within one patch of the back; that is an inference, and the margin of one laser wavelength (6.28 c/ω₀) covers a patch.

### 5.6 The far wing

Between ParticleMesh and Smilei the far wing (2.15 to 2.85 waists from the axis) is 0.049, 0.049, 0.050 % before the plasma (t = 300) on the three grids, and 0.33 to 0.34, 0.27 to 0.28 and 0.24 to 0.25 % in it (t = 700 and 1200). The value before the plasma does not depend on the grid at all; the value in the plasma falls slowly, by 28 % from λ₀/32 to λ₀/64, with an order at one of its two outputs. Against PIConGPU it is 0.96, 0.68, 0.20 % before the plasma, the launch term again (the oblique term acts most where the beam is widest), and at λ₀/64 falls to 0.12 % and 0.08 % in the plasma for PM–P while S–P stays at 0.26 %. So on the finest grid the far wing in plasma is, like the wake, Smilei against the other two.

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

**What it does not fix.** Whether ParticleMesh and Smilei converge to each other. Their distances fall slowly or not at all over the three grids (§5.1), so a bound set from them at one grid holds at the next within a third, and nothing here says whether it would on a grid finer still.

**Option 1: fixed bounds against Smilei, the pair whose launches are matched.** For illustration, bounds at 1.6 to 2.5 times the largest PM–S value over the three grids, rounded: carrier phase 0.2 mrad, centroid 0.0015 c/ω₀, vacuum amplitude 0.01 %, plasma amplitude 0.06 %, wake E_x 0.3 %, wake extrema 0.2 %, maps over all rows 0.12 % and flank 0.2 %, highest bin within one bin: twelve rows with the vacuum and plasma ones counted apart.

| | λ₀/32 | λ₀/48 | λ₀/64 |
|---|---|---|---|
| PM–S | passes 12 of 12, at most 0.63 of a bound | passes 12, at most 0.54 | passes 12, at most 0.58 |
| PM–P | fails 10: every row but the wake minimum and the bin (up to 11 times the bound) | fails 5: vacuum phase 3.5, vacuum amplitude 6.6, plasma phase 2.75, both centroids 1.03 and 1.04 | fails 3: vacuum amplitude 1.6, both centroids 1.10 and 1.16; vacuum phase passes at 0.996 |
| S–P | fails 10 | fails 5 | fails 3 |

One set of bounds serves the three grids because the PM–S values change little over them (their largest by at most a third, the wake's maximum by a factor of 2 with no direction). The bounds rest on one reference code whose remaining differences from ParticleMesh are unexplained in four places (§8), and in the wake rows they are 1.6 times a difference that is Smilei's alone on the finest grid. They would not admit PIConGPU as a second reference on any grid as its launch stands. The wake rows carry ±10 % from the window's anchor and the extremum rows a factor of 1.5 or more from the output instants, which the factors of 1.6 to 2.5 cover for the first and barely for the second.

**Option 2: a bound relative to the two reference codes' own distance on the same grid.** ParticleMesh's distance to each reference no larger than f times Smilei–PIConGPU's, row by row. The largest ratio over the rows of §5 without the far wing is 1.16 at λ₀/32 (plasma centroid, PM–P), 1.33 at λ₀/48 (wake maximum, PM–S, an extremum row) and 1.25 at λ₀/64 (plasma phase, PM–P); the far wing is at most 1.03. With f = 1.5 every row passes on every grid; with f = 1 three rows fail at λ₀/32, seven at λ₀/48 and three at λ₀/64. The bar is not a fixed quantity: at λ₀/32 and λ₀/48 it is PIConGPU's launch term (PM–S passes at 0.04 to 0.5 of it in the laser rows), at λ₀/64 it is Smilei's unexplained difference in the plasma rows (PM–S is 0.90 to 0.99 of S–P in the plasma amplitude, the wake and the maps there). It says that ParticleMesh is no outlier among three, with no absolute scale.

**Option 3: a bound across grids against the scheme's dispersion relation, in vacuum.** The measured phase and centroid differences between two grids over the axial Yee prediction are within 2.5 % of one for ParticleMesh at all twelve outputs and both grid pairs (§4.1). A bound of 5 % passes ParticleMesh and Smilei on both pairs and fails PIConGPU at its first outputs (6 % in the phase, 18 % in the centroid). This is the only option here that compares with a closed form instead of another code, and the only one that is about accuracy; it needs two grids per test, covers the laser in vacuum and not the wake or the electrons, and its 2.5 % includes what a plane-wave prediction leaves out of a focused beam. An order criterion in plasma would have the carrier phase (1.99 to 2.01) and the centroid (1.91 to 2.07) and no more: the wake's orders run from 0.8 to 3.6, and its change with the grid is not one mechanism (§4.2).

**Option 4: the electrons in percent of the bunch, or not at all from this case.** The bunch is about 140, 260 and 470 macro-particles, its weight moves by 8 % to 11 % between λ₀/32 and λ₀/64, and at λ₀/64 the rule that defines it is unstable where the gap is narrow. A bound of 2.5 % passes PM–S on all three grids (2.0, 1.4, 0.6 %) and PM–P too (0.9, 0.1, 1.4 %), and fails S–P at λ₀/32 (3.0 %); it is about 3, 7 and 12 of the bunch's particles on the three grids. The largest values at λ₀/64 for the pairs with PIConGPU (1.4 % and 2.0 %) are from the one output where the rule cuts the three spectra at different places. The highest occupied bin is the same in the three codes at every plasma output on λ₀/48 and λ₀/64 and one bin apart once on λ₀/32.

**What none of them can rest on yet.** A PIConGPU column in a fixed bound needs its launch term removed or tested by a run; until then PM–P is small only where the term happens to be (λ₀/64), and for the wake and the maps it is not shown that the term is all that separates the two. The plasma rows against Smilei rest on a difference that is Smilei's alone on the finest grid and has no explanation, so a bound set from PM–S in the wake (0.14 % to 0.18 %) is set by that difference, while the pair without it agrees to 0.02 %; and whether that difference shrinks further with the grid needs a fourth one. Three grids support an order only as one equation: a bound that uses an extrapolated limit would want a fourth as well.

## 8. Verification status and open issues

- **Reviewed once** (the Review row); a second pass follows this commit. One run per code and grid; nothing here is a validation claim, and nothing is an A7 pass.
- **What the review changed.** Claims withdrawn or narrowed: that ParticleMesh and Smilei "do not converge" (not shown to, §5.1); orders of run maxima (orders output by output, §5); that the wake's change with the grid is mostly the driver's place (tested, §4.2); the bunch as a few macro-particles and the inference to loading (§4.3, §5.3); that all distances to PIConGPU follow its launch term (the vacuum amplitude and phase, §5.1); the wake maximum at t = 400 (the window's edge, §4); how sharply the launch model is confirmed (§3.2); that the depth at Smilei's back "is" c times the time (consistent with, §5.5). Added: the anchor's ±10 % on the wake rows (§2), one construction for the factors in the summary, and the four sets of numbers that had no file as keys of `scan.json`.
- **Measured.** Every number in the tables of §3 to §6, from the keys named there.
- **Inferred, and said so where it stands.** That the laser's change with the grid is the Yee dispersion relation's (a plane-wave formula matched to 2.5 %); the λ₀/64 run's distance from the continuum (formula and three-point extrapolation, which agree and are not independent); that the trapped particles were loaded low on the ramp (their weights); that an extremum row is known to a factor of 1.5 at best (two λ₀/48 sets with outputs at different instants); that the region at Smilei's back is set off by its window's moves (§5.5).
- **Confirmed by a prediction.** The 1D model of PIConGPU's launch, on the two grids it had not been built on: its layout sharply at λ₀/32, that the term is small at λ₀/64 (§3.2). Still no PIConGPU run tests it, and PIConGPU's launch is not corrected.
- **Narrowed, not tested by a run.** Smilei's E_y at the back of its window: one column at its back on every grid, and a depth consistent with the time since its window moved in twelve maps (§5.5).
- **Unexplained**, each with its numbers above:
  - ParticleMesh against Smilei in vacuum: 0.004 % in amplitude, a centroid excursion of −0.8 × 10⁻³ c/ω₀ that does not fall with the grid, and fitted focus offsets that differ between amplitude and phase (§3.2, §5.4).
  - Smilei's wake and plasma-amplitude difference from both other codes, on every grid, and whether it has a grid-independent part (§5.1, §5.2).
  - Smilei's bunch excess of one to four of the bunch's particles (§5.3), and why the trapped charge is 8 % to 11 % higher at λ₀/32 in every code (§4.3).
  - The far wing against Smilei in plasma, 0.24 % to 0.34 % (§5.6).
  - The centroid distances to PIConGPU (§5.4).
  - Why the wake's maximum is furthest from converged at t = 700 and 800 in all three codes, and what the part of the wake's change is that is not a displacement (§4.2).
- **Not shown.**
  - ParticleMesh on the CPU on these grids, in Float32, or in its Fast mode.
  - The engine on `main`: ParticleMesh ran the branch's code of `1e36a43`, which does not have what `main` merged after the branch point (#410, #411).
  - A fourth grid: every order is one equation, and a first-order part of a code-to-code distance is not separated from a constant one; and Δy, Δt and the particles per cell were not varied apart from Δx, so the macro-particle weight and the cell size change together.
  - That the wake and map distances to PIConGPU are its launch term: no run has the term removed.
  - PIConGPU's spectrum over the common region (#412).
  - A repeat of any run; Smilei's and PIConGPU's determinism is assumed.
  - Where along the lineout Smilei's wake difference sits, and a particle-by-particle comparison of the loads.
  - A Smilei run with another patch size or boundary, which would test §5.5.
- **The tests.** `scripts/lwfa/test/runtests.jl`, 1602 assertions. For the scan: T-711 (the maps by band, and column by column from their back), T-714 (the three grids are one case), T-715 (the measures across grids: shared nodes, E_x on the nodes and anywhere, the order of three values and of three distances, the wake's extrema and its displacement, the axial Yee values), T-716 (PIConGPU's launch term per grid, to 0.002, with what a cell of its surface and of its absorber does), T-717 (`compare_scan` end to end on synthetic runs), T-718 (`launch_fit` and the centroid shift), T-719 (Smilei's last window move). They test the generators, the readers and the measures on synthetic data, not the physics of the case.
  - What T-717 covers: synthetic fields on the three lattices with planted orders in each code's own rows; the self wake on a window of 1.5 λ_p, against a value computed from what was planted (a dip 1.25 λ_p behind the pulse is in it, a bump 3 λ_p behind is not); the distances as largest over the outputs with the output they are at, from a difference that grows through the run; the extrema by their ratio; orders output by output, on values whose three largest are of two outputs; the times from which a medium and the bunch count; the floor and the negative limit of a distance's order.
  - What it does not cover: the code-to-code wake window and its anchor (T-692 has those on one grid); maps and spectra as the codes write them (hand-made rows; T-701 to T-706 and T-711 have the readers); a `compare_scan` with three codes; `wake_extrema` at a window's edge inside `scan_self` (T-715 has the function); any property of real fields, such as an anchor that differs between grids.
  - Mutants. The review found six mutants of `lwfa.jl` that passed at `fb2015d` and bands in T-716 wide enough for a surface one cell off. At `16d943d` each of the six fails a named assertion, as does a layout one cell off: the smallest instead of the largest over the outputs (T-717, on the difference that grows: 0.0044 against 0.0076), the extrema's distance as a difference instead of a ratio (T-717: 0.00054 against 0.0099), the self wake's window at 1.0 λ_p and with no end (T-717: 0.070 and 0.120 against 0.151), the edge sample kept as an extremum (T-715), the floor of a distance's order at 10⁻⁹ (T-715: an order of 2 where there is none), the surface a cell further out (T-716: 1.540 against 1.748 at λ₀/32), the absorber a cell longer (T-716: 0.207 against 0.183 at λ₀/64). Fifteen mutants of what this round added fail as well: the order of the run maxima in place of an output's, the first output's time in place of the largest's, the guard against a negative limit, the group velocity, the Yee ratio with the wrong centroid, the displacement's sign and the sign of its lag, the flattest tenth for the steepest, a weight of the cubic, the last move an iteration early, the columns normalised by the other code and started at a map's own first column (a bounds error), the centroid shift's factor and one of its two terms, the plasma rows counted from t = 150. All 23 were evaluated in memory; no file of the repository was changed.
