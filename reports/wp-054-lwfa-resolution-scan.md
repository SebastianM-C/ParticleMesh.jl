# LWFA 2D on three grids: A5's first case at λ₀/32, λ₀/48 and λ₀/64

**Summary.**
- A5's first 2D laser-wakefield case on Δx = λ₀/32, λ₀/48 and λ₀/64 (Δy = 4Δx, Δt ∝ Δx, outputs at the same times) in ParticleMesh, Smilei v5.1 and PIConGPU 0.8.0: 18 runs, compared on the lattice points the three grids share.
- The carrier phase and the centroid of the laser converge at second order in each code: order 2.01 at all twelve vacuum outputs in ParticleMesh and Smilei (1.74 to 2.58 in PIConGPU), and in those two by the amount the axial Yee dispersion relation gives, to 2.5 %.
- The wake converges without a constant order: at λ₀/48 its E_x is 0.5 % to 1.4 % from the λ₀/64 run's.
- The three codes share that error. By the largest values over the run, a code at λ₀/48 is 81 to 1091 times further from its own λ₀/64 run than from another code in the vacuum carrier phase, 84 to 221 times in the centroid and 9.2 to 11.5 times in the wake's E_x. A code-to-code distance therefore says whether two implementations solve one discrete problem, not how accurately.
- ParticleMesh against Smilei is not shown to converge. Only their vacuum carrier phase falls with one order at every output; the wake distance falls slowly (0.18, 0.15, 0.14 %), and three grids do not separate a grid-independent part from a first-order one.
- The vacuum amplitude and phase distances to PIConGPU are its launch term. A 1D model built on the λ₀/48 runs predicted 1.75 and 0.18 of the oblique term for λ₀/32 and λ₀/64 before those ran; the fits give 1.73 and 0.175. At λ₀/64 ParticleMesh and PIConGPU are 0.022 % apart in the wake's E_x.
- This is not a validation. A7's rule for this case (accuracy across grids, consistency with Smilei at λ₀/48) came out of this scan.

| | |
|---|---|
| Work package | WP-054 (`trace/wp/WP-054-lwfa-resolution-scan.md`); goals A5, A7. Follows `lwfa-2d-a5.md` |
| Cases | `scripts/lwfa/cases/lwfa-2d-r32.toml`, `lwfa-2d-r48.toml`, `lwfa-2d-r64.toml` |
| Code and runs | ParticleMesh ran `1e36a43`; Smilei's (CPU box request `runs/lwfa-scan-smilei`) and PIConGPU's inputs were generated at `a991f96`; the comparison ran at `16d943d`; all commits of PR #417's branch. Each run's manifest is `wp-054-lwfa-resolution-scan/scan/r<N>/manifest-<run>.toml`, and `scan/manifest.toml` has the hashes of the 18 summaries |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode, CUDA on an RTX 4080 SUPER. Smilei v5.1, 16 OpenMP threads on an AMD EPYC 9655. PIConGPU 0.8.0, CUDA, Float64, the same RTX 4080 SUPER |
| Data | `wp-054-lwfa-resolution-scan/scan/scan.json` (`self`: each code across the grids; `distances`: each pair on each grid, output by output with orders; `launch`, `yee`, `back_columns`, `bunch_bins`), two figures with sidecars, `scan/r<N>/comparison.json` per grid, and `runs/` for the summaries and maps that are on no run branch |
| Review | Two independent passes by fresh numerics-reviewer subagents (claude-fable-5-2, 2026-11-12) on the long version of this report in PR #417, findings in `wp-054-lwfa-resolution-scan/review.md`. This short version had one more pass of the same kind, which recomputed the table and every number from the committed data; its findings are applied |

## Results

The largest distance of each pair over the run, on the three grids:

| Largest over the run, at λ₀/32 / λ₀/48 / λ₀/64 | ParticleMesh–Smilei | ParticleMesh–PIConGPU | Smilei–PIConGPU |
|---|---|---|---|
| vacuum: carrier phase (mrad) | 0.12 / 0.052 / 0.036 | 0.58 / 0.7 / 0.2 | 0.7 / 0.65 / 0.17 |
| vacuum: centroid (10⁻³ c/ω₀) | 0.61 / 0.77 / 0.79 | 4.4 / 1.5 / 1.7 | 4.0 / 2.0 / 2.4 |
| vacuum: amplitude, relative rms (%) | 0.004 / 0.0038 / 0.0037 | 0.11 / 0.066 / 0.016 | 0.11 / 0.065 / 0.015 |
| plasma: carrier phase (mrad) | 0.11 / 0.052 / 0.038 | 1.1 / 0.55 / 0.17 | 1.1 / 0.5 / 0.14 |
| plasma: centroid (10⁻³ c/ω₀) | 0.61 / 0.75 / 0.77 | 4.6 / 1.6 / 1.7 | 4.0 / 1.7 / 2.4 |
| plasma: amplitude, relative rms (%) | 0.037 / 0.028 / 0.029 | 0.11 / 0.056 / 0.013 | 0.12 / 0.056 / 0.029 |
| wake E_x over 1.5 λ_p, relative rms (%) | 0.18 / 0.15 / 0.14 | 0.4 / 0.12 / 0.022 | 0.4 / 0.15 / 0.14 |
| wake minimum, relative difference (%) | 0.12 / 0.097 / 0.12 | 0.16 / 0.1 / 0.037 | 0.17 / 0.11 / 0.12 |
| wake maximum, relative difference (%) | 0.056 / 0.11 / 0.055 | 0.27 / 0.043 / 0.011 | 0.24 / 0.082 / 0.058 |
| maps of E_y, all rows, relative rms (%) | 0.075 / 0.058 / 0.05 | 0.15 / 0.07 / 0.02 | 0.16 / 0.077 / 0.052 |
| maps of E_y, flank, relative rms (%) | 0.093 / 0.071 / 0.064 | 0.23 / 0.13 / 0.034 | 0.24 / 0.14 / 0.071 |
| maps of E_y, far wing, relative rms (%) | 0.34 / 0.28 / 0.24 | 0.96 / 0.68 / 0.2 | 0.94 / 0.66 / 0.27 |
| bunch weight, relative difference (%) | 2.0 / 1.4 / 0.6 | 0.92 / 0.14 / 1.4 | 3.0 / 1.2 / 2.0 |
| highest occupied bin, difference (mₑc²) | 0 / 0 / 0 | 0.2 / 0 / 0 | 0.2 / 0 / 0 |

The table is the output of `julia +1.13 --project=scripts scripts/lwfa/lwfa.jl table reports/wp-054-lwfa-resolution-scan/scan/scan.json`, and T-720 holds this text to it. The outputs counted and the measures are those of `lwfa-2d-a5.md`. The three values of a row can sit at different outputs on different grids, so they have no order of their own; `scan.json` has the orders output by output. The ParticleMesh–Smilei wake E_x row is known to about ±10 %, because the window's anchor differs between grids at one output, and an extremum row to a factor of 1.5.

Each code against its own λ₀/64 run is in `scan.json` under `self` and in `scan/scan_self.png`. At the end of the vacuum run λ₀/48 is 56 mrad and 0.17 c/ω₀ from λ₀/64, and the Yee relation puts λ₀/64 another 72 mrad and 0.22 c/ω₀ from the continuum.

## Not shown, and open

- A fourth grid, so every order is one equation with no check, and Δy, Δt and the particles per cell were not varied apart from Δx.
- ParticleMesh on the CPU on these grids, in Float32, in its Fast mode, or as the engine stands after `1e36a43`.
- That the wake and map distances to PIConGPU are its launch term: no run has the term removed, and no PIConGPU run tests the model.
- The particles: no run kept them. The bunch is an estimated 140, 260 and 470 macro-particles, 8 % to 11 % heavier at λ₀/32 than at λ₀/64 in every code.
- A repeat of any run.
- Unexplained: Smilei's wake and plasma-amplitude difference from ParticleMesh on every grid; its heavier bunch (2.0, 1.4, 0.6 %); the far wing against Smilei (0.34, 0.28, 0.24 %); the centroid distances to PIConGPU; Smilei's E_y at the back of its window, off by 23 % to 46 % in the first shared column on every grid.

Stepping-loop wall times of the plasma run at λ₀/32, λ₀/48 and λ₀/64, single runs: ParticleMesh on CUDA 112, 379 and 864 s; Smilei 115, 385 and 917 s; PIConGPU 61, 136 and 264 s after compilation. Different hardware and diagnostics, ParticleMesh in Reference mode before any optimization round: no performance conclusion is drawn.
