# PIConGPU and Smilei on the 1D set and the oblique 2D Landau case (cross-code stage 2a)

| | |
|---|---|
| Work package | WP-017 (goals A1, A5, A6, A7); the 32-seed oblique ensembles WP-025 (A5, A7) |
| Runs | PIConGPU: `picongpu-landau-fp64-295ddf42`, `picongpu-landau-fp32-295ddf42` (8 seeds each; FP32 analyzed after the review), `picongpu-two-stream-fp64-736a428f`, `picongpu-two-stream-fp32-736a428f`, `picongpu-two-stream-256-fp64-56d5d9ae`, `picongpu-two-stream-256-fp32-56d5d9ae`, `picongpu-landau-oblique-2d-fp64-bc8fd6af` (4 seeds), `picongpu-landau-oblique-2d-fp64-431f4192` (32 seeds, WP-025); Smilei: `smilei-landau-oblique-2d-bc8fd6af` (4 seeds), `smilei-landau-oblique-2d-431f4192` (32 seeds, WP-025); WP-025's design and post-hoc checks `seed-count-20261102T170156`, `subensembles-landau-oblique-2d-431f4192`. The Smilei 1D runs compared against are WP-012's (`reports/crosscode-smilei-1d/stage-1b/`) on the same case files |
| Commits of the inputs | Every run prepared at `917f460f2050ec1a8efa5c59a2d71c2ae7cd9d1b` (clean tree, in each `manifest.toml`); the analyses are recorded in each manifest's `[analysis]` table. The oblique ensemble observable and its seeds were registered in `d61a812`, before it and before any run of the case in this package. These ids are on the branch `wp-017-runs-provenance`, the history before the PR was rebased onto `main`; the PR's commits carry the same content under new ids. WP-025's runs were prepared at `5707406` and analyzed at `e42917a` (clean trees), its 28 seeds registered in `ada4769` before any of their runs; these ids are on the branch `worktree-WP-025-oblique-landau-seeds`, before the squash merge |
| Case hashes (SHA-256, first 8 hex) | `landau.toml` `295ddf42`, `two-stream.toml` `736a428f`, `two-stream-256.toml` `56d5d9ae` (the files WP-012's Smilei runs used), `landau-oblique-2d.toml` `bc8fd6af` (WP-019's file with the seeds and the ensemble entry added) and `431f4192` (the same with 32 seeds, WP-025); full hashes in each manifest under `inputs` |
| PIConGPU | 0.8.0, commit `6bb3470ee16dd5a3a93f24c85b111be3b8440d22`, CUDA backend for `sm_89`; image `localhost/particlemesh-picongpu:a743cbb40f17cdae`, id `1b4f3d0fb3c291e2d861732bda21463920167f39227c21b7947c37a9238ff50b` (toolchain as `pm-versions` printed it in each run's `versions.toml`, whose hash is in the manifest: nvcc 12.5.82, gcc 13.3.0, CMake 3.28.3, OpenMPI 4.1.6, openPMD-api 0.15.2, HDF5 1.10.10, Boost 1.83 from Ubuntu's package; the file is not committed, because the Debian revision of the Boost version trips the repository's IP-address scan) |
| Smilei | v5.1, commit `438d43d8ab727d02dd8f4402c50393343e871b99`, CPU build; image `localhost/particlemesh-smilei:e4f0148b8952fc87`, id `7030c120c62ce14dbae971cce76a3973f0a031ef63632a6c4bf4fa20e3937b4c` |
| Hardware | PIConGPU: NVIDIA GeForce RTX 4080 SUPER (driver 595.104.02; WP-025's runs 595.112.07), one device, one MPI rank. Smilei: Intel Core i9-14900K, 16 OpenMP threads (one per patch), one MPI rank |
| Generator, reader, analysis | `scripts/crosscode/picongpu.jl`, `read.jl` (`PIConGPURead`), `analyze.jl`, `compare.jl` at the commits above; the analysis commit of each run is in its manifest's `[analysis]` table |
| Review | numerics-reviewer (`.claude/agents/numerics-reviewer.md`), fresh subagent, claude-opus-5-5, 2026-11-01; findings `review.md` (one blocking, three major, eight minor), answered in "Review"; WP-025: numerics-reviewer, fresh subagent, claude-opus-5-5, 2026-11-03; findings `review-wp-025.md` (one blocking, one major, eight minor), answered in "Review of WP-025" |

This report records measurements and sets no tolerance (A7: thresholds rest on the measured Smilei–PIConGPU spread, are proposed in `questions/wp-017-a7-thresholds.md`, and the owner accepts them by editing A7). Each run directory holds its manifest (inputs, outputs and their hashes, image, hardware, observables), the generated `params/`, `run.sh`, `units.toml`, the timing record `container.txt`, `observables.json` and the standard plots with JSON sidecars; seeded runs have the same per seed under `seed-<seed>/`. The dumps themselves are not committed. Comparisons are in `compare/`, each a `comparison.json` with the numbers below.

## Setup

### Matching (A5)

| Setting | Smilei | PIConGPU |
|---|---|---|
| Shape | `interpolation_order = 2` | `particles::shapes::TSC` (quadratic B-spline) |
| Current deposition | Esirkepov (`Projector1D2Order`, `Projector2D2Order`) | `currentSolver::Esirkepov<TSC>` |
| Filter | none | none (`--currentInterpolation none`) |
| Pusher | Boris (default `pusher = "boris"`) | `particles::pusher::Boris` |
| Field solver | Yee | `maxwellSolver::Yee` |
| Ions | frozen for twice the run, deposit charge | no pusher, no current; uniform |
| Particles per cell | the case's | the case's in 2D; in quasi-1D the 1D count per x-column (below) |
| Loading | `regular` within 97.5 % of the cell, profile at cell centres | `Quiet` lattice (k + ½)/n over the whole cell, profile at cell centres |
| Thermal momenta | classical Maxwell–Boltzmann energy, \|p\| = √((1 + E)² − 1) | non-relativistic Maxwellian in momentum (`unary::Temperature`) |
| Initial field | Poisson solve of the loaded charge, plus Smilei's constant offset | the discrete Poisson field of the loaded charge as a static `FieldBackgroundE` (below) |
| Output cadence | the case's `scalars_every`, `fields_every` | the same, in steps |
| Precision | FP64 | FP64; the 1D set also in FP32 |
| Reference wavelength | not needed (normalized input) | 0.8 µm (SI input; it cancels in every observable) |

Every PIConGPU-specific choice is listed with the source file it was read from in `scripts/crosscode/README.md` ("PIConGPU specifics").

### Quasi-1D runs of the 1D set

PIConGPU has no 1D geometry. A schema-1 case runs in a 2D box periodic in y, 8 cells wide, with Δy = 8Δx:

- **Width.** PIConGPU needs at least 2·guard + 1 supercells per axis and a guard of 3 cells for TSC's Esirkepov stencil. With supercells of 16 × 1 cells and a guard of 3 supercells along y, the floor is 7 transverse cells. The per-cell counts (50 000 electrons in Landau, 1000 per beam in two-stream) are not divisible by 7, so the generator takes 8.
- **Transverse cell.** With Δy = Δx, the 1D time step 0.95Δx would exceed the 2D Courant limit Δx/√2, which PIConGPU refuses at compile time. With Δy = 8Δx the limit is 0.992Δx. The ky = 0 component of the Yee update is the 1D update with the same Δx and Δt, so the y-averaged fields evolve as in 1D, up to the coupling through the particles.
- **Particles.** Each cell holds the 1D count divided by 8, all on the x lattice (6250 × 1 in Landau, 125 × 1 per beam in two-stream), so an x-column holds the 1D cell's count: 3.2 × 10⁶ electrons per Landau seed, 2.56 × 10⁵ in two-stream at 128 cells, as in Smilei. A column has only 6250 (125) distinct x positions, each repeated in the 8 rows; the velocities are independent, so the noise is that of the 1D count, while the quiet-lattice form factor is that of 6250 (125) per cell (T-255 uses it). The immobile ions keep one per cell (8 per column).
- **Observables** come from the y-averaged Ex and from the energies per unit transverse area, the 1D quantities of Smilei.

T-254 reads these back from the generated files: grid, lattice per column, density, perturbation, time step and step count.

### The initial field

Smilei solves Poisson's equation at t = 0. PIConGPU solves none: its fields start at zero. With Esirkepov's exact charge conservation, a density perturbation loaded with E = 0 keeps a static Gauss-law defect that acts as a neutralizing background, so the Langmuir wave would not be excited at all. The generator therefore supplies the field.

A cosine density sampled at the cell centres and loaded on a quiet lattice with TSC deposits exactly one Fourier mode at the nodes, with amplitude ρ̂ = Σ_s q_s n_s α_s K̂_x K̂_y, where K̂ is the lattice's form factor. The Yee-grid field that satisfies the discrete Gauss law for that charge is analytic: E_d = (ρ̂/K²) κ_d sin(k·x_{E_d}), with κ_d = (2/Δ_d) sin(k_dΔ_d/2) and K² = Σ κ_d². T-255 checks it against a brute-force TSC deposit of the same load to 10⁻¹⁰.

The field enters as a static `FieldBackgroundE` that the pusher sees. PIConGPU adds it to the grid field before the push and for the plugins between steps, and removes it before the field solver, so the total field obeys Maxwell's equations with ∇·E = ρ from t = 0, as after a Poisson solve. It is curl-free on the Yee grid and drives no B.

PIConGPU's own `chargeConservation` plugin (max |∇·E − ρ| on the total field, at every dump) and the analysis's Gauss check on the dumps (`gauss_residual_t0_relative`) confirm it in the runs (Results).

## Linear theory

As in `reports/crosscode-smilei-1d/README.md` and `reports/crosscode-smilei-2d/README.md` (`scripts/crosscode/theory.jl`, T-160, T-273): Landau damping at kλ_D = 0.5, γ = 0.1533595 ω_p, ω_r = 1.4156619 ω_p, in 1D and along k in 2D; cold two-stream with the longitudinal mass γ₀³m, γ = 0.3508758 ω_p; the Ex/Ey energy ratio of the oblique wave on the Yee grid, 3.9975909.

## Results

Rates are of the field amplitude in units of ω_p. "±" is one jackknife standard error over seeds for an ensemble, and one standard error of the fitted slope for a single run; neither includes systematics. Differences are PIConGPU − Smilei unless stated, and they are measurements, not thresholds.

### Checks of the PIConGPU setup

From each run's `observables.json` (`checks`, written by `picongpu_dump_checks` on the 2D dumps before any y-average) and `plots/energy-conservation.json`. Gauss residuals are max |∇·E − ρ| in the units of A1. The relative value is relative to the loaded charge perturbation, max |ρ − ⟨ρ⟩| at t = 0.

| Check | Landau, FP64 (8 seeds) | Two-stream, FP64 (128 / 256 cells) | Two-stream, FP32 (128 / 256 cells) | Oblique 2D, FP64 (4 seeds) |
|---|---|---|---|---|
| Gauss at t = 0, relative | 9.2 × 10⁻¹³ | 1.1 × 10⁻¹² / 2.0 × 10⁻¹² | 7.1 × 10⁻⁴ / 6.7 × 10⁻⁴ | 1.1 × 10⁻¹² (Smilei 6.5 × 10⁻¹⁰) |
| Gauss over the run, absolute | ≤ 1.7 × 10⁻¹⁴ at the unflagged dumps; up to 0.022 at the 18 to 32 flagged dumps per seed (below) | ≤ 8.8 × 10⁻¹⁴ / 3.0 × 10⁻¹³, no dump flagged | ≤ 5.8 × 10⁻⁵ / 1.4 × 10⁻⁴, no dump flagged | ≤ 1.8 × 10⁻¹⁴, no dump flagged |
| \|Ê(0)\| of the perturbed mode, PIConGPU / Smilei | 9.98796 × 10⁻⁴ / 9.98816 × 10⁻⁴ (seed 20261101) | 1.632501 × 10⁻⁴ / 1.632509 × 10⁻⁴ (128 cells) | as FP64 to 7 digits | ensemble \|⟨E·k̂⟩(0)\|² 9.9667 × 10⁻⁷ / 9.9672 × 10⁻⁷ |
| Initial kinetic energy against (3/2) n T L (1D) or (3/2) n T L² (2D), seed mean | −0.32 % (Smilei +0.001 %) | equal to Smilei's to 10⁻⁸ | 10⁻⁵ to 10⁻⁶ below FP64 | −0.34 % (Smilei +0.018 %) |
| Largest relative change of the total energy | 2.4 × 10⁻⁷ to 4.3 × 10⁻⁷ per seed (Smilei 2.7 × 10⁻⁷ to 3.6 × 10⁻⁷) | 1.016 × 10⁻³ / 0.545 × 10⁻³ (Smilei 1.015 × 10⁻³ / 0.545 × 10⁻³) | 1.004 × 10⁻³ / 0.544 × 10⁻³ | 2.5 × 10⁻⁷ to 2.7 × 10⁻⁷ (Smilei 1.3 × 10⁻⁶) |

The kinetic energy is the exact Σ(γ − 1)mc², formed by the reader from PIConGPU's total-energy column minus the rest energy. PIConGPU's own kinetic-energy column switches to p²/2m below γ = 1.005 (`algorithms/KinEnergy.hpp`). The review (B1) found this, and with that column the two-stream energy drift looked twice Smilei's at 256 cells; with the exact energy the two codes agree to three digits.

**Thermal loads.** The initial kinetic energies differ as the loads do. PIConGPU's thermal momenta are the non-relativistic Maxwellian in momentum; Smilei draws the classical energy and sets |p| relativistically (Setup). The review computed by quadrature that PIConGPU's load is 0.31 % below (3/2)nT in exact kinetic energy. The measured seed means are −0.32 % in 1D (eight seeds of 3.2 × 10⁶ electrons) and −0.34 % in 2D (four of 1.68 × 10⁷). The 2D mean is about 2.5 standard errors of the mean from the prediction, which was not followed up.

**Smilei's 2D field energy.** Smilei's t = 0 field energy exceeds PIConGPU's by 2.4 %: Smilei's `Uelm` scalars count extra columns (131/128) and include the constant field Smilei adds after its Poisson solve (`reports/crosscode-smilei-2d/`). PIConGPU's are ½ ΣE²ΔV over the cells.

**FP32 Landau.** The FP32 Landau seeds keep Gauss's law to 4.7 × 10⁻⁴ of the perturbation at t = 0 and to 9 × 10⁻⁶ absolute at unflagged dumps. They show the charge-density glitch at 26 to 41 of 671 dumps per seed. Their total energy changes by at most 2.7 × 10⁻⁵ to 4.6 × 10⁻⁵, which includes the float rounding of the exact kinetic energy formed by subtraction (not separated; FP64: 2.4 × 10⁻⁷ to 4.3 × 10⁻⁷).

**FP32 Gauss.** The FP32 runs keep Gauss's law to the precision of the format: 7 × 10⁻⁴ of the perturbation at t = 0 (the background and the density rounded to float), growing to 5.8 × 10⁻⁵ (128 cells) and 1.4 × 10⁻⁴ (256 cells) absolute over 5253 and 10 506 steps. The review estimates about 4 × 10⁻⁵ for a random walk of FP32 roundoff in E over 5253 steps.

**A glitch in PIConGPU's derived charge density, in the thermal quasi-1D runs only.** In 18 to 32 of the 671 dumps of each Landau seed, the dumped electron charge density summed over the box is short by 1 × 10⁻⁴ to 8 × 10⁻⁴ of the electron charge. At every other dump it is exact to 10⁻¹⁵, and Gauss's law on the dumped fields holds to 10⁻¹⁴. Three observations, from the review (M1), show that the derived density is wrong and not the state:

1. The openPMD dump and the `chargeConservation` plugin derive the density separately from the same particle state at the same iterations. Their spikes fall on different steps (in seed 20261101, 21 against 20, sharing one step). One state gives two densities, so the derivation is not deterministic.
2. The particle-energy plugin runs at the same steps over the same particles and is smooth to 10⁻⁸. A loss of 10⁻⁴ of the electrons would show at 10⁻⁴.
3. At the glitched steps examined, the missing charge sits in one 16 × 1 supercell plus the TSC footprint, away from the periodic seam. About 0.35 % of that supercell's electrons are missing, some 22 frames of 16 particles. That is the footprint of a race in the write-back of `FieldTmp`'s deposit (#241): the supercell stride is fixed at 3 (`fields/FieldTmpOperations.hpp:66`) and the global add is not atomic, so with TSC on supercells one cell high two blocks of the same pass overlap on one row.

None of the cold two-stream runs (same layout, no motion in y) and none of the 2D runs (8 × 8 supercells) shows it. The observables use E only, and the analysis uses the density at t = 0 only, so no number in this report is affected.

The current deposition is unaffected: it computes its stride from the margins and uses atomics (`fields/currentDeposition/Deposit.hpp:79`). #241 does not cover a push and a deposit skipped together for one particle, which would leave no trace in Gauss's law; nothing points to it, and it is not tested here (open items).

### Two-stream: the codes agree to about 10⁻⁶

Pre-registered `growth_rate` (field energy, the 10⁻³ to 3 × 10⁻² window), and the same rule on |Ê₁|²:

| | Smilei | PIConGPU FP64 | PIConGPU FP32 | PIConGPU − Smilei | FP32 − FP64 |
|---|---|---|---|---|---|
| 128 cells, γ | 0.3456117 ± 0.0009 | 0.3456123 ± 0.0009 | 0.3456120 | +5.8 × 10⁻⁷ (+1.7 × 10⁻⁶ relative, 6 × 10⁻⁴ fit errors) | −2.9 × 10⁻⁷ |
| 128 cells, γ from \|Ê₁\|² | 0.3369876 | 0.3369876 | 0.3369885 | +6.9 × 10⁻⁸ | +8.6 × 10⁻⁷ |
| 256 cells, γ | 0.3459258 ± 0.0009 | 0.3459259 ± 0.0009 | 0.3459269 | +7.5 × 10⁻⁸ (+2.2 × 10⁻⁷ relative) | +1.05 × 10⁻⁶ |
| 256 cells, γ from \|Ê₁\|² | 0.3370343 | 0.3370343 | 0.3370342 | −2.8 × 10⁻⁹ | −9.2 × 10⁻⁸ |

PIConGPU's `Uelm_Ex` is the 16-digit total field energy minus the other components (it prints the components with 6 digits, review m2). Both codes fit the same 164 (163) samples over t = 13.80 to 18.76 (13.83 to 18.76), and their saturation energies agree to 3 × 10⁻⁶. Every run is relative to the theory 0.3508758 by the common −1.5 % (−1.4 % at 256 cells) of WP-012, the fit window sampling a transient (#168).

**ParticleMesh against both.** ParticleMesh's WP-012 run gives 0.345904 at 128 cells, +2.9 × 10⁻⁴ from both codes (+8.4 × 10⁻⁴ relative). A post-hoc investigation of this package traced it to two things:

- **The window.** Fitted over the reference codes' window, ParticleMesh gives 0.345577, −3.5 × 10⁻⁵ from them (−1.0 × 10⁻⁴ relative). Its own window starts one output later, because its saturation peak falls one output later, and because log U bends over the window, a one-sample shift moves γ by 1 × 10⁻³ in any of the codes.
- **The initial condition.** ParticleMesh's U(t) differs from PIConGPU's by +1.3 % at t = 1 and −1.0 % from t = 10 on. The cause is ParticleMesh's half-step momentum kick at start-up (A2's centring: the loaded momenta are those at t = 0 and are pushed back to −Δt/2 in E⁰), while Smilei and PIConGPU take the loaded momenta as those at −Δt/2. With the kick removed, ParticleMesh's U(t) is a constant 1.0002 × PIConGPU's, and its γ equals both codes' to six digits. The deviation is linear in the kick (doubling and reversing it were checked), and a linear cold-fluid model predicts the late-time energy ratio 0.9893 against the measured 0.9896.

This is a choice of initial condition, not a defect, and the agreement between Smilei and PIConGPU holds because they share theirs. #226 proposes an option for ParticleMesh's convention; the threshold proposal depends on it.

### Landau damping (quasi-1D) over eight seeds

Pre-registered `damping_rate_ensemble` of `landau.toml` (WP-012), PIConGPU's eight seeds against WP-012's Smilei ensemble on the same seed numbers (the generators' random streams differ, so the seeds are independent draws in each code):

| | Smilei (WP-012) | PIConGPU FP64 | PIConGPU FP32 |
|---|---|---|---|
| ensemble γ ± jackknife | **0.1303 ± 0.0144** | **0.1360 ± 0.0131** | **0.1323 ± 0.0142** |
| maxima used, fit range | 6, t = 4.75 to 15.93 | 6, t = 4.77 to 16.02 | 5, t = 4.75 to 13.64 |
| relative to theory 0.15336 | −15.1 % | −11.3 % | −13.7 % |
| ω_r from the maxima (theory 1.41566) | 1.4046 | 1.3963 | 1.4144 |
| ensemble floor of \|⟨Ê₁⟩\|² | 5.9 × 10⁻¹⁰ | 3.5 × 10⁻¹⁰ | 6.5 × 10⁻¹⁰ |
| per-seed γ at the ensemble's maxima, mean ± std | 0.1296 ± 0.0268 | 0.1343 ± 0.0413 | 0.1302 ± 0.0356 |

PIConGPU − Smilei (FP64): **+0.0057** (+4.4 %), 0.29 combined jackknife errors, 0.21 Smilei seed spreads, 0.14 PIConGPU seed spreads. PIConGPU FP32 − FP64: **−0.0036** (−2.7 %), 0.19 combined jackknife errors; FP32 − Smilei: +0.0021, 0.10 combined errors. The FP32 ensemble is an independent draw of the same seeds (the float load and push give different trajectories), so the difference is noise-dominated like the code difference.

Per seed (γ at the ensemble's maxima):

| seed | 20261101 | 20262101 | 20263101 | 20264101 | 20265101 | 20266101 | 20267101 | 20268101 |
|---|---|---|---|---|---|---|---|---|
| Smilei | 0.1035 | 0.1651 | 0.1221 | 0.1235 | 0.1472 | 0.1664 | 0.1004 | 0.1081 |
| PIConGPU FP64 | 0.1273 | 0.0799 | 0.1049 | 0.2118 | 0.1071 | 0.1690 | 0.1306 | 0.1437 |
| PIConGPU FP32 | 0.1493 | 0.1850 | 0.1080 | 0.1562 | 0.1469 | 0.1216 | 0.0749 | 0.0997 |

The PIConGPU per-seed spread (0.041) is 1.5 times Smilei's (0.027). With eight seeds each this is not significant (an F ratio of 2.4 on 7 and 7 degrees of freedom), and the PIConGPU ensemble floor is lower, not higher. The single-run rule refuses every seed in both codes, as in WP-012.

![Landau ensembles, Smilei and PIConGPU](compare/landau-smilei-vs-picongpu-fp64/landau-ensemble-energy.png)
![PIConGPU ensemble](picongpu-landau-fp64-295ddf42/plots/ensemble-energy.png)

### Oblique Landau damping in 2D over four seeds

The pre-registered ensemble observable `damping_rate_mode_ensemble` (added before these runs, `d61a812`) and WP-019's per-run observables on each seed:

| | Smilei | PIConGPU FP64 | PIConGPU − Smilei |
|---|---|---|---|
| ensemble γ ± jackknife (theory 0.15336) | **0.1570 ± 0.0060** (+2.4 %) | **0.1405 ± 0.0096** (−8.4 %) | **−0.0165**: 1.46 combined jackknife errors, 1.25 Smilei seed spreads, 2.49 PIConGPU seed spreads |
| maxima used, fit range | 5, t = 4.75 to 13.70 | 7, t = 4.79 to 18.12 | |
| ω_r from the maxima (theory 1.41566) | 1.4051 | 1.4135 | |
| ensemble floor | 4.3 × 10⁻¹⁰ | 3.2 × 10⁻¹⁰ | |
| per-seed γ at the ensemble's maxima, mean ± std | 0.1571 ± 0.0132 | 0.1366 ± 0.0066 | |
| `damping_rate_mode` per seed (WP-019's single-run rule), mean ± std over 4 | 0.1465 ± 0.0267 | 0.1522 ± 0.0208 | +0.0057: 0.33 combined errors of the means |
| `damping_rate_energy` | none in any seed (the field-energy scalars reach their floor, as pre-registered) | none in any seed | |
| `energy_ratio` (t = 0, scalars; Yee theory 3.9975909) | 3.98629 in every seed | 3.99760 in every seed | Smilei: its constant post-Poisson field; PIConGPU: the background field, to the 6 digits PIConGPU prints |
| `energy_ratio_mode`, mean ± std over 4 | 3.947 ± 0.053 | 4.088 ± 0.110 | +0.140: 2.3 combined errors of the means |

Smilei's seed 20261031 reproduces WP-019's single run (`damping_rate_mode` 0.16093 in both).

The ensembles differ by 1.5 combined errors. They straddle the theory, Smilei's 0.6 errors above it and PIConGPU's 1.35 errors below. Each ensemble's fit stops at a different maximum (5 against 7), the variation the jackknife over four seeds samples poorly (WP-012's review, F1). The single-run rule averaged over the same seeds agrees to 0.3 errors. With four seeds per code this is a 1.5σ difference; the 32-seed ensembles below show it to be noise.

PIConGPU's `energy_ratio` is the background field, not a measurement of PIConGPU: the generator put the discrete Poisson field there. `energy_ratio_mode` is a ratio of noisy mode energies at the fit maxima, and its two codes differ by 2.3 errors of the means: PIConGPU is 1.6 errors above the Yee value and Smilei 1.9 below.

![Oblique ensembles, Smilei and PIConGPU](compare/landau-oblique-2d-smilei-vs-picongpu-fp64/landau-oblique-2d-ensemble-energy.png)
![PIConGPU oblique ensemble](picongpu-landau-oblique-2d-fp64-bc8fd6af/plots/ensemble-energy.png)
![Smilei oblique ensemble](smilei-landau-oblique-2d-bc8fd6af/plots/ensemble-energy.png)

### Oblique Landau damping in 2D over 32 seeds (WP-025)

The four-seed difference above, 1.46 combined jackknife errors, left open whether the codes differ. WP-025 ran the case over 32 seeds per code, the 28 new seeds registered in the case file (`ada4769`) before any run of them. The case is otherwise unchanged; T-390 checks that against WP-017's copy. The new hash `431f4192` puts the runs into new directories, and the four-seed runs above stay as they are. WP-017's four seeds are the first four of the 32 and were rerun: their dumps are not committed, and the ensemble needs each seed's complex mode coefficient.

**Seed count.** It was chosen from the four-seed runs before the new runs (`seed-count-20261102T170156/`, `scripts/crosscode/seed_count.jl`). For the synthetic estimate, ensembles at each run's measured per-seed noise, 400 per seed count and amplitude model, went through the pre-registered rule (the model of `ensemble_bias.jl`). For the measured estimate, the jackknife errors were scaled as 1/√M. At M = 32 per code the spread of the two-code difference is 0.0058 (synthetic) and 0.0040 (measured), so a real difference of WP-017's 0.0165 would stand 2.9 or 4.1 of them from zero. A 4-spread separation by both estimates would have needed 56 to 65 seeds per code (the synthetic spread falls a little slower than 1/√M, so the count depends on the M it is scaled from), 14 h to 16 h of runner time; 32 was the compromise. The measured jackknife errors at 32 seeds are 0.0033 in both codes. The synthetic prediction was 0.0030 to 0.0039 for PIConGPU and 0.0034 to 0.0042 for Smilei, whose measured value sits just below its range: its per-seed noise after t = 3 measured at 32 seeds (4.7 × 10⁻¹⁰) is lower than the four-seed estimate the prediction used (5.8 × 10⁻¹⁰).

**Runs.** `smilei-landau-oblique-2d-431f4192` and `picongpu-landau-oblique-2d-fp64-431f4192` were prepared at `5707406` (clean tree) and analyzed at `e42917a`. The images are those of the four-seed runs; the logs of all six runner jobs name the same image id per code. The PIConGPU driver was 595.112.07 (WP-017: 595.104.02). The work ran as six runner jobs on the two directories: the generated `run.sh` skips finished seeds and starts none after a 5400 s budget (T-391). Per seed, Smilei took 550 s to 563 s on 16 threads and PIConGPU 314 s to 329 s, plus a compile of 103 s and 67 s in its two jobs. Committed per run: the top directory complete, with `seed-coefficients.json` (each seed's complex coefficient of E·k̂, from which every ensemble number below can be recomputed), and per seed the observables, manifest, sidecars and PIConGPU's energy files. Only the first seed keeps its PNGs: the standard plots of all 64 seeds would add 54 MB.

**Reruns of WP-017's seeds.** Smilei reproduces every committed per-seed observable of the four seeds bit for bit, and its first four seeds reproduce WP-017's ensemble γ (0.15697805). PIConGPU's reruns are not bitwise. The atomic current deposit would allow that, but the cause is not tested: the driver changed since WP-017 and the binary was rebuilt. `energy_ratio` (t = 0) is identical. The single-seed `damping_rate_mode` moves by 3.5 × 10⁻⁶ to 0.010, within the per-seed spread of 0.021, and `energy_ratio_mode` by up to 0.05. The four-seed ensemble γ from the reruns is 0.14047, against WP-017's 0.14045; its jackknife error moves from 0.0096 to 0.0141.

**Checks**, every seed: Gauss's law at t = 0 relative to the loaded perturbation, 1.1 × 10⁻¹² in PIConGPU (6.5 × 10⁻¹⁰ in Smilei). PIConGPU's dumps hold ≤ 1.8 × 10⁻¹⁴ absolute over the run, with no dump flagged. The largest relative change of the total energy is 2.4 × 10⁻⁷ to 2.9 × 10⁻⁷ in PIConGPU and 1.3 × 10⁻⁶ to 1.4 × 10⁻⁶ in Smilei. The initial kinetic energy against (3/2) n T L², seed mean ± standard error, is −0.0013 % ± 0.0032 % in Smilei and −0.321 % ± 0.004 % in PIConGPU. For PIConGPU's non-relativistic Maxwellian in momentum the series −(5/4)T + (35/8)T² gives −0.310 %. That leaves 2.7 standard errors unexplained (four seeds: −0.34 %, "not followed up" above; still not followed up).

| | Smilei | PIConGPU FP64 | PIConGPU − Smilei |
|---|---|---|---|
| ensemble γ ± jackknife (theory 0.15336) | **0.1504 ± 0.0033** (−1.9 %, −0.9 errors) | **0.1458 ± 0.0033** (−4.9 %, −2.3 errors) | **−0.0047**: 1.01 combined jackknife errors, 0.25 seed spreads of either code |
| fit's own standard error | 0.0022 | 0.0025 | |
| maxima used, fit range | 5, t = 4.76 to 13.69 | 6, t = 4.76 to 15.91 | |
| ω_r from the maxima (theory 1.41566) | 1.4066 | 1.4090 | |
| ensemble floor; floor / (ln 2 × noise of the mean) | 4.7 × 10⁻¹⁰; 51 | 3.5 × 10⁻¹⁰; 42 | |
| per-seed γ at the ensemble's maxima, mean ± std | 0.1499 ± 0.0189 | 0.1450 ± 0.0191 | |
| `damping_rate_mode` per seed (single-run rule), mean ± std over 32 | 0.1454 ± 0.0183 | 0.1465 ± 0.0212 | +0.0011: 0.23 combined errors of the means |
| `damping_rate_energy` | none in any seed, as pre-registered | none in any seed | |
| `energy_ratio_mode`, mean ± std over 32 (Yee 3.99759) | 4.033 ± 0.138 (+1.4 errors of the mean) | 3.980 ± 0.168 (−0.6) | −0.052: 1.4 combined errors of the means |

**The floor no longer falls with M.** At 32 seeds the last 20 % of the run, where the rule takes its floor, holds residual signal (floor / (ln 2 × noise of the mean) is 42 to 51, against 2.9 to 5.5 at four seeds). The floor barely moved from the four-seed runs (4.3 and 3.2 × 10⁻¹⁰). More seeds therefore lower the noise at the maxima used (signal-to-noise ratio ≥ 350 at the last one) but do not lengthen the fit. The synthetic ensembles include this, since they put the signal through the same rule.

**Sub-ensembles** (`subensembles-landau-oblique-2d-431f4192/`, post hoc): the rule on disjoint groups of consecutive seeds, group 1 being WP-017's four.

| seeds per ensemble | groups | Smilei: sd of γ / mean jackknife / mean γ | PIConGPU: sd / mean jackknife / mean γ | sd of the group-by-group difference |
|---|---|---|---|---|
| 4 | 8 | 0.0116 / 0.0101 / 0.1465 | 0.0129 / 0.0095 / 0.1443 | 0.0138 |
| 8 | 4 | 0.0086 / 0.0073 / 0.1479 | 0.0098 / 0.0057 / 0.1435 | 0.0136 |
| 16 | 2 | 0.0060 / 0.0066 / 0.1497 | 0.0014 / 0.0068 / 0.1457 | 0.0074 |

One split into eight groups is one draw, and a spread from eight values is uncertain by about ±27 %, so the same file also holds 300 random splits into groups of four and a bootstrap of each run's seeds (the review's B1 and M1). Over the random splits, the measured spread of the four-seed γ over their mean jackknife error has a median of 0.97 in Smilei (5 % to 95 %: 0.54 to 1.50) and 0.93 in PIConGPU (0.50 to 1.49). The consecutive split's 1.15 and 1.36 are ordinary draws from that, and the synthetic ensembles at M = 4 give 1.00 to 1.09. So the four-seed jackknife is consistent with an unbiased one, to about ±30 %. The four-seed difference scatters with sd 0.0138 over the eight consecutive group pairs, against 0.0155 from the synthetic ensembles; WP-017's −0.0165 is 1.2 of those. The mean four-seed γ minus the 32-seed γ has a median of −0.0017 in Smilei (−0.0040 to +0.0002) and −0.0007 in PIConGPU (−0.0027 to +0.0016) over the random splits. The synthetic ensembles give the rule a bias of −0.0009 to −0.0027 (Smilei) and −0.0001 to −0.0016 (PIConGPU) at M = 4, and at most 3 × 10⁻⁴ at M = 32. A comparison with the rule is therefore made at equal seed counts, as a precaution.

**A common window** (the same file, post hoc): all 32 seeds fitted on the first k maxima after t = 3, the same dumps in both codes, with the jackknife on fixed samples:

| maxima k (last at t) | Smilei | PIConGPU | PIConGPU − Smilei |
|---|---|---|---|
| 3 (9.20) | 0.1451 ± 0.0028 | 0.1477 ± 0.0027 | +0.0025 (0.66 combined errors) |
| 4 (11.44) | 0.1467 ± 0.0032 | 0.1513 ± 0.0032 | +0.0046 (1.02) |
| 5 (13.68) | **0.1504 ± 0.0034** (Smilei's rule) | 0.1496 ± 0.0035 | −0.0008 (0.17) |
| 6 (15.93) | 0.1501 ± 0.0031 | **0.1458 ± 0.0033** (PIConGPU's rule) | −0.0043 (0.96) |

Smilei's rule stops at five maxima because its sixth, 4.6925 × 10⁻⁹, lies 0.054 % below 10 × its floor (4.6950 × 10⁻⁹). Taking it would not change Smilei's γ (0.1501 against 0.1504). PIConGPU's sixth maximum, 5.27 × 10⁻⁹, lies 12 % above Smilei's: 1.1 standard deviations of the noise of the two codes' difference there, 1.5 of one code's. The difference changes sign with the window and stays within about one combined error at every k.

The stop is structural at this seed count. The floor is the damped mode itself (about half its envelope over the last 20 % of the run), so 10 × floor falls near the sixth maximum whatever M is, and which side a run lands on is close to a coin flip. Of Smilei's 32 leave-one-out fits, 17 take five maxima and 15 take six, and its bootstrap samples split 159 to 141. Every PIConGPU leave-one-out fit takes six (bootstrap: 275 six, 25 seven), so its jackknife does not sample the window shift. The bootstrap spreads (0.0031 Smilei, 0.0035 PIConGPU, 0.0047 for the difference) agree with the jackknife errors (0.0033, 0.0033, 0.0046) all the same. A third code running the rule meets the same cut.

**Decision.** The four-seed difference was noise. At 32 seeds per code the ensemble γ of the two codes differ by −0.0047 ± 0.0046 under the pre-registered rule, and by −0.0043 to +0.0046 over common windows of three to six maxima, each within 1.02 combined errors. A real difference of the four-seed size, 0.0165, would lie 2.5 combined errors from the rule's value. The 95 % interval of a code difference under the rule is −0.014 to +0.004 in the normal approximation (−0.014 to +0.005 with the bootstrap spread). Smilei and PIConGPU do not differ at this resolution, and A7's 2D spread for this case is the statistical error of the ensembles. The proposal (`questions/wp-017-a7-thresholds.md`, P2) now rests on the 32-seed errors.

**Against theory.** Against the case's pre-registered value, the non-relativistic root 0.15336, both 32-seed ensembles lie low: Smilei by 0.9 jackknife errors, PIConGPU by 2.3, and their mean 0.1481 ± 0.0023 by 2.3. The rule's own bias on synthetic ensembles at 32 seeds is at most 3 × 10⁻⁴ (`seed-count.json`). That value is not the linear theory of either code, though. Both push with v = p/γ, and at T = 2.5 × 10⁻³ mₑc² the resonant electrons (v ≈ ω/k ≈ 3 v_th) move slower than their momenta say. The physics breaker of this package found this; `landau_rate_relativistic` (`scripts/crosscode/theory.jl`, T-394, post hoc) solves the linearized initial-value problem in time for each code's load. A second implementation by the breaker, with a numerical angular quadrature, agrees to 10⁻⁵. Neither is checked against the literature, and the grid is not included (the breaker estimates the TSC form factor at +0.0005 in γ, not checked here). Comparison: `landau-relativistic-431f4192/`, `scripts/crosscode/landau_relativistic.jl`.

| | theory, own load, relativistic push | measured − theory | the same against 0.15336 |
|---|---|---|---|
| Smilei (classical energy, \|p\| relativistic) | γ 0.1478, ω 1.4101 | +0.0026 (+0.80 errors) | −0.0029 (−0.89) |
| PIConGPU (Maxwellian in momentum) | γ 0.1463, ω 1.4097 | −0.0005 (−0.15 errors) | −0.0076 (−2.32) |
| PIConGPU − Smilei | −0.0015 | −0.0031 (−0.68 combined errors) | |
| ParticleMesh's 3V load (Maxwellian in velocity, u = γv), for P2 | γ 0.1525, ω 1.4110 | | |

The same solver gives 0.153365 for the non-relativistic Maxwellian, the pre-registered root to 6 × 10⁻⁶. Without a free parameter (the amplitude is |⟨E·k̂⟩(0)|²), the theory curve of each load lies on its code's ensemble: the ratio of measured to theory |⟨E·k̂⟩|² at the maxima the rule used is 0.95 to 1.03 (Smilei) and 0.93 to 1.06 (PIConGPU), against 1.04 to 1.13 and 1.05 to 1.29 for the non-relativistic curve. The deficit below 0.15336 is therefore the relativistic push with each code's load, and the codes' measured frequencies (1.4066, 1.4090) are those of their loads too. The two loads differ in kinetic energy by O(T) = 2.5 × 10⁻³, but in γ by 1 %, because the resonance sits in the tail. The same correction applies to the 1D Landau ensembles of #167 at this temperature; it is not computed there.

![Both 32-seed ensembles against linear theory with each code's load](landau-relativistic-431f4192/landau-relativistic.png)

![Oblique ensembles over 32 seeds, Smilei and PIConGPU](compare/landau-oblique-2d-431f4192-smilei-vs-picongpu-fp64/landau-oblique-2d-ensemble-energy.png)
![Sub-ensembles of 4, 8, 16 and 32 seeds](subensembles-landau-oblique-2d-431f4192/subensembles.png)
![Seed count: spread of the difference against M, from the four-seed runs](seed-count-20261102T170156/seed-count.png)
![Smilei, 32 seeds](smilei-landau-oblique-2d-431f4192/plots/ensemble-energy.png)
![PIConGPU, 32 seeds](picongpu-landau-oblique-2d-fp64-431f4192/plots/ensemble-energy.png)

### Compile and run times

Wall time inside the container (`container.txt`). This is not a throughput comparison: the codes ran on different hardware, the quasi-1D runs are far from PIConGPU's design point, and A5's fair comparison on one card is a non-goal of this package.

| Case | PIConGPU compile | PIConGPU run (per seed) | ns per mobile particle-step | Smilei run (per seed) | ns per mobile particle-step |
|---|---|---|---|---|---|
| two-stream, 128 cells, FP64 / FP32 | 71 s / 71 s | 103 s / 96 s | 77 / 71 | 17 s (16 threads, WP-012) | 12 |
| two-stream, 256 cells, FP64 / FP32 | 71 s / 71 s | 147 s / 126 s | 27 / 23 | 48 s | 9 |
| Landau, FP64 | 71 s | 1110 s to 1203 s | 129 to 140 | 79 s to 96 s (8 threads, WP-012) | 9 to 11 |
| Landau, FP32 | 70 s | 803 s to 853 s | 94 to 99 | | |
| oblique 2D, FP64 | 68 s | 327 s to 333 s | 5.7 to 5.9 | 559 s to 569 s (16 threads) | 9.8 to 10.0 |

In quasi-1D the GPU has little work to spread: 512 or 1024 cells, so 512 or 1024 threads per kernel. In addition, each openPMD dump costs about 0.6 s whatever its size (README, "PIConGPU specifics"), and the Landau case dumps 671 times. In 2D (16 384 cells) PIConGPU on the RTX 4080 SUPER in FP64 is 5.9 ns per particle-step with output, the compile once per setup.

## Review

A fresh numerics-reviewer subagent (claude-opus-5-5, 2026-11-01) reviewed the code, the tests, the runs and this report before the FP32 Landau run finished. Its findings are in `review.md`, and the answers below have been applied.

| Finding | Answer |
|---|---|
| **B1** (blocking): PIConGPU's kinetic-energy column is p²/2m below γ = 1.005 | Fixed. `read_picongpu_scalars` forms Σ(γ − 1)mc² from the exact total-energy column minus the rest energy of the load, and keeps PIConGPU's column as `Ukin_threshold`. All runs were re-analyzed. The energy rows above are the exact ones: the two-stream drift agrees with Smilei's to three digits, and the open item about it is withdrawn. The threshold is listed in `scripts/crosscode/README.md` and in `questions/wp-017-lwfa-case.md` |
| **M1**: the Esirkepov argument does not rule out a skipped push | Fixed. The paragraph now rests on the review's three observations (two derivations of one state disagree, the energy plugin sees every particle, the deficit is confined to one supercell), and a skipped push is stated as untested. The localization went to the upstream reduction |
| **M2**: quasi-1D Gauss and charge numbers had no committed source | Fixed. `picongpu_dump_checks` (`analyze.jl`) writes the per-dump Gauss residual and each species' total charge into every PIConGPU run's `observables.json`, and the checks table cites them |
| **M3**: the #226 numbers are relative | Fixed: both are labelled, with the absolute values next to them, and the as-is value is 0.345577 |
| m1: per-seed energy ranges | Regenerated from the re-analysis |
| m2: 6-digit `Uelm_Ex` | Fixed. `Uelm_Ex` is the 16-digit total minus the other components. The 256-cell difference is now +7.5 × 10⁻⁸ (2.2 × 10⁻⁷ relative), and `T-215.wp017` was updated |
| m3: cold two-stream says nothing about thermal momenta | Reworded (open items) |
| m4: a constant factor in U does not change γ | The relaxation argument is removed from the proposal |
| m5: FP32 Gauss not reported | Added: an FP32 column in the checks table, and a paragraph |
| m6: stale Gauss-check comment | Fixed in `analyze.jl` |
| m7: provenance | All runs re-analyzed from a clean tree (each manifest's `[analysis]`). `compare.jl` records the committed report directories |
| m8: repeated x positions in quasi-1D | One clause added under "Quasi-1D runs" |

Not verified by the review, and still open: Smilei's side of the initial-momentum convention and the 131/128 explanation of Smilei's 2D `Uelm` (Smilei's source not checked); whether the push can skip particles; the cell-centred-weight explanation of ParticleMesh's remaining 2 × 10⁻⁴; the 0.6 s per dump; the FP32 Landau run, analyzed after the review.

## Review of WP-025

| Finding | Answer |
|---|---|
| **B1** (blocking): the four-seed jackknife "underestimates" rests on one split | Fixed. `subensembles.jl` now draws 300 random splits and a bootstrap per run; the report gives their distribution (medians 0.97 and 0.93) and the synthetic ratios, and calls the four-seed jackknife consistent with an unbiased one to about ±30 %. The limitations bullet and P2 say the same |
| **M1**: the four-seed mean below the 32-seed value is mostly the split | Fixed. The report gives the random-split distribution of that shift and rests the equal-seed-count requirement on the synthetic bias, as a precaution; P2 likewise |
| m1: rounding in the common-window table | Fixed (0.1451, 0.1504) |
| m2: 0.054 %, and which noise "one standard deviation" refers to | Fixed |
| m3: the seed count for a 4-spread separation, the per-code synthetic ranges | Fixed: 56 to 65 seeds, 14 h to 16 h; the ranges per code, with Smilei's lower per-seed noise at 32 seeds |
| m4: the interval is the normal approximation | Fixed: stated, with the bootstrap's +0.005 |
| m5: the stop is structural | Added: a paragraph with the leave-one-out stops (17 to 15 in Smilei, all six in PIConGPU) and the bootstrap spreads, all from `subensembles.json` |
| m6: P2's false-fail rate and its assumption of a shared deficit | Added to P2 for the owner |
| m7: the rerun's jackknife; the cause of the non-bitwise reruns | Added the jackknife (0.0096 to 0.0141); the cause is stated as untested |
| m8: the window jackknife and the group mean jackknife untested | Fixed: T-392 checks the group mean and the resampling bookkeeping; T-393 checks that the fixed-window jackknife equals the rule's when every leave-one-out fit takes the same maxima, and fails without the (M − 1)/M factor |

## Verification status

- T-250 (unit conversion), T-251 (the oblique profile in both generators), T-252 (openPMD reader on files in PIConGPU's layout), T-253 (recipe pinned), T-254 (quasi-1D equivalence), T-255 (the initial field against a brute-force deposit) and T-256 (the oblique seed ensemble, WP-019's entries unchanged by hash) pass on the CPU: `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`. CI does not run this file.
- The reader's layout assumptions (meshes under `fields/`, F[y][x], spacing in axisLabels order, position x first) were checked on a run's file.
- The generated setups compiled and ran first time in the runner's image; PIConGPU's own `chargeConservation` and the analysis's Gauss check confirm the initial field.
- The oblique ensemble entry was committed (`d61a812`) before any run of the case in this package, and T-256 checks that WP-019's four entries are unchanged.
- T-390 (the 32-seed case is WP-017's apart from the seed list), T-391 (the generated run scripts skip finished seeds and stop at their budget, with fake codes on PATH) T-392 (sub-ensembles, the common window and the resampling on synthetic seeds) and T-393 (the fixed-window jackknife against the rule's) pass on the CPU. WP-025's 28 seeds were committed (`ada4769`) before any of their runs, and the observable is WP-017's, unchanged.
- The ParticleMesh two-stream investigation is post hoc: it reran WP-012's ParticleMesh case with the start-up kick scaled (0, 1, 2, −1), from scripts outside the repository, with no repository file changed; its baseline reproduced WP-012's stored series exactly. The proposal of an option is #226.
- Not verified: whether a particle can be skipped in the push and the deposit together (#241 covers only the `FieldTmp` race; a skipped push leaves no Gauss trace); the 2D seed-mean kinetic energy sits about 2.5 standard errors from the quadrature prediction; the cell-centred-weight explanation of ParticleMesh's remaining 2 × 10⁻⁴ in U.

## Known limitations and open items

- **Seeds.** Eight seeds per code in 1D leave the Landau difference unresolved (0.3 combined errors). In 2D, 32 seeds per code resolve the difference to ±0.0046 and find none (1.0 combined errors). At four seeds the jackknife is consistent with the measured spread to about ±30 %, and the rule has a small bias that falls with the seed count (−0.001 to −0.003 at four seeds on synthetic ensembles), so comparisons use equal seed counts. The rule's stop at this case's floor is structural (WP-025, "over 32 seeds").
- **Theory values of the Landau cases** are the non-relativistic root, while every code pushes relativistically: at T = 2.5 × 10⁻³ the matching rate is 3.6 % (Smilei's load) and 4.6 % (PIConGPU's) lower, and 0.5 % lower for ParticleMesh's 3V load (`landau_rate_relativistic`, WP-025, post hoc, without the grid). The case files still carry the pre-registered value (#268).
- **PIConGPU's 2D initial kinetic energy** is 0.011 % (2.7 standard errors over 32 seeds) further below (3/2) n T than the non-relativistic-Maxwellian series predicts. Not followed up.
- **Initial conditions are not identical across the codes**, by design of each code: the in-cell lattice (whole cell against 97.5 %), the thermal momenta (non-relativistic Maxwellian in momentum against Smilei's classical energy with relativistic |p|, O(T) = 2.5 × 10⁻³ in energy but 1 % in the Landau rate, WP-025), Smilei's constant post-Poisson field, and ParticleMesh's half-step kick (#226). The cold two-stream case shows that the in-cell lattice does not matter at 10⁻⁶ between Smilei and PIConGPU; being cold, it says nothing about the thermal momenta.
- **The charge-density glitch** of PIConGPU in the thermal quasi-1D layout (Results, checks): a race in `FieldTmp`'s deposit, reported as #241; a push and a deposit skipped together for one particle is untested.
- **PIConGPU's kinetic-energy column** is p²/2m below γ = 1.005; the reader forms the exact kinetic energy from the total-energy column (README, "PIConGPU specifics").
- **Throughput** is not compared (non-goal); the quasi-1D runs are I/O- and occupancy-bound in PIConGPU.
- **Precision of PIConGPU's scalars**: the per-component field energies are printed with 6 significant digits, which sets the 1.6 × 10⁻⁶ offset of the oblique `energy_ratio` from the Yee value; the totals have 16.
- **Weibel** is not run in PIConGPU (non-goal), and the 2D LWFA case is not written; its prerequisites are in `questions/wp-017-lwfa-case.md`.
- The proposed thresholds are in `questions/wp-017-a7-thresholds.md`, for the owner.
