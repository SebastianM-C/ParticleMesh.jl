# PIConGPU and Smilei on the 1D set and the oblique 2D Landau case (cross-code stage 2a)

| | |
|---|---|
| Work package | WP-017 (goals A1, A5, A6, A7) |
| Runs | PIConGPU: `picongpu-landau-fp64-295ddf42`, `picongpu-landau-fp32-295ddf42` (8 seeds each), `picongpu-two-stream-fp64-736a428f`, `picongpu-two-stream-fp32-736a428f`, `picongpu-two-stream-256-fp64-56d5d9ae`, `picongpu-two-stream-256-fp32-56d5d9ae`, `picongpu-landau-oblique-2d-fp64-bc8fd6af` (4 seeds); Smilei: `smilei-landau-oblique-2d-bc8fd6af` (4 seeds). The Smilei 1D runs compared against are WP-012's (`reports/crosscode-smilei-1d/stage-1b/`) on the same case files |
| Commits of the inputs | Every run prepared at `917f460f2050ec1a8efa5c59a2d71c2ae7cd9d1b` (clean tree, in each `manifest.toml`). The oblique ensemble observable and its seeds were registered in `d61a812`, before it and before any run of the case in this package |
| Case hashes (SHA-256, first 8 hex) | `landau.toml` `295ddf42`, `two-stream.toml` `736a428f`, `two-stream-256.toml` `56d5d9ae` (the files WP-012's Smilei runs used), `landau-oblique-2d.toml` `bc8fd6af` (WP-019's file with the seeds and the ensemble entry added); full hashes in each manifest under `inputs` |
| PIConGPU | 0.8.0, commit `6bb3470ee16dd5a3a93f24c85b111be3b8440d22`, CUDA backend for `sm_89`; image `localhost/particlemesh-picongpu:a743cbb40f17cdae`, id `1b4f3d0fb3c291e2d861732bda21463920167f39227c21b7947c37a9238ff50b` (toolchain in each run's `versions.toml`: nvcc 12.5.82, gcc 13.3.0, openPMD-api 0.15.2, HDF5 1.10.10) |
| Smilei | v5.1, commit `438d43d8ab727d02dd8f4402c50393343e871b99`, CPU build; image `localhost/particlemesh-smilei:e4f0148b8952fc87`, id `7030c120c62ce14dbae971cce76a3973f0a031ef63632a6c4bf4fa20e3937b4c` |
| Hardware | PIConGPU: NVIDIA GeForce RTX 4080 SUPER (driver 595.104.02), one device, one MPI rank. Smilei: Intel Core i9-14900K, 16 OpenMP threads (one per patch), one MPI rank |
| Generator, reader, analysis | `scripts/crosscode/picongpu.jl`, `read.jl` (`PIConGPURead`), `analyze.jl`, `compare.jl` at the commits above; the analysis commit of each run is in its manifest's `[analysis]` table |
| Review | see "Review" below |

This report records measurements and sets no tolerance (A7: thresholds rest on the measured Smilei–PIConGPU spread, are proposed in `questions/wp-017-a7-thresholds.md`, and the owner accepts them by editing A7). Each run directory holds its manifest (inputs, outputs and their hashes, image, hardware, observables), the generated `params/`, `run.sh`, `units.toml`, `versions.toml`, the timing record `container.txt`, `observables.json` and the standard plots with JSON sidecars; seeded runs have the same per seed under `seed-<seed>/`. The dumps themselves are not committed. Comparisons are in `compare/`, each a `comparison.json` with the numbers below.

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
- **Particles.** Each cell holds the 1D count divided by 8, all on the x lattice (6250 × 1 in Landau, 125 × 1 per beam in two-stream), so an x-column holds the 1D cell's count: 3.2 × 10⁶ electrons per Landau seed, 2.56 × 10⁵ in two-stream at 128 cells, as in Smilei. The immobile ions keep one per cell (8 per column).
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

| Check | Landau (quasi-1D) | Two-stream (quasi-1D) | Oblique 2D |
|---|---|---|---|
| Gauss's law on the dumps, max \|∇·E − ρ\| (normalized units) at t = 0, and over the run | 9 × 10⁻¹⁵ at t = 0; 1 × 10⁻¹⁴ in 650 of 671 dumps of seed 20261101, up to 0.022 in the other 21 (below) | 9 × 10⁻¹⁴ at most over all 165 dumps | 1.6 × 10⁻¹⁴ at most over all 212 dumps of seed 20262031; relative to max \|ρ\| at t = 0, 1.1 × 10⁻¹² (Smilei 6.5 × 10⁻¹⁰) |
| PIConGPU `chargeConservation` (max \|∇·E − ρ\| per cell as a charge), relative to the loaded charge perturbation of a cell | 0.9 × 10⁻¹² at t = 0; about 20 of 671 outputs per seed jump to about 2 (below) | 1.2 × 10⁻¹² at t = 0, 8.7 × 10⁻¹¹ at most | 1.1 × 10⁻¹² at t = 0, 1.8 × 10⁻¹² at most |
| \|Ê(0)\| of the perturbed mode, PIConGPU / Smilei | 9.98796 × 10⁻⁴ / 9.98816 × 10⁻⁴ (seed 20261101) | 1.632501 × 10⁻⁴ / 1.632509 × 10⁻⁴ | ensemble \|⟨E·k̂⟩(0)\|² 9.9667 × 10⁻⁷ / 9.9672 × 10⁻⁷ |
| Initial kinetic energy against (3/2) n T L (Landau) or (3/2) n T L² (2D) | −0.30 % (Smilei +0.002 %) | equal to Smilei's to 10⁻⁸ | −0.25 % (Smilei +0.016 %) |
| Largest relative change of the total energy | 3.0 × 10⁻⁷ to 3.8 × 10⁻⁷ per seed (Smilei 3.5 × 10⁻⁷) | 1.22 × 10⁻³ (Smilei 1.02 × 10⁻³); 256 cells 1.09 × 10⁻³ (Smilei 0.55 × 10⁻³) | 2.2 × 10⁻⁷ to 3.4 × 10⁻⁷ (Smilei 1.3 × 10⁻⁶) |

**A glitch in PIConGPU's derived charge density, in the thermal quasi-1D runs only.** In 21 of the 671 dumps of a Landau seed, the dumped electron charge density summed over the box is short by 1 × 10⁻⁴ to 8 × 10⁻⁴ of the electron charge; at every other dump it is exact (−9 × 10⁻¹⁶), and Gauss's law on the dumped E holds to 10⁻¹⁴. The `chargeConservation` plugin shows the same kind of spike at other steps, and both return to roundoff at the next output. Esirkepov deposition keeps any Gauss defect for the rest of a run, so a missed current or a lost particle would persist; neither does. The fields and the particles are therefore consistent, and the derived density (`FieldTmp`) occasionally leaves particles out. The cold two-stream runs (same layout, no motion in y) and the 2D runs (8 × 8 supercells) show none. The observables use E only, and the analysis uses the density at t = 0 only, so no number in this report is affected; whether the same frame handling could skip particles in the push was not tested (open items). The suspect is particles crossing the one-cell-high supercells of the quasi-1D layout; it is being reduced to a minimal setup for an `upstream` issue.

The initial kinetic energies differ as the loads do. PIConGPU's thermal momenta are the non-relativistic Maxwellian in momentum, and their energy is computed as (γ − 1)mc². Smilei draws the classical energy and sets |p| relativistically. That is an O(T) = 2.5 × 10⁻³ difference in the load (Setup). The two-stream energy drift at 256 cells does not halve in PIConGPU as it does in Smilei; that was not investigated (open items). In 2D, Smilei's t = 0 field energy exceeds PIConGPU's by 2.4 %: Smilei's `Uelm` scalars count extra columns (131/128) and include the constant field Smilei adds after its Poisson solve (`reports/crosscode-smilei-2d/`). PIConGPU's are ½ ΣE²ΔV over the cells.

### Two-stream: the codes agree to about 10⁻⁶

Pre-registered `growth_rate` (field energy, the 10⁻³ to 3 × 10⁻² window), and the same rule on |Ê₁|²:

| | Smilei | PIConGPU FP64 | PIConGPU FP32 | PIConGPU − Smilei | FP32 − FP64 |
|---|---|---|---|---|---|
| 128 cells, γ | 0.3456117 ± 0.0009 | 0.3456123 ± 0.0009 | 0.3456120 | +5.9 × 10⁻⁷ (+1.7 × 10⁻⁶ relative, 5 × 10⁻⁴ fit errors) | −3.2 × 10⁻⁷ |
| 128 cells, γ from \|Ê₁\|² | 0.3369876 | 0.3369876 | 0.3369885 | +6.9 × 10⁻⁸ | +8.6 × 10⁻⁷ |
| 256 cells, γ | 0.3459258 ± 0.0009 | 0.3459258 ± 0.0009 | 0.3459269 | +4.4 × 10⁻⁸ | +1.1 × 10⁻⁶ |
| 256 cells, γ from \|Ê₁\|² | 0.3370343 | 0.3370343 | 0.3370342 | −2.8 × 10⁻⁹ | −9.2 × 10⁻⁸ |

Both codes fit the same 164 (163) samples over t = 13.80 to 18.76 (13.83 to 18.76), and their saturation energies agree to 10⁻⁶. Every run is relative to the theory 0.3508758 by the common −1.5 % (−1.4 % at 256 cells) of WP-012, the fit window sampling a transient (#168).

**ParticleMesh against both.** ParticleMesh's WP-012 run gives 0.345904 at 128 cells, +8.4 × 10⁻⁴ from both codes. A post-hoc investigation of this package traced it to two things:

- **The window.** Fitted over the reference codes' window, ParticleMesh gives 0.345578, −1.0 × 10⁻⁴ from them. Its own window starts one output later, because its saturation peak falls one output later, and because log U bends over the window, a one-sample shift moves γ by 1 × 10⁻³ in any of the codes.
- **The initial condition.** ParticleMesh's U(t) differs from PIConGPU's by +1.3 % at t = 1 and −1.0 % from t = 10 on. The cause is ParticleMesh's half-step momentum kick at start-up (A2's centring: the loaded momenta are those at t = 0 and are pushed back to −Δt/2 in E⁰), while Smilei and PIConGPU take the loaded momenta as those at −Δt/2. With the kick removed, ParticleMesh's U(t) is a constant 1.0002 × PIConGPU's, and its γ equals both codes' to six digits. The deviation is linear in the kick (doubling and reversing it were checked), and a linear cold-fluid model predicts the late-time energy ratio 0.9893 against the measured 0.9896.

This is a choice of initial condition, not a defect, and the agreement between Smilei and PIConGPU holds because they share theirs. #226 proposes an option for ParticleMesh's convention; the threshold proposal depends on it.

### Landau damping (quasi-1D) over eight seeds

Pre-registered `damping_rate_ensemble` of `landau.toml` (WP-012), PIConGPU's eight seeds against WP-012's Smilei ensemble on the same seed numbers (the generators' random streams differ, so the seeds are independent draws in each code):

| | Smilei (WP-012) | PIConGPU FP64 | PIConGPU FP32 |
|---|---|---|---|
| ensemble γ ± jackknife | **0.1303 ± 0.0144** | **0.1360 ± 0.0131** | FP32LANDAU_G |
| maxima used, fit range | 6, t = 4.75 to 15.93 | 6, t = 4.77 to 16.02 | FP32LANDAU_M |
| relative to theory 0.15336 | −15.1 % | −11.3 % | FP32LANDAU_R |
| ω_r from the maxima (theory 1.41566) | 1.4046 | 1.3963 | FP32LANDAU_W |
| ensemble floor of \|⟨Ê₁⟩\|² | 5.9 × 10⁻¹⁰ | 3.5 × 10⁻¹⁰ | FP32LANDAU_F |
| per-seed γ at the ensemble's maxima, mean ± std | 0.1296 ± 0.0268 | 0.1343 ± 0.0413 | FP32LANDAU_S |

PIConGPU − Smilei (FP64): **+0.0057** (+4.4 %), 0.29 combined jackknife errors, 0.21 Smilei seed spreads, 0.14 PIConGPU seed spreads. FP32LANDAU_DIFF

Per seed (γ at the ensemble's maxima):

| seed | 20261101 | 20262101 | 20263101 | 20264101 | 20265101 | 20266101 | 20267101 | 20268101 |
|---|---|---|---|---|---|---|---|---|
| Smilei | 0.1035 | 0.1651 | 0.1221 | 0.1235 | 0.1472 | 0.1664 | 0.1004 | 0.1081 |
| PIConGPU FP64 | 0.1273 | 0.0799 | 0.1049 | 0.2118 | 0.1071 | 0.1690 | 0.1306 | 0.1437 |

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

The ensembles differ by 1.5 combined errors. They straddle the theory, Smilei's 0.6 errors above it and PIConGPU's 1.35 errors below. Each ensemble's fit stops at a different maximum (5 against 7), the variation the jackknife over four seeds samples poorly (WP-012's review, F1). The single-run rule averaged over the same seeds agrees to 0.3 errors. With four seeds per code this is a 1.5σ difference, and nothing here shows it to be a code difference; more seeds would decide it.

PIConGPU's `energy_ratio` is the background field, not a measurement of PIConGPU: the generator put the discrete Poisson field there. `energy_ratio_mode` is a ratio of noisy mode energies at the fit maxima, and its two codes differ by 2.3 errors of the means: PIConGPU is 1.6 errors above the Yee value and Smilei 1.9 below.

![Oblique ensembles, Smilei and PIConGPU](compare/landau-oblique-2d-smilei-vs-picongpu-fp64/landau-oblique-2d-ensemble-energy.png)
![PIConGPU oblique ensemble](picongpu-landau-oblique-2d-fp64-bc8fd6af/plots/ensemble-energy.png)
![Smilei oblique ensemble](smilei-landau-oblique-2d-bc8fd6af/plots/ensemble-energy.png)

### Compile and run times

Wall time inside the container (`container.txt`). This is not a throughput comparison: the codes ran on different hardware, the quasi-1D runs are far from PIConGPU's design point, and A5's fair comparison on one card is a non-goal of this package.

| Case | PIConGPU compile | PIConGPU run (per seed) | ns per mobile particle-step | Smilei run (per seed) | ns per mobile particle-step |
|---|---|---|---|---|---|
| two-stream, 128 cells, FP64 / FP32 | 71 s / 71 s | 103 s / 96 s | 77 / 71 | 17 s (16 threads, WP-012) | 12 |
| two-stream, 256 cells, FP64 / FP32 | 71 s / 71 s | 147 s / 126 s | 27 / 23 | 48 s | 9 |
| Landau, FP64 | 71 s | 1110 s to 1203 s | 129 to 140 | 79 s to 96 s (8 threads, WP-012) | 9 to 11 |
| Landau, FP32 | FP32LANDAU_C | FP32LANDAU_T | FP32LANDAU_N | | |
| oblique 2D, FP64 | 68 s | 327 s to 333 s | 5.7 to 5.9 | 559 s to 569 s (16 threads) | 9.8 to 10.0 |

In quasi-1D the GPU has little work to spread: 512 or 1024 cells, so 512 or 1024 threads per kernel. In addition, each openPMD dump costs about 0.6 s whatever its size (README, "PIConGPU specifics"), and the Landau case dumps 671 times. In 2D (16 384 cells) PIConGPU on the RTX 4080 SUPER in FP64 is 5.9 ns per particle-step with output, the compile once per setup.

## Review

REVIEW

## Verification status

- T-250 (unit conversion), T-251 (the oblique profile in both generators), T-252 (openPMD reader on files in PIConGPU's layout), T-253 (recipe pinned), T-254 (quasi-1D equivalence), T-255 (the initial field against a brute-force deposit) and T-256 (the oblique seed ensemble, WP-019's entries unchanged by hash) pass on the CPU: `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`. CI does not run this file.
- The reader's layout assumptions (meshes under `fields/`, F[y][x], spacing in axisLabels order, position x first) were checked on a run's file.
- The generated setups compiled and ran first time in the runner's image; PIConGPU's own `chargeConservation` and the analysis's Gauss check confirm the initial field.
- The oblique ensemble entry was committed (`d61a812`) before any run of the case in this package, and T-256 checks that WP-019's four entries are unchanged.
- The ParticleMesh two-stream investigation is post hoc: it reran WP-012's ParticleMesh case with the start-up kick scaled (0, 1, 2, −1), from scripts outside the repository, with no repository file changed; its baseline reproduced WP-012's stored series exactly. The proposal of an option is #226.
- Not verified: whether the charge-density glitch of the thermal quasi-1D runs reaches the push (only the derived density is shown to be affected); the cause of the larger two-stream energy drift at 256 cells in PIConGPU; the cell-centred-weight explanation of ParticleMesh's remaining 2 × 10⁻⁴ in U.

## Known limitations and open items

- **Seeds.** Four seeds per code in 2D and eight in 1D leave the Landau differences unresolved (0.3 and 1.5 combined errors). The jackknife over four seeds samples the variation of the fit's stop poorly (WP-012 review, F1).
- **Initial conditions are not identical across the codes**, by design of each code: the in-cell lattice (whole cell against 97.5 %), the thermal momenta (non-relativistic Maxwellian in momentum against Smilei's classical energy with relativistic |p|, O(T) = 2.5 × 10⁻³), Smilei's constant post-Poisson field, and ParticleMesh's half-step kick (#226). The deterministic two-stream case shows that the first two do not matter at 10⁻⁶ between Smilei and PIConGPU.
- **The charge-density glitch** of PIConGPU in the thermal quasi-1D layout (Results, checks): reduced to an `upstream` issue separately; not shown to touch the push.
- **Two-stream energy drift at 256 cells**: PIConGPU 1.09 × 10⁻³ against Smilei 0.55 × 10⁻³ (128 cells: 1.22 against 1.02 × 10⁻³). Not investigated; the growth rates agree to 10⁻⁷.
- **Throughput** is not compared (non-goal); the quasi-1D runs are I/O- and occupancy-bound in PIConGPU.
- **Precision of PIConGPU's scalars**: the per-component field energies are printed with 6 significant digits, which sets the 1.6 × 10⁻⁶ offset of the oblique `energy_ratio` from the Yee value; the totals have 16.
- **Weibel** is not run in PIConGPU (non-goal), and the 2D LWFA case is not written; its prerequisites are in `questions/wp-017-lwfa-case.md`.
- The proposed thresholds are in `questions/wp-017-a7-thresholds.md`, for the owner.
