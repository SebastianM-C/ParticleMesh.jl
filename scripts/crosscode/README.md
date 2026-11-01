# Cross-code reference runs

Reference numbers from other codes on the cases of the A5 benchmark set, matched to ParticleMesh's inputs through one case file per case. Stage 1 (WP-003): Smilei on the two 1D electrostatic cases; stage 1b (WP-012): Landau over a seed ensemble in both codes, two-stream as a resolution pair (`two-stream.toml`, `two-stream-256.toml`). The 2D case core (WP-019): case schema 2 and Smilei on two 2D periodic cases with linear theory, `landau-oblique-2d` and `weibel-2d`.

```
julia +1.13 --project=scripts scripts/crosscode/smilei.jl prepare landau      # writes scripts/runs/smilei-landau-<hash>/
scripts/crosscode/run-smilei.sh scripts/runs/smilei-landau-<hash>              # outside the factory (podman)
julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/smilei-landau-<hash> [runner log]
julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl              # T-010 to T-161, T-180 to T-188, T-253, T-270 to T-275
julia +1.13 --project=scripts -t 16 scripts/es1d/crosscode.jl run landau       # the ParticleMesh column, one run per seed
julia +1.13 --project=scripts scripts/es1d/crosscode.jl compare <pm run> <smilei run>
```

A seeded case (a `seeds` list) prepares one `seed-<seed>/` subdirectory per seed, each with its namelist and a copy of the case, and one `run.sh` that runs them in turn; `analyze.jl` then analyzes every seed directory as a single run and writes the ensemble observable, its plots and sidecars into the top directory.

Inside the factory the run goes through the machine's runner (`command: smilei scripts/runs/<tag>`, `gpu: no`), which executes the same `run.sh` in the same container with the same mounts.

## Files

| File | Role |
|---|---|
| `cases/<case>.toml` | The case in the units of A1, with its observables and their linear-theory values |
| `theory.jl` | Dispersion relations the theory values come from, with the 2D Weibel relation derived in its header (T-160, T-273) |
| `smilei.jl` | `load_case` and `case_view` (both schemas), `smilei_parameters` (the unit conversion, T-010, T-271), `namelist`, `prepare_run` |
| `test/namelist-hashes.toml` | Hashes of the schema-1 namelists, recorded before schema 2 (T-270) |
| `containers/smilei.Containerfile` | The Smilei image (T-013); build and tag convention in `containers/README.md` |
| `run-smilei.sh` | Plain `podman run` of a prepared run directory, for runs outside the factory |
| `read.jl` | Readers for `scalars.txt` and `Fields<n>.h5`, 1D and 2D (T-012, T-275) |
| `analysis.jl` | Rate fits, mode amplitudes, the complex mode coefficient (1D, and 2D with the projection on k) (T-161, T-274); `checked_fit` refuses a fit on too few points (T-186) |
| `ensemble.jl` | The ensemble fit with the jackknife over seeds, per-seed fits at the ensemble's maxima, `aggregate_seeds`, the seeds' scatter (T-180, T-183, T-187) |
| `ensemble_report.jl` | The ensemble observable of either code, its plots, sidecars and `observables.json`; the two-code overlay |
| `ensemble_bias.jl` | Post hoc: the bias of the ensemble rule on synthetic ensembles at a run's measured noise |
| `analyze.jl` | Observables, the standard plot set with JSON sidecars, and the manifest update of a finished run |
| `manifest.jl` | Run manifests (F3); `image_from_header` reads the bare image id from the runner's log (T-184) |

## Case schema, version 1

A proposal; the spike's (WP-010) write-up may change it, and then the schema changes, not the spike. Units of A1 throughout: time 1/ω₀, length c/ω₀, velocity c, density n_c, temperature m_e c².

| Key | Meaning |
|---|---|
| `schema`, `name`, `description` | Version 1, the case id, one line |
| `[grid]` `length`, `cells`, `boundaries` | Box length, number of cells, `"periodic"` |
| `[time]` `dt`, `t_end` | Time step and end time; the run takes `round(t_end/dt)` steps |
| `[numerics]` `shape_order`, `current_deposition`, `filter_passes`, `precision`, `random_seed` | Shape order (2, quadratic, A2), deposition scheme, binomial filter passes, element type, seed of the velocity sampling |
| `[numerics]` `seeds = [...]` | Optional (WP-012), in place of `random_seed`: one run per seed, integers in [0, 2³¹), distinct. Smilei's `random_seed`, and the Philox key of ParticleMesh's velocity draws. Smilei seeds patch p with `random_seed + p`, so seeds closer than the patch count share streams (T-181 checks the spacing). Without the list the single `random_seed` applies |
| `[[species]]` `name`, `charge`, `mass`, `density`, `particles_per_cell`, `position`, `velocity`, `thermal_velocity`, `drift`, `mobile` | Charge in e, mass in m_e, density in n_c; `position` `"regular"`; `velocity` `"maxwellian"` (with `thermal_velocity` v_th = √(T/m)) or `"cold"`; drift velocity along x; `mobile = false` for immobile ions |
| `transverse_drift = [v_y, v_z]` | Optional (WP-014, for the Weibel case): the drift velocity across the resolved axis, in c; `[0, 0]` when absent. Not passed to Smilei: the schema-1 generator refuses it, and the 2D Weibel case is `weibel-2d` |
| `perturbation = { amplitude, mode }` | n(x) = density (1 + amplitude cos(2π mode x / length)); a negative amplitude is the opposite phase |
| `[diagnostics]` `scalars_every`, `fields_every`, `fields` | Output cadence in steps and the fields dumped |
| `[observables.<name>]` | `signal`, `method` and its parameters (fixed before the run), `theory` (the value, in ω_p units), `formula` (where it comes from) |
| `[observables.<name>]` with `method = "ensemble_peak_envelope"` | A seeded case's ensemble observable: `damping_from_peaks` on \|⟨Ê⟩\|², the seed mean of the complex mode coefficient of `signal`, with `fit_start` and `floor_factor`; `minimum_maxima`, `stderr` (`"jackknife"` over seeds), `per_seed` (`"at_ensemble_peaks"`: each seed's γ from its own \|Ê_s\|² at the ensemble's maxima), `theory_from` (the observable that holds the theory value) |

## Case schema, version 2

A 2D periodic problem (`schema = 2`, WP-019). A version check in `load_case` selects the schema: a schema-1 file keeps its meaning and is a 1D grid (`case_view` gives both schemas per axis). Keys and units as in version 1, except:

| Key | Meaning |
|---|---|
| `[grid]` `length = [L_x, L_y]`, `cells = [N_x, N_y]`, `boundaries = ["periodic", "periodic"]` | Per axis; only periodic boundaries |
| `[wavevector]` `mode = [m_x, m_y]`, `magnitude`, `angle` | The perturbed mode, stated: \|k\| in ω₀/c and atan(k_y, k_x) in rad. `load_case` checks that k_d L_d/2π is the integer m_d on every axis, i.e. that the mode is periodic in the box |
| `particles_per_cell = [n_x, n_y]` | Per species, per axis: a regular n_x × n_y lattice per cell, matching the engine's product lattice; the count per cell is the product |
| `drift = [v_x, v_y, v_z]` | Per species, the drift velocity in c, all three components (a beam along the collapsed axis z is `[0, 0, v]`); `[0, 0, 0]` when absent. Schema 1's scalar `drift` and `transverse_drift` are refused in schema 2 |
| `perturbation = { amplitude, mode = [m_x, m_y] }` | n(x, y) = density (1 + amplitude cos(2π (m_x x/L_x + m_y y/L_y))); `mode` must be integers and equal the `[wavevector]` mode |
| `[observables.<name>]` | As in version 1, with `signal` one of `mode_projection` (the mode coefficient of an in-plane vector, `fields = [a, b]`, along k̂ for `component = "parallel"` or along ẑ × k̂ for `"perpendicular"`), `scalar_sum` and `scalar_ratio` (of the `columns` of `scalars.txt`), `mode_ratio` (\|â\|²/\|b̂\|² of two fields); `method` one of `peak_envelope`, `exponential_window` (as in version 1), `first_output`, `median_over_fit` (at the maxima used by the fit of `fit_of`) |

Validation rejects a non-integer or wrong-length mode, a boundary that is not periodic, a box in which the stated wavevector is not periodic, per-axis keys of the wrong length, and a drift of c or more. The `seeds` field is WP-012's and is not defined here; when WP-012 merges, schema 2 keeps it.

## Smilei specifics the generator handles

From the Smilei v5.1 source (`src/Python/pyprofiles.py`, `src/Params/Params.cpp`, `src/Particles/ParticleCreator.cpp`):

- `cosine(base, xamplitude, ...)` takes an absolute amplitude and defaults to `xnumber = 2`; the generator passes `xamplitude = amplitude × density` and `xnumber` explicitly.
- Profiles are sampled at cell centres and all particles of a cell share its weight, so the loaded perturbation is piecewise constant per cell.
- `regular` places particles deterministically (inside 97.5 % of the cell); momenta of `maxwell-juettner` are random, seeded by `random_seed` plus the patch index. Only the positions are quiet. Below T = 0.1 m_e c², `maxwell-juettner` samples the kinetic energy from the classical Maxwell–Boltzmann energy distribution and sets |p| = √((1 + E)² − 1) (`ParticleCreator::maxwellJuttner`), so the cases (T = 2.5 × 10⁻³) load that, not a Maxwell–Jüttner distribution.
- Patches: a power of 2 that divides the cells, more than 5 cells per patch at interpolation order 2.
- Smilei runs n_time = (int)(simulation_time/timestep) steps (`Params.cpp`); `number_of_timesteps` only sets simulation_time = n·Δt when `simulation_time` is absent (`pyinit.py`), and n·Δt/Δt can truncate to n − 1. The 1D generator passes `number_of_timesteps`, and its cases keep their counts (2681 and 5253, checked); the 2D generator also passes simulation_time = (n + ½)Δt, which truncates to n (T-271). `landau-oblique-2d`'s run predates that and ran 3390 of its 3391 steps.
- `solve_poisson = True` (the default): Smilei solves Poisson at t = 0 for the initial Ex.
- Ions are made immobile with `time_frozen` longer than the run; frozen species still deposit charge.

In 2D (`2Dcartesian`, schema 2), from the same source (`src/Profiles/Profile.cpp`, `src/Particles/ParticleCreator.cpp`, `src/Params/Params.cpp`, `src/Patch/VectorPatch.cpp`, `src/ElectroMagn/ElectroMagn2D.*`, `src/Diagnostic/DiagnosticScalar.cpp`, `src/Diagnostic/DiagnosticFields2D.cpp`, `src/Params/OpenPMDparams.cpp`):

- A non-separable profile is a Python function of (x, y) passed as `number_density`. Smilei first calls it with numpy arrays and uses the vectorized path when an array comes back, otherwise one point at a time; the generator writes it with `np.cos` and puts `import numpy as np` at the top of the namelist (the image has numpy, so `SMILEI_USE_NUMPY` is set at build). Profiles are evaluated at the cell centres ((i + ½)Δx, (j + ½)Δy).
- Dual-axis positions in the dumps: Smilei writes the openPMD `position = 0.5` on a dual axis (`DiagnosticFields.cpp`, stagger = 0.5·isDual) but the data starts at local index `oversize` (`DiagnosticFields2D.cpp`), which is x = −Δ/2 on a dual axis. So point i of a dual axis is at (i − ½)Δ, and the 2D reader places it there; with that, Gauss's law on the Landau dump at t = 0 closes to 6.5 × 10⁻¹² (max |ρ| = 10⁻²), against 8.8 × 10⁻⁴ with (i + ½)Δ. The 1D reader still uses (i + ½)Δ (T-012), which the same source makes wrong by one cell (issue #161).
- Patches: `number_of_patches` per axis, each a power of 2 that divides the cells of its axis, with more than 2·oversize + 1 = 5 cells per patch at order 2. The generator grows the axis with more cells per patch, keeps at least 8 per patch, caps the total at 16 and runs one thread per patch.
- `regular` placement with `regular_number = [n_x, n_y]`: the product must equal `particles_per_cell`, and the particles sit at (i + ½)·0.975 Δ/n per axis from the cell corner, x fastest. Without `regular_number` the count must be a perfect square.
- `mean_velocity = [v_x, v_y, v_z]` boosts the loaded momenta (Zenitani's volume transform); for a cold species it gives exactly p = γv with γ = (1 − v²)^(-1/2). The boost draws one uniform number per particle, so a cold drifting species also uses the random stream.
- The initial Poisson solve in 2D is the same conjugate-gradient solve as in 1D (`poisson_max_error = 1e-14`, at most 50 000 iterations). Smilei then adds a constant to each component, −½ (E(x_min, y_max) + E(x_max, y_min)), read at two corner points of the ghost-extended patches, whatever the boundaries (its own comment: "This correction is always done, independantly of the periodicity. Is this correct ?"). Measured in `smilei-landau-oblique-2d-c8709f77`: at t = 0, Ey has a box mean of 1.7 × 10⁻⁵ (3.8 % of its wave amplitude) and Ex one of 1.6 × 10⁻⁶. A uniform E in a periodic box is a k = 0 plasma oscillation that is not in the case's initial condition. It does not project on k, but it enters the field-energy scalars: with the box mean removed, the t = 0 Ex/Ey energy ratio equals the Yee value to roundoff (`reports/crosscode-smilei-2d/`). Symmetry would make the shift vanish for a density even about the origin if the ghost cells held periodic copies at both points. A minimal run (issue #157) finds the Ey offset equal to −½ E_y at the (x_max, y_min) corner alone, as if the (x_min, y_max) ghost point were still empty when read; that is inferred from the numbers, not traced in the exchange code. A density odd about the origin makes the offset of the order of the field amplitude; the schema's cosine form keeps it at a few per cent of it. `analyze.jl` records the box mean of every dumped field at t = 0.
- The 2D Yee solver is stable for Δt < 1/√(1/Δx² + 1/Δy²) (`dtCFL`); the cases use 0.95 of it.
- `EM_boundary_conditions` and each species' `boundary_conditions` take one list per axis, `[["periodic"], ["periodic"]]`.
- `scalars.txt` holds per-component field energies `Uelm_Ex`, `Uelm_Ey`, `Uelm_Ez`, `Uelm_Bx_m`, `Uelm_By_m`, `Uelm_Bz_m`, with Smilei's `_m` copies of B. They are not exactly ½ ΔxΔy Σ F² over the distinct points of a dump: at t = 0 of `landau-oblique-2d` (4 patches per axis) both Uelm_Ex and Uelm_Ey exceed that sum by 131/128, as if three extra columns were counted (measured, not traced in the source). Ratios of the scalars are unaffected; absolute field energies compared with another code are not. The electrostatic energy is `Uelm_Ex + Uelm_Ey`, the in-plane magnetic energy `Uelm_Bx_m + Uelm_By_m`.
- `DiagFields` can dump `Bx` or `Bx_m` (and likewise for y, z); the two differ by half a time step, and Smilei writes the openPMD `timeOffset` ½Δt for the `_m` names and 0 otherwise (`DiagnosticFields.cpp`). Which of them sits at the integer step was not traced further in the source. The cases dump `Bx`, `By`; a constant time offset does not change a rate fit.

## Reader schemas

- `read_scalars(dir)` returns `(; time, columns)`: the time column (1/ω₀) and a `Dict` from each column name of `scalars.txt` (`Utot`, `Ukin`, `Uelm`, `Uelm_Ex`, ...) to its values.
- `read_fields(dir; file = "Fields0.h5", names)` returns `(; iterations, time, x, fields)`: iteration numbers, times (from the `time` attribute), the positions of each field's points (openPMD `gridGlobalOffset + (i + position) gridSpacing`), and `fields[name]`, a points × outputs matrix. 1D dumps hold cells + 1 points; the last repeats the first for periodic boundaries.
- In 2D, `fields[name]` is a points_x × points_y × outputs array (x first) and `x[name]` the tuple `(x, y)` of per-axis positions, gridGlobalOffset + (i − position) gridSpacing, so a dual axis starts at −Δ/2 (Smilei specifics above). Smilei writes each dataset in C order with dimensions [N_x + 1, N_y + 1] (`dataOrder = "C"`, `axisLabels = ["x", "y"]`), which HDF5.jl shows reversed; the reader transposes it. The last row and column repeat the first.
