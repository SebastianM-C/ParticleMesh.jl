# Cross-code reference runs

Reference numbers from other codes on the cases of the A5 benchmark set, matched to ParticleMesh's inputs through one case file per case. Stage 1 (WP-003): Smilei on the two 1D electrostatic cases. The 2D case core (WP-019): case schema 2 and Smilei on two 2D periodic cases with linear theory, `landau-oblique-2d` and `weibel-2d`.

```
julia +1.13 --project=scripts scripts/crosscode/smilei.jl prepare landau      # writes scripts/runs/smilei-landau-<hash>/
scripts/crosscode/run-smilei.sh scripts/runs/smilei-landau-<hash>              # outside the factory (podman)
julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/smilei-landau-<hash> [runner log]
julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl              # T-010, T-012, T-013, T-160, T-161, T-270 to T-275
```

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
| `analysis.jl` | Rate fits, mode amplitudes, and the 2D mode coefficient and projection on k (T-161, T-274) |
| `analyze.jl` | Observables, the standard plot set with JSON sidecars, and the manifest update of a finished run |
| `manifest.jl` | Run manifests (F3) |

## Case schema, version 1

A proposal; the spike's (WP-010) write-up may change it, and then the schema changes, not the spike. Units of A1 throughout: time 1/ω₀, length c/ω₀, velocity c, density n_c, temperature m_e c².

| Key | Meaning |
|---|---|
| `schema`, `name`, `description` | Version 1, the case id, one line |
| `[grid]` `length`, `cells`, `boundaries` | Box length, number of cells, `"periodic"` |
| `[time]` `dt`, `t_end` | Time step and end time; the run takes `round(t_end/dt)` steps |
| `[numerics]` `shape_order`, `current_deposition`, `filter_passes`, `precision`, `random_seed` | Shape order (2, quadratic, A2), deposition scheme, binomial filter passes, element type, seed of the velocity sampling |
| `[[species]]` `name`, `charge`, `mass`, `density`, `particles_per_cell`, `position`, `velocity`, `thermal_velocity`, `drift`, `mobile` | Charge in e, mass in m_e, density in n_c; `position` `"regular"`; `velocity` `"maxwellian"` (with `thermal_velocity` v_th = √(T/m)) or `"cold"`; drift velocity along x; `mobile = false` for immobile ions |
| `transverse_drift = [v_y, v_z]` | Optional (WP-014, for the Weibel case): the drift velocity across the resolved axis, in c; `[0, 0]` when absent. Not passed to Smilei: the schema-1 generator refuses it, and the 2D Weibel case is `weibel-2d` |
| `perturbation = { amplitude, mode }` | n(x) = density (1 + amplitude cos(2π mode x / length)); a negative amplitude is the opposite phase |
| `[diagnostics]` `scalars_every`, `fields_every`, `fields` | Output cadence in steps and the fields dumped |
| `[observables.<name>]` | `signal`, `method` and its parameters (fixed before the run), `theory` (the value, in ω_p units), `formula` (where it comes from) |

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
- `regular` places particles deterministically (inside 97.5 % of the cell); Maxwell–Jüttner momenta are random, seeded by `random_seed` plus the patch index. Only the positions are quiet.
- Patches: a power of 2 that divides the cells, more than 5 cells per patch at interpolation order 2.
- The step count is passed as `number_of_timesteps`, since Smilei truncates `simulation_time / timestep`.
- `solve_poisson = True` (the default): Smilei solves Poisson at t = 0 for the initial Ex.
- Ions are made immobile with `time_frozen` longer than the run; frozen species still deposit charge.

In 2D (`2Dcartesian`, schema 2), from the same source (`src/Profiles/Profile.cpp`, `src/Particles/ParticleCreator.cpp`, `src/Params/Params.cpp`, `src/Patch/VectorPatch.cpp`, `src/ElectroMagn/ElectroMagn2D.*`, `src/Diagnostic/DiagnosticScalar.cpp`, `src/Diagnostic/DiagnosticFields2D.cpp`, `src/Params/OpenPMDparams.cpp`):

- A non-separable profile is a Python function of (x, y) passed as `number_density`. Smilei first calls it with numpy arrays and uses the vectorized path when an array comes back, otherwise one point at a time; the generator writes it with `np.cos` and puts `import numpy as np` at the top of the namelist (the image has numpy, so `SMILEI_USE_NUMPY` is set at build). Profiles are evaluated at the cell centres ((i + ½)Δx, (j + ½)Δy).
- Patches: `number_of_patches` per axis, each a power of 2 that divides the cells of its axis, with more than 2·oversize + 1 = 5 cells per patch at order 2. The generator grows the axis with more cells per patch, keeps at least 8 per patch, caps the total at 16 and runs one thread per patch.
- `regular` placement with `regular_number = [n_x, n_y]`: the product must equal `particles_per_cell`, and the particles sit at (i + ½)·0.975 Δ/n per axis from the cell corner, x fastest. Without `regular_number` the count must be a perfect square.
- `mean_velocity = [v_x, v_y, v_z]` boosts the loaded momenta (Zenitani's volume transform); for a cold species it gives exactly p = γv with γ = (1 − v²)^(-1/2). The boost draws one uniform number per particle, so a cold drifting species also uses the random stream.
- The initial Poisson solve in 2D is the same conjugate-gradient solve as in 1D (`poisson_max_error = 1e-14`, at most 50 000 iterations). Smilei then shifts E by a constant, −½ (E(x_min, y_max) + E(x_max, y_min)) per component, taken at two corner points of the ghost-extended patches. The two points are mirror images through the origin under periodicity, so for a density even about the origin (the cases' cosine) E is odd there and the shift vanishes, up to the 0.975 placement offset. That argument assumes the ghost cells hold periodic copies at that point, which was not traced in the source. `analyze.jl` records the box mean of every dumped field at t = 0 as the check.
- The 2D Yee solver is stable for Δt < 1/√(1/Δx² + 1/Δy²) (`dtCFL`); the cases use 0.95 of it.
- `EM_boundary_conditions` and each species' `boundary_conditions` take one list per axis, `[["periodic"], ["periodic"]]`.
- `scalars.txt` holds per-component field energies `Uelm_Ex`, `Uelm_Ey`, `Uelm_Ez`, `Uelm_Bx_m`, `Uelm_By_m`, `Uelm_Bz_m`, each ½ ΔxΔy Σ F² over the patch interiors (the scalars use Smilei's `_m` copies of B). The electrostatic energy is `Uelm_Ex + Uelm_Ey`, the in-plane magnetic energy `Uelm_Bx_m + Uelm_By_m`.
- `DiagFields` can dump `Bx` or `Bx_m` (and likewise for y, z); the two differ by half a time step, and Smilei writes the openPMD `timeOffset` ½Δt for the `_m` names and 0 otherwise (`DiagnosticFields.cpp`). Which of them sits at the integer step was not traced further in the source. The cases dump `Bx`, `By`; a constant time offset does not change a rate fit.

## Reader schemas

- `read_scalars(dir)` returns `(; time, columns)`: the time column (1/ω₀) and a `Dict` from each column name of `scalars.txt` (`Utot`, `Ukin`, `Uelm`, `Uelm_Ex`, ...) to its values.
- `read_fields(dir; file = "Fields0.h5", names)` returns `(; iterations, time, x, fields)`: iteration numbers, times (from the `time` attribute), the positions of each field's points (openPMD `gridGlobalOffset + (i + position) gridSpacing`), and `fields[name]`, a points × outputs matrix. 1D dumps hold cells + 1 points; the last repeats the first for periodic boundaries.
- In 2D, `fields[name]` is a points_x × points_y × outputs array (x first) and `x[name]` the tuple `(x, y)` of per-axis positions, from the per-axis `gridSpacing`, `gridGlobalOffset` and `position`. Smilei writes each dataset in C order with dimensions [N_x + 1, N_y + 1] (`dataOrder = "C"`, `axisLabels = ["x", "y"]`), which HDF5.jl shows reversed; the reader transposes it. The last row and column repeat the first.
