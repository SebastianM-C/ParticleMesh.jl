# Cross-code reference runs

Reference numbers from other codes on the cases of the A5 benchmark set, matched to ParticleMesh's inputs through one case file per case. Stage 1 (WP-003): Smilei on the two 1D electrostatic cases; stage 1b (WP-012): Landau over a seed ensemble in both codes, two-stream as a resolution pair (`two-stream.toml`, `two-stream-256.toml`).

```
julia +1.13 --project=scripts scripts/crosscode/smilei.jl prepare landau      # writes scripts/runs/smilei-landau-<hash>/
scripts/crosscode/run-smilei.sh scripts/runs/smilei-landau-<hash>              # outside the factory (podman)
julia +1.13 --project=scripts scripts/crosscode/analyze.jl scripts/runs/smilei-landau-<hash> [runner log]
julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl              # T-010 to T-161, T-180 to T-186
julia +1.13 --project=scripts -t 16 scripts/es1d/crosscode.jl run landau       # the ParticleMesh column, one run per seed
julia +1.13 --project=scripts scripts/es1d/crosscode.jl compare <pm run> <smilei run>
```

A seeded case (a `seeds` list) prepares one `seed-<seed>/` subdirectory per seed, each with its namelist and a copy of the case, and one `run.sh` that runs them in turn; `analyze.jl` then analyzes every seed directory as a single run and writes the ensemble observable, its plots and sidecars into the top directory.

Inside the factory the run goes through the machine's runner (`command: smilei scripts/runs/<tag>`, `gpu: no`), which executes the same `run.sh` in the same container with the same mounts.

## Files

| File | Role |
|---|---|
| `cases/<case>.toml` | The case in the units of A1, with its observables and their linear-theory values |
| `theory.jl` | Dispersion relations the theory values come from (tested by T-160) |
| `smilei.jl` | `smilei_parameters` (the unit conversion, T-010), `namelist`, `prepare_run` |
| `containers/smilei.Containerfile` | The Smilei image (T-013); build and tag convention in `containers/README.md` |
| `run-smilei.sh` | Plain `podman run` of a prepared run directory, for runs outside the factory |
| `read.jl` | Readers for `scalars.txt` and `Fields<n>.h5` (T-012) |
| `analysis.jl` | Rate fits and mode amplitudes (T-161); `checked_fit` refuses a fit on too few points (T-186) |
| `ensemble.jl` | Complex mode coefficients, the ensemble fit with the jackknife over seeds, per-seed fits at the ensemble's maxima, `aggregate_seeds` (T-180, T-183) |
| `ensemble_report.jl` | The ensemble observable of either code, its plots, sidecars and `observables.json`; the two-code overlay |
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
| `transverse_drift = [v_y, v_z]` | Optional (WP-014, for the Weibel case): the drift velocity across the resolved axis, in c; `[0, 0]` when absent. Not yet passed to Smilei |
| `perturbation = { amplitude, mode }` | n(x) = density (1 + amplitude cos(2π mode x / length)); a negative amplitude is the opposite phase |
| `[diagnostics]` `scalars_every`, `fields_every`, `fields` | Output cadence in steps and the fields dumped |
| `[observables.<name>]` | `signal`, `method` and its parameters (fixed before the run), `theory` (the value, in ω_p units), `formula` (where it comes from) |
| `[observables.<name>]` with `method = "ensemble_peak_envelope"` | A seeded case's ensemble observable: `damping_from_peaks` on \|⟨Ê⟩\|², the seed mean of the complex mode coefficient of `signal`, with `fit_start` and `floor_factor`; `minimum_maxima`, `stderr` (`"jackknife"` over seeds), `per_seed` (`"at_ensemble_peaks"`: each seed's γ from its own \|Ê_s\|² at the ensemble's maxima), `theory_from` (the observable that holds the theory value) |

## Smilei specifics the generator handles

From the Smilei v5.1 source (`src/Python/pyprofiles.py`, `src/Params/Params.cpp`, `src/Particles/ParticleCreator.cpp`):

- `cosine(base, xamplitude, ...)` takes an absolute amplitude and defaults to `xnumber = 2`; the generator passes `xamplitude = amplitude × density` and `xnumber` explicitly.
- Profiles are sampled at cell centres and all particles of a cell share its weight, so the loaded perturbation is piecewise constant per cell.
- `regular` places particles deterministically (inside 97.5 % of the cell); Maxwell–Jüttner momenta are random, seeded by `random_seed` plus the patch index. Only the positions are quiet.
- Patches: a power of 2 that divides the cells, more than 5 cells per patch at interpolation order 2.
- The step count is passed as `number_of_timesteps`, since Smilei truncates `simulation_time / timestep`.
- `solve_poisson = True` (the default): Smilei solves Poisson at t = 0 for the initial Ex.
- Ions are made immobile with `time_frozen` longer than the run; frozen species still deposit charge.

## Reader schemas

- `read_scalars(dir)` returns `(; time, columns)`: the time column (1/ω₀) and a `Dict` from each column name of `scalars.txt` (`Utot`, `Ukin`, `Uelm`, `Uelm_Ex`, ...) to its values.
- `read_fields(dir; file = "Fields0.h5", names)` returns `(; iterations, time, x, fields)`: iteration numbers, times (from the `time` attribute), the positions of each field's points (openPMD `gridGlobalOffset + (i + position) gridSpacing`), and `fields[name]`, a points × outputs matrix. 1D dumps hold cells + 1 points; the last repeats the first for periodic boundaries.
