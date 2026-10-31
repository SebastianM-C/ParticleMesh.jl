# WP-003: cross-code reference runs with Smilei and PIConGPU

Status: proposed. Depends on WP-001 (run records). Tier 2 (touches the runner protocol in the contract; adds dependencies through the package-manager).

## Scope

The harness that produces the reference numbers decision A5 needs, from two external codes, through the GPU runner, as run records with the standard plot set. The package produces numbers and records; it does not set tolerances. The owner sets A5 and the A7 thresholds from its output.

## Deliverables

1. **Cases.** `scripts/crosscode/cases/`: one TOML file per benchmark case in normalized units (A1): the 1D3V set (Landau damping, two-stream instability, Weibel instability) and the matched 2D LWFA case of the validation plan, each with grid, time step, particles per cell, shape order, boundary conditions, laser or beam parameters, diagnostics and their cadence, and the scalar observables that the comparison uses. The case file is the single source; every deck is generated from it.
2. **Deck generators.** `scripts/crosscode/smilei.jl` and `scripts/crosscode/picongpu.jl` turn a case into a complete Smilei input (a Python namelist) and a complete PIConGPU input set (`.param` and `.cfg` files, the simulation directory layout its tooling expects, with the openPMD output plugin selected with its HDF5 backend so that the readers below apply), with the unit conversion from A1 done in exactly one function per code and tested against hand-computed values.
3. **Runner protocol.** A request to the `gpu-runner` session may name `smilei` or `picongpu` as the command family, with the generated input directory under `scripts/runs/` of the worktree, a tag and a timeout; documented in the "GPU" bullet of the machine instructions and in `scripts/crosscode/README.md`. The runner validates the command family and the directory the same way it validates `julia` commands. How the wrapper launches each code (a native build for Smilei, a container for PIConGPU) is the factory's side, listed below.
4. **Readers.** `scripts/crosscode/read.jl`: reads Smilei's HDF5 diagnostics and PIConGPU's openPMD output into the same in-memory tables, using `HDF5.jl` from the `scripts/` environment (one package request to the package-manager; openPMD through its HDF5 backend, no further dependency).
5. **Comparison and records.** `scripts/crosscode/compare.jl`: for each case and code, the standard plot set with JSON sidecars (A13), the scalar observables (growth rates, damping rates, field energies, maximum electron energy, dephasing length as the case defines), a `RUN-` record per run with inputs hashed, and an overlay plot per case across codes. A Markdown report in `reports/` per case, with the header `reports/README.md` prescribes, that states the numbers side by side and the observed spread, without judging them.
6. **ParticleMesh column.** The same comparison accepts ParticleMesh output once the engine exists; until then the column is marked absent, so the harness is ready for the core packages.

## Factory side (owner, outside the repository)

The `gpu-run` wrapper gains the two command families: Smilei built from source into the factory's area (the build needs the network once: `--net`), PIConGPU installed by its documented route, chosen when the package starts (a container launched by the wrapper directly rather than inside bubblewrap, or a Spack build if containers prove awkward next to the sandbox). The runner prompt names the families. Until this exists, the package can finish everything except the runs themselves; the implementer writes `questions/<id>.md` when it reaches that point.

## Non-goals

Setting A5 numbers or A7 tolerances; multi-GPU; the ParticleMesh engine itself. A GPU build of Smilei is in scope only for the throughput reference of A7 (nanoseconds per particle-step on the same card); the correctness numbers come from CPU runs, and if the GPU build does not come together the report says so and the throughput column stays empty.

## Acceptance

- `T-010 unit conversion to Smilei [R-nnn]` and `T-011 unit conversion to PIConGPU [R-nnn]`: the generated decks for a case reproduce hand-computed values for density, laser amplitude, cell size and time step.
- `T-012 readers agree on synthetic files [R-nnn]`: two small HDF5 files written by the test, one in Smilei's diagnostic layout and one in the openPMD layout, are read by their readers into tables with the same schema and the same values.
- One 1D3V case run through the runner with each code (once the factory side exists), its `RUN-` records and overlay plot in the PR, and the per-case report in `reports/`.
- The requirement ids above are the validation and cross-code requirements of landscape section 6, as transcribed by WP-001.

## Depends on

WP-001; decisions A1, A5 (as the consumer of this package's output), A7, A13; `docs/plan/landscape-and-requirements.md` section 6 and the validation plan in `overview.md` section 2.

## Budgets

The 1D3V cases run in minutes on the CPU or the local GPU; the 2D LWFA case is sized to the 16 GB card and may need a rented GPU for the full-resolution point, which the report says explicitly if so. One `HDF5.jl` dependency in `scripts/` only.
