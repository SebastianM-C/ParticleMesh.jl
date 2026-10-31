+++
id = "WP-003"
status = "proposed"
goals = ["A1", "A5", "A7"]
depends_on = []
adds_dependencies = ["HDF5"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "b4cade448159d5276774626df7f94299f8621395"
+++
# WP-003: cross-code reference runs, stage 1: Smilei on the 1D electrostatic cases

The first external reference numbers for decision A5, from Smilei on the two 1D cases that the spike WP-010 also runs (Landau damping, two-stream instability). Everything lives in `scripts/`, touches no engine code and needs no engine, so it runs in parallel with the spike. Stage 2 (PIConGPU, the 2D LWFA case, the GPU throughput reference and the comparison harness with a ParticleMesh column) is cut as its own package after the spike's write-up, when the shape of ParticleMesh's output is known; the notes on the PIConGPU container (a build environment with a per-case compile, openPMD output through HDF5) are in this file's history before this revision.

## Scope

A Smilei CPU container built from a recipe in this repository, one TOML case file per 1D case in normalized units (A1), a namelist generator with the unit conversion in one tested function, a reader for Smilei's HDF5 diagnostics, the two runs with the standard plot set and sidecars (F3), and a report that puts the measured damping and growth rates next to linear theory. The package produces numbers; it sets no tolerances (A7: thresholds are proposed from measured spread, and the owner accepts them by editing the goals).

## Deliverables

0. **Container recipe.** `scripts/crosscode/containers/smilei/Containerfile` with the base image pinned by digest and Smilei fetched at a release tag, a CPU build with MPI, HDF5 and the Python interface (happi), plus a `README.md` with the build command and the image's tag convention (the hash of the recipe). Check upstream for a maintained image first and build on it if it can be pinned. The implementer cannot build images from the sandbox; the owner builds it from the recipe (Factory side).
1. **Cases.** `scripts/crosscode/cases/landau.toml` and `two-stream.toml` in normalized units (A1): grid, time step, particles per cell, shape order, periodic boundaries, the initial perturbation or the beam parameters, the diagnostics and their cadence, and the scalar observables the comparison uses (the damping rate from the field-energy envelope, the growth rate from the linear phase), each with the linear-theory value and the formula it comes from referenced. The case file is a proposal for the schema the spike's `scripts/` runs read later; if the spike's write-up says the schema does not fit, the schema changes, not the spike.
2. **Namelist generator.** `scripts/crosscode/smilei.jl` turns a case into a complete Smilei namelist; the unit conversion from A1 is one function, tested against hand-computed values.
3. **Run script.** `scripts/crosscode/run-smilei.sh`: a plain `podman run` on the CPU with the generated input directory under `scripts/runs/<tag>/` mounted read-write, the rest read-only, no network and no GPU. The 1D cases run in minutes on the CPU, so the GPU runner and its command families are not involved at this stage. Raw output stays in `scripts/runs/`, which git ignores.
4. **Reader.** `scripts/crosscode/read.jl` reads Smilei's HDF5 diagnostics (field energies, the field on the grid at the saved times) into tables, using `HDF5.jl` from the `scripts/` environment (added in this PR, F2).
5. **Results.** For each case: the standard plot set with JSON sidecars (CairoMakie from `scripts/`), the scalar observables, and a run manifest with the inputs hashed, the commit and the image digest (F3). A Markdown report in `reports/crosscode-smilei-1d/` with its figures next to it and the header `reports/README.md` prescribes, stating the measured rates next to linear theory and the resolution and particle count used, without judging them.

## Factory side (owner, outside the repository)

Build the image from deliverable 0 with network access, outside the sandbox, and tag it with the hash of its recipe. Nothing else: no GPU passthrough and no runner change at this stage. Until the image exists the package can finish everything except the runs; the implementer writes `questions/<id>.md` when it reaches that point and opens the PR with the runs marked absent.

## Non-goals

PIConGPU; the 2D LWFA case; a GPU build of Smilei and the throughput reference; the `smilei` command family in the GPU runner; the comparison harness, the overlay plots and the ParticleMesh column; setting A5 numbers or A7 tolerances; multi-GPU; any engine code.

## Acceptance

- `T-010 unit conversion to Smilei [A1, WP-003]`: the generated namelist for a case reproduces hand-computed values for density, cell size, time step and the perturbation or beam parameters.
- `T-012 Smilei reader on a synthetic file [WP-003]`: a small HDF5 file written by the test in Smilei's diagnostic layout is read into a table with the documented schema and values.
- `T-013 container recipe is pinned [WP-003]`: the `FROM` line names a digest and Smilei is fetched at a tag or commit, never a branch.
- Both cases run through the container, their run manifests, plots and sidecars in the PR, and the report in `reports/`; or, if the image does not exist yet, the question file and the report with the runs marked absent.
- `julia +1.13 --project=scripts scripts/trace/build.jl --check` passes.

## Depends on

Nothing in the repository. Goals A1, A5 (as the consumer of its numbers) and A7; factory setting F3.

## Budgets

Minutes of CPU per case. One `HDF5.jl` dependency in `scripts/` only. The image is a few GB in podman's storage, never in the repository.
