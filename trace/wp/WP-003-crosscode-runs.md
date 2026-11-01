+++
id = "WP-003"
status = "done"
goals = ["A1", "A5", "A7"]
depends_on = []
adds_dependencies = ["HDF5", "SpecialFunctions"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
+++
# WP-003: cross-code reference runs, stage 1: Smilei on the 1D electrostatic cases

The first external reference numbers for decision A5, from Smilei on the two 1D cases that the spike WP-010 also runs (Landau damping, two-stream instability). Everything lives in `scripts/`, touches no engine code and needs no engine, so it runs in parallel with the spike; this does not conflict with A0's rule of small sequential spikes, which is about packages written against an engine specification, and this one is a scripts-only reference run. PIConGPU, the 2D LWFA case, the GPU throughput reference and the comparison harness with a ParticleMesh column are cross-code stage 2; WP-017 is its first package.

## Scope

A Smilei CPU container built from a recipe in this repository, one TOML case file per 1D case in normalized units (A1), a namelist generator with the unit conversion in one tested function, a reader for Smilei's HDF5 diagnostics, the two runs with the standard plot set and sidecars (F3), and a report that puts the measured damping and growth rates next to linear theory. The package produces numbers; it sets no tolerances (A7: thresholds are proposed from measured spread, and the owner accepts them by editing the goals).

## Deliverables

0. **Container recipe.** `scripts/crosscode/containers/smilei.Containerfile` (build context `scripts/crosscode/containers/`, extra files in `scripts/crosscode/containers/smilei/`, the layout the machine's runner builds from) with the base image pinned by digest and Smilei fetched at a release tag, a CPU build with MPI, HDF5 and the Python interface (happi), plus a `README.md` with the build command and the image's tag convention (the hash of the recipe). Check upstream for a maintained image first and build on it if it can be pinned. The implementer cannot build images from the sandbox; the owner builds it from the recipe (Factory side).
1. **Cases.** `scripts/crosscode/cases/landau.toml` and `two-stream.toml` in normalized units (A1): grid, time step, particles per cell, shape order, periodic boundaries, the initial perturbation or the beam parameters, the diagnostics and their cadence, and the scalar observables the comparison uses (the damping rate from the field-energy envelope, the growth rate from the linear phase), each with the linear-theory value and the formula it comes from referenced. The case file is a proposal for the schema the spike's `scripts/` runs read later; if the spike's write-up says the schema does not fit, the schema changes, not the spike.
2. **Namelist generator.** `scripts/crosscode/smilei.jl` turns a case into a complete Smilei namelist; the unit conversion from A1 is one function, tested against hand-computed values.
3. **Run script.** The generator writes `scripts/runs/<tag>/run.sh`, which the container executes with the run directory mounted read-write at `/run-dir` and the repository read-only at `/worktree`, no network and no GPU. Inside the factory the machine's runner executes it (`command: smilei scripts/runs/<tag>`, `gpu: no`), since a sandboxed session cannot run podman; outside it, `scripts/crosscode/run-smilei.sh` is the plain `podman run` with the same mounts. The 1D cases run in minutes on the CPU; no GPU build is involved at this stage. Raw output stays in `scripts/runs/`, which git ignores.
4. **Reader.** `scripts/crosscode/read.jl` reads Smilei's HDF5 diagnostics (field energies, the field on the grid at the saved times) into tables, using `HDF5.jl` from the `scripts/` environment (added in this PR, F2).
5. **Results.** For each case: the standard plot set with JSON sidecars (CairoMakie from `scripts/`), the scalar observables, and a run manifest with the inputs hashed, the commit and the image digest (F3). A Markdown report in `reports/crosscode-smilei-1d/` with its figures next to it and the header `reports/README.md` prescribes, stating the measured rates next to linear theory and the resolution and particle count used, without judging them.

## Factory side (owner, outside the repository)

Build the image from deliverable 0 with network access, outside the sandbox, and tag it with the hash of its recipe. Nothing else: no GPU passthrough and no runner change at this stage. Until the image exists the package can finish everything except the runs; the implementer writes `questions/<id>.md` when it reaches that point and opens the PR with the runs marked absent.

## Non-goals

PIConGPU; the Weibel case, which A5 counts in the 1D set and which waits for stage 2; the 2D LWFA case; a GPU build of Smilei and the throughput reference; changes to the machine's runner; the comparison harness, the overlay plots and the ParticleMesh column; setting A5 numbers or A7 tolerances; multi-GPU; any engine code.

## Acceptance

- `T-010 unit conversion to Smilei [A1, WP-003]`: the generated namelist for a case reproduces hand-computed values for density, cell size, time step and the perturbation or beam parameters.
- `T-012 Smilei reader on a synthetic file [WP-003]`: a small HDF5 file written by the test in Smilei's diagnostic layout is read into a table with the documented schema and values.
- `T-013 container recipe is pinned [WP-003]`: the `FROM` line names a digest and Smilei is fetched at a tag or commit, never a branch.
- Both cases run through the container, their run manifests, plots and sidecars in the PR, and the report in `reports/`; or, if the image does not exist yet, the question file and the report with the runs marked absent.
- The trace check passes.

## Depends on

Nothing in the repository. Goals A1, A5 (as the consumer of its numbers) and A7; factory setting F3. The case files stay consistent with the validation plan in `docs/plan/overview.md` section 1 and the requirements in `docs/plan/landscape-and-requirements.md` section 6.

## Budgets

Minutes of CPU per case. Two dependencies in `scripts/` only: `HDF5.jl` and `SpecialFunctions.jl` (`erfcx`, for the Faddeeva function of the Landau theory). The image is a few GB in podman's storage, never in the repository.
