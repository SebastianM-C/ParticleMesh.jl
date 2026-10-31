+++
id = "WP-003"
status = "parked"
goals = ["A1", "A5", "A7"]
depends_on = []
adds_dependencies = ["HDF5"]

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "518cf2ad3eb6dec7543f35463c93dd93a643a67d"
+++
# WP-003: cross-code reference runs with Smilei and PIConGPU

Parked until the factory side below exists and the engine core has landed. Both codes run in containers built from recipes in this repository; building the images and passing the GPU into them is the factory's side. The runner protocol is documented in the machine instructions and in `scripts/crosscode/README.md`, not in `CLAUDE.md`, so the package touches no code-owner path.

## Scope

The harness that produces the reference numbers decision A5 needs, from two external codes, through the GPU runner, as run manifests with the standard plot set. The package produces numbers and manifests; it does not set tolerances. The owner sets A5 and the A7 thresholds from its output.

## Deliverables

0. **Container recipes.** `scripts/crosscode/containers/`: one Containerfile per code, with every base image pinned by digest and every upstream source pinned to a release tag or commit, plus a `README.md` with the build command. Before writing one, check upstream for a maintained image or a documented container or Spack route, and build on it if it can be pinned. Smilei is built once: a CPU build with MPI, HDF5 and its Python interface, and a GPU build as a second target only for the throughput reference (see Non-goals). PIConGPU's container is a build environment rather than a binary, because its `.param` files are compiled per case: the CUDA toolkit for the RTX 4080 SUPER (compute capability 8.9), its C++ dependencies and its `pic-create` and `pic-build` tooling. The implementer cannot build images from the sandbox; the owner builds them from these recipes (Factory side), and the package's pull request says which recipe changes need a rebuild.
1. **Cases.** `scripts/crosscode/cases/`: one TOML file per benchmark case in normalized units (A1): the 1D3V set (Landau damping, two-stream instability, Weibel instability) and the matched 2D LWFA case of the validation plan, each with grid, time step, particles per cell, shape order, boundary conditions, laser or beam parameters, diagnostics and their cadence, and the scalar observables that the comparison uses. The case file is the single source; every deck is generated from it.
2. **Deck generators.** `scripts/crosscode/smilei.jl` and `scripts/crosscode/picongpu.jl` turn a case into a complete Smilei input (a Python namelist) and a complete PIConGPU input set (`.param` and `.cfg` files, the simulation directory layout its tooling expects, with the openPMD output plugin selected with its HDF5 backend so that the readers below apply), with the unit conversion from A1 done in exactly one function per code and tested against hand-computed values.
3. **Runner protocol.** A request to the `gpu-runner` session may name `smilei` or `picongpu` as the command family, with the generated input directory under `scripts/runs/` of the worktree, a tag and a timeout; documented in the "GPU" bullet of the machine instructions and in `scripts/crosscode/README.md`. The runner validates the command family and the directory the same way it validates `julia` commands. The wrapper runs each code in its image with podman, outside bubblewrap: the run directory mounted read-write, the rest of the worktree read-only, no network, and the GPU passed in only for GPU runs. For PIConGPU the compile step runs in the same container invocation, just before the run. Raw output stays in `scripts/runs/<tag>/`, which git ignores; only the reduced outputs of deliverable 5 are committed. The wrapper itself is the factory's side, listed below.
4. **Readers.** `scripts/crosscode/read.jl`: reads Smilei's HDF5 diagnostics and PIConGPU's openPMD output into the same in-memory tables, using `HDF5.jl` from the `scripts/` environment (added in this package's pull request, F2; openPMD through its HDF5 backend, no further dependency).
5. **Comparison and manifests.** `scripts/crosscode/compare.jl`: for each case and code, the standard plot set with JSON sidecars (F3), the scalar observables (growth rates, damping rates, field energies, maximum electron energy, dephasing length as the case defines), a run manifest per run with its inputs hashed and the digest of the image it ran in (F3), and an overlay plot per case across codes. A Markdown report in `reports/` per case, with the header `reports/README.md` prescribes, that states the numbers side by side and the observed spread, without judging them.
6. **ParticleMesh column.** The same comparison accepts ParticleMesh output once the engine exists; until then the column is marked absent, so the harness is ready for the core packages.

## Factory side (owner, outside the repository)

1. GPU access for containers: the NVIDIA container toolkit and a CDI specification, so that `podman run --device nvidia.com/gpu=all <image> nvidia-smi` lists the card.
2. The images, built from the recipes of deliverable 0 with network access, outside the sandbox, and tagged with the hash of their recipe.
3. The `gpu-run` wrapper gains the command families `smilei` and `picongpu`. It launches the image for the requested family with the mounts and network setting of deliverable 3, and writes the image digest into the log next to the finish line. The runner prompt names the families.

Until this exists, the package can finish everything except the runs themselves; the implementer writes `questions/<id>.md` when it reaches that point.

## Non-goals

Setting A5 numbers or A7 tolerances; multi-GPU; the ParticleMesh engine itself. A GPU build of Smilei is in scope only for the throughput reference of A7 (nanoseconds per particle-step on the same card); the correctness numbers come from CPU runs, and if the GPU build does not come together the report says so and the throughput column stays empty.

## Acceptance

- `T-010 unit conversion to Smilei [A1, WP-003]` and `T-011 unit conversion to PIConGPU [A1, WP-003]`: the generated decks for a case reproduce hand-computed values for density, laser amplitude, cell size and time step.
- `T-012 readers agree on synthetic files [WP-003]`: two small HDF5 files written by the test, one in Smilei's diagnostic layout and one in the openPMD layout, are read by their readers into tables with the same schema and the same values.
- `T-013 container recipes are pinned [WP-003]`: every `FROM` line in `scripts/crosscode/containers/` names a digest, and every upstream source is fetched at a tag or commit, never a branch.
- One 1D3V case run through the runner with each code (once the factory side exists), its run manifests and overlay plot in the PR, and the per-case report in `reports/`.

## Depends on

Goals A1, A5 (as the consumer of this package's output) and A7; factory setting F3; `docs/plan/landscape-and-requirements.md` section 6 and the validation plan in `overview.md` section 2.

## Budgets

The 1D3V cases run in minutes on the CPU or the local GPU; the 2D LWFA case is sized to the 16 GB card and may need a rented GPU for the full-resolution point, which the report says explicitly if so. One `HDF5.jl` dependency in `scripts/` only. The images are several GB each (a CUDA development image for PIConGPU), kept in podman's storage, never in the repository.
