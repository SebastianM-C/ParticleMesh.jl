+++
id = "WP-017"
status = "proposed"
goals = ["A1", "A5", "A6", "A7"]
depends_on = ["WP-003", "WP-012", "WP-019"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
+++
# WP-017: cross-code stage 2a: PIConGPU on the GPU runner, the 1D set as quasi-1D runs, and the oblique 2D Landau case over seeds in both codes

A5 asks for verification against both PIConGPU and Smilei, and A7 bases its agreement thresholds on the measured spread between the two codes. WP-012 measures the Smilei seed-to-seed spread and deliberately proposes no threshold, because the second reference code is missing. This package adds PIConGPU and measures that spread.

It also carries the 2D path into the second code. PIConGPU runs only in 2D3V or 3D3V (`dimension.param`: `DIM2`, `DIM3`), so its deck generator is 2D from the start and reads WP-019's case schema 2, and the 1D cases run in a thin periodic 2D box. WP-019 owns schema 2, the Smilei 2D generator and reader, and the 2D case files with their pre-registered single-run observables; this package builds on them, so the generators, readers and containers are proven in both codes when the 2D LWFA case and the 2D engine arrive. Bugs then show up against linear theory instead of inside an LWFA comparison.

Like WP-003 and WP-012, the package touches only `scripts/`, `reports/` and `questions/`, needs no engine code, and runs in parallel with the engine packages.

## Scope

**Schema 2 is WP-019's.** PIConGPU's generator reads schema 2 as WP-019 defines it, and this package adds no schema field. A schema-1 case run through PIConGPU becomes a quasi-1D 2D run as follows:

- The box is periodic in y and as narrow as PIConGPU's supercell and domain constraints allow.
- The total macro-particle count per x-column matches the 1D case. The report states the transverse cells and particles per cell used.
- Observables are taken from the y-averaged fields.

**PIConGPU.** A container recipe with a pinned PIConGPU release and the toolchain to compile a setup for the RTX 4080 SUPER (CUDA backend). PIConGPU fixes most of a setup at compile time through `.param` files, so each run's `run.sh` compiles its setup and then runs it. The image therefore carries the full toolchain and every dependency, because cross-code jobs have no network. The deck generator `scripts/crosscode/picongpu.jl` writes the `.param` files and the run configuration from a case file. It converts A1's normalized units to PIConGPU's units in one tested function. Since PIConGPU works in SI, that function takes a reference wavelength (λ₀ = 0.8 µm unless the case says otherwise) and records it in the manifest.

**Matching.** The PIConGPU setup has to match the Smilei one on every setting A5 lists:

- shape order 2 (quadratic);
- Esirkepov current deposition;
- no current filter on the 1D set;
- the Boris pusher;
- immobile ions;
- particles per cell (for a schema-1 case, the per-column count of the quasi-1D bullet above);
- the output cadence;
- precision. PIConGPU defaults to FP32 and Smilei to FP64. The 1D set runs in FP64 in both codes, and in FP32 in PIConGPU as a second column, so the precision effect is measured, not mixed into the code difference.

Every PIConGPU-specific choice the generator makes is verified from the source at the pinned release and listed in the README, as WP-003 did for Smilei: the names of the shape and solver classes, how the seed is set, how the initial field is obtained, and the domain constraints.

**Output.** PIConGPU writes openPMD. The package uses the HDF5 backend so that the existing `HDF5.jl` reader extends to it. `scripts/crosscode/read.jl` gains PIConGPU's openPMD layout in 1D and 2D, read into the same schema as Smilei's output (Smilei's 2D layout is WP-019's).

**Cases.**

1. Landau over WP-012's seed ensemble, and two-stream with WP-012's resolution pair, in PIConGPU quasi-1D.
2. WP-019's `landau-oblique-2d.toml` over a smaller seed set, in Smilei 2D (through WP-019's generator) and in PIConGPU 2D. This is the 2D path's exact test in both generators. The ensemble observable (the rate over the seed set) is added as a new pre-registered entry in a commit before this package's own runs; WP-019's pre-registered single-run observables and windows are not altered.

**Runs.** They go through the machine's runner: PIConGPU with `command: picongpu scripts/runs/<tag>`, which always gets the GPU, and Smilei 2D as before with `gpu: no`. Each request sets a `timeout` long enough for the compile.

## Deliverables

0. **Container recipe**: `scripts/crosscode/containers/picongpu.Containerfile` (build context `scripts/crosscode/containers/`, extra files in `picongpu/`). The base image is pinned by digest, and PIConGPU is fetched at a release tag and checked against its commit. The recipe includes the CUDA toolchain for the RTX 4080 SUPER and the dependencies PIConGPU's install guide lists for that release. Check upstream first for a maintained image that can be pinned, and build on it if one exists. `containers/README.md` gains the row. The implementer cannot build images; the owner builds with `crosscode-image <worktree> picongpu`.
1. **Generator**: `scripts/crosscode/picongpu.jl` reading schema 1 and WP-019's schema 2, with `picongpu_parameters` (the unit conversion), the `.param` and configuration writers, and `prepare_run`, which writes `scripts/runs/<tag>/` with a `run.sh` that compiles, then runs, then converts output where needed.
2. **Reader**: PIConGPU's openPMD in 1D and 2D, into the schema `read.jl` documents. 2D fields come back as points × points × outputs, with the y-average helper the quasi-1D observables use.
3. **Ensemble observable**: the oblique Landau rate over the seed set, added to `landau-oblique-2d.toml` as a new pre-registered entry in a commit before any run of this package; WP-019's entries stay as they are.
4. **Runs**: the PIConGPU 1D set in FP64 and FP32, and the oblique case over the seed set in both codes, each with its manifest, `observables.json` and the standard plot set with JSON sidecars (F3). The runner requests and their exact commands go in the PR.
5. **Report**: `reports/crosscode-picongpu/README.md` covers:
   - γ per seed and per code;
   - the Smilei–PIConGPU difference per observable, in units of each code's seed spread;
   - the FP32–FP64 difference in PIConGPU;
   - the oblique 2D rates in both codes against theory, next to WP-019's single Smilei run;
   - the compile and run times per case;
   - every matching choice, and anything the codes could not be made to match.

   The numerics-reviewer role reviews it as a fresh subagent, as in WP-012. Its findings are committed next to the report and answered.
6. **Proposals**: `questions/wp-017-a7-thresholds.md` proposes the A7 agreement thresholds from the measured Smilei–PIConGPU spread, with their derivation, for the owner to accept by editing A7. `tolerances.toml` gets them only as `status = "proposed"` entries.
7. **LWFA prerequisites**: `questions/wp-017-lwfa-case.md` lists what A5's 2D LWFA case still needs before it can be written as a case file in both codes:
   - the parameters A5 names, with no values invented;
   - the schema fields neither this package nor WP-019 added (laser antenna, moving window, density profile and ramp, injected bunch);
   - the matching issues found here that will carry over.

## Factory side (owner, outside the repository)

Build the PIConGPU image from deliverable 0 with `crosscode-image <worktree> picongpu`. The runner already accepts `picongpu` jobs and gives them the GPU. Until the image exists, the package finishes everything except the PIConGPU runs. It then writes `questions/wp-017-picongpu-image.md` and opens the PR with those runs marked absent; the Smilei 2D seed runs go ahead.

## Non-goals

- The 2D LWFA case itself: no laser, no moving window, no density ramp, no bunch.
- Weibel in PIConGPU: neither `weibel.toml` nor WP-019's `weibel-2d.toml` is run in PIConGPU here.
- No schema-2 definition, no Smilei 2D generator, no Smilei 2D reader layout and no authoring of the 2D case files (WP-019); no change of WP-019's pre-registered observables or windows.
- The throughput comparison with Smilei on the same card, and multi-GPU runs.
- The ParticleMesh column in 2D, which waits for the 2D engine.
- Changes to the machine's runner, and any engine code.
- Accepting A5 values or A7 thresholds; this package only proposes them.

## Acceptance

Test ids T-250 to T-259.

- T-250 unit conversion to PIConGPU [A1, WP-017]: the generated `.param` values for a case reproduce hand-computed SI values for density, cell size, time step, the reference wavelength and the perturbation.
- T-251 the oblique profile in both generators [A5, WP-017]: for `landau-oblique-2d.toml`, PIConGPU's generated density profile and the Smilei profile from WP-019's generator agree at the cell centres to roundoff. Schema-2 parsing is WP-019's T-270.
- T-252 openPMD reader [WP-017]: small files written by the test in PIConGPU's 1D and 2D openPMD layout are read into the documented schema and values.
- T-253 the PIConGPU recipe is pinned [A6, WP-017]: the `FROM` line names a digest, and PIConGPU is fetched at a tag or commit, never a branch.
- T-254 quasi-1D equivalence [A5, WP-017]: the quasi-1D deck has the same per-x-column particle count, density and perturbation as the 1D case, read back from the generated files.
- The ensemble observable's commit precedes every run commit of this package, and WP-019's pre-registered entries are unchanged on the diff.
- The runs, manifests, plots, sidecars, report, review and both question files are in the PR, or the runs are marked absent with the image question.
- The trace check passes.

## Depends on

- WP-003: the Smilei container, the case schema, the reader and the analysis.
- WP-012: the seed ensemble, the ensemble observables and the Smilei noise floors this package compares against.
- WP-019: case schema 2, the Smilei 2D generator and reader, and `landau-oblique-2d.toml` with its theory value and pre-registered single-run observables.

Goals A1, A5 (as consumer and as the source of the matching list), A6 (containers and the runner) and A7 (the thresholds it proposes); factory setting F3.

## Budgets

- **GPU:** the PIConGPU 1D-scale runs take minutes each on the RTX 4080 SUPER, plus one compile per distinct setup. They queue on the one GPU behind other runner jobs.
- **CPU:** Smilei 2D on the oblique case takes minutes of CPU per seed, each run within WP-019's per-run bound.
- **Dependencies:** no new Julia dependency; `HDF5.jl` is already in `scripts/`.
- **Storage:** the image is several GB in podman's storage and is never committed to the repository.
