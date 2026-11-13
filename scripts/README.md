# Scripts: user code that produces results

Everything here uses the package the way a user would, to produce result artifacts: benchmark cases with their expected values, validation and cross-code comparison runs, campaigns and parameter sweeps (`campaigns/`), and the figures and numbers that `reports/` embed. It is its own Pkg environment, outside the root workspace, with a committed `Manifest.toml` so that a run is reproducible from the commit alone: `julia +1.13 --project=scripts <script>` after `Pkg.instantiate()`.

Conventions (F3 in `docs/plan/factory-and-observability.md`): outputs go to `scripts/runs/` (ignored), every run writes its manifest next to them, and every plot is a CairoMakie PNG with its JSON sidecar. Any pull request may add a dependency here (F2); re-resolve this manifest whenever the root one changes.

`gpu-lock.sh` serializes GPU use between agents on one machine.

`digest/digest.jl` shows that a change of the electromagnetic module's state or step layer changes no run result: it hashes the full state and the histories of 23 short 1D and 2D runs and of two harness cases per element type and deposition mode, and its output from the old tree and from the new one, both on one machine, must be the same lines. It runs in the module's test environment, which picks the tree: `julia +1.13 --project=<tree>/lib/ParticleMeshElectromagnetic/test scripts/digest/digest.jl <out.txt> [modes] [cases]` (its header says what not to compare; T-746 runs two of its cases).
