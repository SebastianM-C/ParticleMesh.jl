# Scripts: user code that produces results

Everything here uses the package the way a user would, to produce result artifacts: benchmark cases with their expected values, validation and cross-code comparison runs, campaigns and parameter sweeps (`campaigns/`), and the figures and numbers that `reports/` embed. It is its own Pkg environment, outside the root workspace, with a committed `Manifest.toml` so that a run is reproducible from the commit alone: `julia +1.13 --project=scripts <script>` after `Pkg.instantiate()`.

Conventions: plots with CairoMakie, each saved as PNG next to a JSON sidecar with the scalar values the plot shows, so that agents and the trace builder can read the numbers without the image; outputs go to `scripts/runs/` (ignored) and every run is recorded in `trace/` with its inputs hashed. The `package-manager` role owns `Project.toml` and `Manifest.toml` here as everywhere else; request additions through the pinned "Package requests" issue.

`gpu-lock.sh` serializes GPU use between agents on one machine.
