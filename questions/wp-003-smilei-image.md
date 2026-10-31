# WP-003: the Smilei image, and where the recipe and the runs live

Work package: `trace/wp/WP-003-crosscode-runs.md`. Files: `scripts/crosscode/containers/smilei.Containerfile`, `scripts/crosscode/run-smilei.sh`, `scripts/crosscode/smilei.jl` (`prepare`).

## 1. The image has to be built by the owner

The sandbox cannot build images, and the runs need one. The recipe is `scripts/crosscode/containers/smilei.Containerfile` (build context `scripts/crosscode/containers/`, extra files in `scripts/crosscode/containers/smilei/`); the build command and the tag convention are in `scripts/crosscode/containers/README.md`.

Assumption I proceeded on: the PR carries the recipe, the case files, the generator, the reader and the analysis, with the runs marked absent in `reports/crosscode-smilei-1d/README.md` until the image exists. The runs are then requested from the machine's runner (`command: smilei scripts/runs/<tag>`, `gpu: no`) and committed in a follow-up commit or PR.

To answer: build the image from the recipe (`crosscode-image` for `localhost/particlemesh-smilei:e4f0148b8952fc87`), then say so here or on the PR; the runs are then requested with the prepared directories `scripts/runs/smilei-landau-9a47a107` and `scripts/runs/smilei-two-stream-736a428f` (regenerate them with `smilei.jl prepare` in a fresh worktree; `scripts/runs/` is not committed).

## 2. Recipe path and run route differ from the work package's text

WP-003 deliverables 0 and 3 name `scripts/crosscode/containers/smilei/Containerfile` and a plain `podman run` from `run-smilei.sh`. The machine's runner expects the recipe at `scripts/crosscode/containers/<family>.Containerfile` with the build context `scripts/crosscode/containers/`, and runs `bash /run-dir/run.sh` from the run directory in its container; podman cannot run inside a factory sandbox.

Decision (the owner's, given in the session): follow the runner's layout, `scripts/crosscode/containers/smilei.Containerfile`, run through the runner with `gpu: no`, and keep `run-smilei.sh` as the plain-podman route for runs outside the factory. Both routes execute the same generated `run.sh` with the same mounts (run directory read-write at `/run-dir`, worktree read-only at `/worktree`, no network, no GPU).

The PR edits WP-003's deliverables 0 and 3 (and the non-goal that excluded the runner's `smilei` command family) to name this layout, so that the package text and the code agree.

## 3. Image tag convention

A run request for `scripts/runs/smilei-landau-9a47a107` (tag `wp-003-smilei-1d-smilei-landau-1`, 2026-10-31) was refused before it started, as expected: `image localhost/particlemesh-smilei:e4f0148b8952fc87 is not built for this worktree's smilei recipe; the owner builds it with crosscode-image`. The runner's hash is not the SHA-256 of the recipe alone nor the rule `run-smilei.sh` uses outside the factory (`6517bfe50b31069b` for this recipe), so `scripts/crosscode/containers/README.md` now says that the factory's builder sets the tag by its own rule and that `run-smilei.sh`'s default tag applies outside the factory only. To answer: the builder's rule, if `run-smilei.sh` should follow it.

## 4. No upstream image was checked

WP-003 asks to check for a maintained upstream Smilei image first. The sandbox cannot reach container registries, so no image was checked; the recipe builds Smilei v5.1 from source on a digest-pinned Ubuntu 24.04 base. If a maintained image exists and can be pinned by digest, switching to it is a change of the `FROM` line and the removal of the build steps.
