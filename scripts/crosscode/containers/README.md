# Container recipes for cross-code runs

One recipe per code family, `<family>.Containerfile`, with this directory as the build context and any extra files in `<family>/`. Images are built outside the repository and never stored in it.

| Family | Recipe | Contents |
|---|---|---|
| `smilei` | `smilei.Containerfile` | Smilei v5.1 (commit `438d43d8`), CPU build with OpenMPI, OpenMP, parallel HDF5 and happi, on Ubuntu 24.04 pinned by digest |

## Tags

Inside the factory the machine's image builder (`crosscode-image`, run by the owner) builds the image and tags it `localhost/particlemesh-<family>:<hash>` with a hash of the recipe files computed by its own rule; the runner refuses a run when the image for the worktree's recipe is not built and names the expected tag. The run manifest records the image id from the runner's log.

Outside the factory `run-smilei.sh` takes the image as its second argument, or defaults to its own tag, `particlemesh-smilei:<first 16 hex of SHA-256 over the recipe and the files in smilei/ in sorted path order>`, which is not the factory runner's tag:

```
cd scripts/crosscode/containers
hash=$( { echo smilei.Containerfile; find smilei -type f 2>/dev/null; } | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)
podman build -f smilei.Containerfile -t particlemesh-smilei:$hash .
```

The build needs the network (apt and GitHub). A changed recipe gives a new tag either way, so a run names the recipe it came from.

## Pinning

The `FROM` line names a digest, and the source is fetched at a release tag and checked against the commit (`T-013`). To move to a new release, change `SMILEI_TAG` and `SMILEI_COMMIT` together, and the base digest when the base changes.

## Running

The case runs use the generated `scripts/runs/<tag>/run.sh` (see `scripts/crosscode/README.md`): inside the factory through the machine's runner, outside it with `scripts/crosscode/run-smilei.sh`.
