# Container recipes for cross-code runs

One recipe per code family, `<family>.Containerfile`, with this directory as the build context and any extra files in `<family>/`. Images are built outside the repository and never stored in it.

| Family | Recipe | Contents |
|---|---|---|
| `smilei` | `smilei.Containerfile` | Smilei v5.1 (commit `438d43d8`), CPU build with OpenMPI, OpenMP, parallel HDF5 and happi, on Ubuntu 24.04 pinned by digest |

## Tag convention

An image is tagged `particlemesh-<family>:<hash>`, where `<hash>` is the first 16 hex digits of the SHA-256 over the recipe and the files in `<family>/`, in sorted path order:

```
cd scripts/crosscode/containers
hash=$( { echo smilei.Containerfile; find smilei -type f 2>/dev/null; } | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)
podman build -f smilei.Containerfile -t particlemesh-smilei:$hash .
```

The build needs the network (apt and GitHub). A changed recipe gives a new tag, so a run always names the recipe it came from; the run manifest records the image id the run used.

## Pinning

The `FROM` line names a digest, and the source is fetched at a release tag and checked against the commit (`T-013`). To move to a new release, change `SMILEI_TAG` and `SMILEI_COMMIT` together, and the base digest when the base changes.

## Running

The case runs use the generated `scripts/runs/<tag>/run.sh` (see `scripts/crosscode/README.md`): inside the factory through the machine's runner, outside it with `scripts/crosscode/run-smilei.sh`.
