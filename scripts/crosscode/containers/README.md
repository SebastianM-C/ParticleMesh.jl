# Container recipes for cross-code runs

One recipe per code family, `<family>.Containerfile`, with this directory as the build context and any extra files in `<family>/`. Images are built outside the repository and never stored in it.

| Family | Recipe | Contents |
|---|---|---|
| `smilei` | `smilei.Containerfile` | Smilei v5.1 (commit `438d43d8`), CPU build with OpenMPI, OpenMP, parallel HDF5 and happi, on Ubuntu 24.04 pinned by digest |
| `picongpu` | `picongpu.Containerfile` | PIConGPU 0.8.0 (commit `6bb3470e`) source with the toolchain to compile setups for the RTX 4080 SUPER (`sm_89`): CUDA 12.5 from NVIDIA's apt repository, GCC 13, OpenMPI, parallel HDF5, Boost, openPMD-api 0.15.2 (commit `e8e2aeb2`, HDF5 backend only), on the same Ubuntu 24.04 digest. The build compiles a 2D, 64-bit smoke setup; `pm-versions` prints the toolchain for the run manifest |

## Tags

Inside the factory the machine's image builder (`crosscode-image`, run by the owner) builds the image and tags it `localhost/particlemesh-<family>:<hash>` with a hash of the recipe files computed by its own rule; the runner refuses a run when the image for the worktree's recipe is not built and names the expected tag. The run manifest records the image id from the runner's log.

Outside the factory an image is tagged by its own rule (`run-smilei.sh` takes the image as its second argument or defaults to this tag; PIConGPU has no outside runner yet), `particlemesh-<family>:<first 16 hex of SHA-256 over the recipe and the files in <family>/ in sorted path order>`, which is not the factory runner's tag:

```
cd scripts/crosscode/containers
family=smilei   # or picongpu
hash=$( { echo $family.Containerfile; find $family -type f 2>/dev/null; } | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)
podman build -f $family.Containerfile -t particlemesh-$family:$hash .
```

The build needs the network (apt, NVIDIA's CUDA repository for `picongpu`, and GitHub). The `picongpu` build needs no GPU. Compiling one 2D setup in a runner job took about 1 min 10 s with `-j 16` in the smoke run (`questions/wp-017-picongpu-image.md` §5), and every run pays that compile again; the time and memory of the image build, whose own smoke compile uses `-j "$(nproc)"`, were not measured. A changed recipe gives a new tag either way, so a run names the recipe it came from.

## Pinning

The `FROM` line names a digest, and every source is fetched at a release tag and checked against the commit (`T-013` for Smilei, `T-253` for PIConGPU and openPMD-api). To move to a new release, change `<NAME>_TAG` and `<NAME>_COMMIT` together, and the base digest when the base changes. For an annotated tag the commit is the peeled one (`git ls-remote <repo> 'refs/tags/<tag>^{}'`), not the tag object. CUDA is pinned to a minor series (`CUDA_SERIES`, `CUDA_RELEASE`), and the build refuses an `nvcc` of another release; the patch level is whatever NVIDIA's repository serves for that series at build time, and `pm-versions` records it. Choose the series from the CUDA and GCC pairs PIConGPU's CI accepts at the pinned tag (`share/ci/n_wise_generator.py`).

## Running

The case runs use the generated `scripts/runs/<tag>/run.sh` (see `scripts/crosscode/README.md`): inside the factory through the machine's runner, outside it with `scripts/crosscode/run-smilei.sh`.
