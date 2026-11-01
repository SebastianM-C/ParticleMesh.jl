# WP-017: the PIConGPU image

Work package: `trace/wp/WP-017-crosscode-picongpu.md`, deliverable 0, done ahead of the rest of the package so that the image can be built while WP-012 and WP-019 run. Files: `scripts/crosscode/containers/picongpu.Containerfile`, `scripts/crosscode/containers/picongpu/pm-versions.sh`, the `picongpu` row in `scripts/crosscode/containers/README.md`, T-253 in `scripts/crosscode/test/runtests.jl`.

## 1. The image has to be built by the owner

Sessions cannot build images. The runner expects `localhost/particlemesh-picongpu:a743cbb40f17cdae` for this recipe; a smoke request (tag `picongpu-recipe-smoke-1`) was refused with "image ... is not built", as expected. To answer: build it with `crosscode-image <worktree> picongpu` from the worktree of the PR (or from `main` once it merges), and note the image id here or on the PR.

Two steps of the recipe could not be tried from the factory sandbox, which reaches only GitHub and the Julia servers:

- **CUDA from NVIDIA's apt repository.** The recipe installs `cuda-minimal-build-12-5` and `cuda-libraries-dev-12-5` from `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/` through `cuda-keyring_1.1-1_all.deb`. That the `ubuntu2404` repository carries the 12.5 series under these package names is from memory, unverified. If it does not, two fallbacks: `cuda-toolkit-12-5` (larger: it pulls Nsight), or the base `ubuntu:22.04` pinned by digest with the `ubuntu2204` repository and `g++-12` (PIConGPU's CI accepts CUDA 12.5 with g++ ≤ 13).
- **The smoke compile.** The last `RUN` compiles PIConGPU's `Empty` example in 2D at 64-bit precision for `sm_89` and discards it, so a broken toolchain fails at build time rather than in a runner job. Its time and memory are not measured; if it runs out of memory, lower its `-j`.

## 2. Driver and GPU passthrough

The image carries no driver. The runner gives `picongpu` jobs the GPU, which supplies the host driver's user-space libraries at run time. A CUDA 12.5 build needs a host driver that supports 12.5, or minor-version compatibility from an older 12.x driver; the exact minimum is in NVIDIA's CUDA compatibility table, not checked here. To answer: the host driver version (`nvidia-smi`), if it is older than the 555 series.

## 3. No upstream image was used

WP-017 asks to build on a maintained upstream image if one exists. Two candidates exist, and neither was inspected, because the sandbox cannot reach container registries:

- PIConGPU's CI images, `registry.hzdr.de/crp/alpaka-group-container/alpaka-ci-<ubuntu>-cuda<ver>-gcc-pic:<tag>` (named in `share/ci/compiler_nvcc_cuda.yml` at 0.8.0). They are CI images for alpaka and PIConGPU, not release images.
- NVIDIA's `nvidia/cuda:12.5.x-devel-ubuntu24.04`.

Assumption I proceeded on: build on the digest-pinned Ubuntu 24.04 that `smilei.Containerfile` already uses, so the recipe is fully pinned without a registry lookup. Switching to `nvidia/cuda` is a change of the `FROM` line (with its digest) and removing the CUDA `RUN` step. To answer: say so if you prefer either image, and give its digest.

## 4. Pins and choices

- PIConGPU `0.8.0`, commit `6bb3470ee16dd5a3a93f24c85b111be3b8440d22`, the newest release tag (December 2024).
- openPMD-api `0.15.2`, commit `e8e2aeb28eda917b6b4e21f8d072d55c651b0915`. That is the commit the annotated tag points to; the tag object itself is `3f71e0fd`. PIConGPU 0.8.0 requires openPMD-api 0.15 or newer, and its profiles use 0.15.2. The build has HDF5 only, no ADIOS2, because the repository's reader is `HDF5.jl`.
- CUDA 12.5 with Ubuntu 24.04's g++ 13. PIConGPU 0.8.0's CI (`share/ci/n_wise_generator.py`) excludes CUDA 12.4 with g++ 13, and accepts 12.4–12.5 with g++ ≤ 13.
- `PIC_BACKEND=cuda:89`, so `pic-build` targets the RTX 4080 SUPER without a flag.
- Single-rank OpenMPI without CUDA awareness: `OMPI_MCA_mpi_leave_pinned=0` (PIConGPU's dependencies.rst), and `OMPI_MCA_btl_vader_single_copy_mechanism=none` because runs have no capabilities.
- `pm-versions` prints the toolchain as TOML lines for the run manifest. The build also writes that output to `/opt/picongpu/VERSION`.

## 5. Smoke run once the image exists

Before any WP-017 case runs, one runner job checks the image, the GPU passthrough and the driver end to end. The run directory is `scripts/runs/picongpu-smoke/` (not committed), and its `run.sh` is:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /run-dir
pm-versions | tee versions.toml
rm -rf setup
pic-create "$PIC_EXAMPLES/Empty" setup
sed 's/#define SIMDIM DIM3/#define SIMDIM DIM2/' "$PICSRC/include/picongpu/param/dimension.param" > setup/include/picongpu/param/dimension.param
sed 's/namespace precisionPIConGPU = precision32Bit;/namespace precisionPIConGPU = precision64Bit;/' "$PICSRC/include/picongpu/param/precision.param" > setup/include/picongpu/param/precision.param
cd setup
time pic-build -j 16
mpiexec -n 1 bin/picongpu -d 1 1 -g 64 64 -s 10 --periodic 1 1 --versionOnce
```

It is requested from the runner with `command: picongpu scripts/runs/picongpu-smoke` after the image is built. Until then, a request fails with "image ... is not built", as expected.
