# WP-017: the PIConGPU image

Work package: `trace/wp/WP-017-crosscode-picongpu.md`, deliverable 0, done ahead of the rest of the package so that the image can be built while WP-012 and WP-019 run. Files: `scripts/crosscode/containers/picongpu.Containerfile`, `scripts/crosscode/containers/picongpu/pm-versions.sh`, the `picongpu` row in `scripts/crosscode/containers/README.md`, T-253 in `scripts/crosscode/test/runtests.jl`.

## 1. The image

The owner built `localhost/particlemesh-picongpu:a743cbb40f17cdae`, image id `1b4f3d0fb3c291e2d861732bda21463920167f39227c21b7947c37a9238ff50b`, from commit `89984e7` of this recipe. NVIDIA's `ubuntu2404` repository served `cuda-minimal-build-12-5` and `cuda-libraries-dev-12-5` (nvcc 12.5.82), and the build-time smoke compile passed. A recipe change gives a new tag, which the owner builds again with `crosscode-image <worktree> picongpu`.

Toolchain in the image, as `pm-versions` reports it: PIConGPU 0.8.0 (`6bb3470e`), openPMD-api 0.15.2, nvcc 12.5.82, gcc 13.3.0, CMake 3.28.3, OpenMPI 4.1.6, HDF5 1.10.10 (Ubuntu `1.10.10+repack-3.1ubuntu4`), Boost 1.83, `cuda_arch = 89`. PNGwriter is not installed, so the PNG plugin is unavailable; the runs use openPMD.

## 2. Driver and GPU passthrough

The image carries no driver. The runner gives `picongpu` jobs the GPU, which supplies the host driver's user-space libraries at run time. The host driver is 595.104.02, which supports up to CUDA 13.2 according to `nvidia-smi`, so the image's CUDA 12.5 runs on it without compatibility packages. A driver downgrade below CUDA 12.5 support would break the image.

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

## 5. Smoke run

Runner job `picongpu-recipe-smoke-2` (worktree HEAD `d34008f`, image `1b4f3d0f…`, GPU on, network off) ran the script below and exited 0 after 93 s, a figure that may include time in the queue. `pic-build` of the 2D, 64-bit `Empty` setup took 1 min 10 s of wall time inside the job. Ten steps on a 64×64 grid ran in 12 ms, with initialization taking 1.05 s. The setup was compiled only for `sm_89` and ran, so the job ran on the RTX 4080 SUPER. PIConGPU reports its random number generator as `RNGProvider2XorMin` with seed 42 by default; WP-017's generator sets the seed per run. The run directory is `scripts/runs/picongpu-smoke/` (not committed), and its `run.sh` is:

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

Request it again with `command: picongpu scripts/runs/picongpu-smoke` after any recipe change.
