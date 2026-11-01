# PIConGPU with the CUDA toolchain, for compiling and running setups on the RTX 4080 SUPER.
# Build context: scripts/crosscode/containers/ (see README.md there). PIConGPU fixes a setup at
# compile time, so every run compiles inside this image without network: the image carries the
# compiler, CUDA, MPI, Boost, HDF5, openPMD-api and the PIConGPU source.
#
# Base: the same digest-pinned Ubuntu 24.04 as smilei.Containerfile, with CUDA from NVIDIA's apt
# repository. No upstream PIConGPU release image is used: upstream CI images
# (registry.hzdr.de/crp/alpaka-group-container/...) are CI-only and could not be inspected from the
# factory sandbox (questions/wp-017-picongpu-image.md).
FROM docker.io/library/ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d

ARG PICONGPU_TAG=0.8.0
ARG PICONGPU_COMMIT=6bb3470ee16dd5a3a93f24c85b111be3b8440d22
ARG OPENPMD_TAG=0.15.2
ARG OPENPMD_COMMIT=e8e2aeb28eda917b6b4e21f8d072d55c651b0915
# PIConGPU 0.8.0 CI (share/ci/n_wise_generator.py) accepts nvcc 12.5 with g++ <= 13 and excludes
# 12.4 with g++ 13; Ubuntu 24.04's default g++ is 13. sm_89 is the RTX 4080 SUPER.
ARG CUDA_SERIES=12-5
ARG CUDA_RELEASE=12.5
ARG CUDA_ARCH=89

ENV DEBIAN_FRONTEND=noninteractive
# Dependencies from PIConGPU's docs/source/install/dependencies.rst at the tag: C++17 compiler,
# CMake >= 3.22, MPI, Boost >= 1.74 (program_options, atomic), parallel HDF5 for openPMD-api;
# rsync for pic-create, git for the fetch.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc g++ git ca-certificates curl gnupg cmake file rsync \
      libboost-program-options-dev libboost-atomic-dev \
      openmpi-bin libopenmpi-dev libhdf5-openmpi-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# CUDA compiler and libraries only (no driver, no Nsight); the runner provides the driver at run time.
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      cuda-minimal-build-${CUDA_SERIES} cuda-libraries-dev-${CUDA_SERIES} \
 && rm -rf /var/lib/apt/lists/* \
 && /usr/local/cuda-${CUDA_RELEASE}/bin/nvcc --version | grep -q "release ${CUDA_RELEASE}"

ENV CUDA_HOME=/usr/local/cuda-${CUDA_RELEASE} \
    PATH=/usr/local/cuda-${CUDA_RELEASE}/bin:${PATH}

# openPMD-api with MPI and HDF5 only (the repository's reader is HDF5.jl); fetched at the tag,
# refused unless it is the pinned commit. Flags follow PIConGPU's share/ci/install/openPMD.sh.
RUN git clone --depth 1 --branch "${OPENPMD_TAG}" https://github.com/openPMD/openPMD-api.git /tmp/openPMD-api \
 && test "$(git -C /tmp/openPMD-api rev-parse HEAD)" = "${OPENPMD_COMMIT}" \
 && cmake -S /tmp/openPMD-api -B /tmp/openPMD-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=/opt/openPMD-api \
      -DopenPMD_USE_MPI=ON -DopenPMD_USE_HDF5=ON -DopenPMD_USE_ADIOS2=OFF -DopenPMD_USE_PYTHON=OFF \
      -DopenPMD_BUILD_CLI_TOOLS=OFF -DopenPMD_BUILD_EXAMPLES=OFF -DopenPMD_BUILD_TESTING=OFF \
 && cmake --build /tmp/openPMD-build -j"$(nproc)" \
 && cmake --install /tmp/openPMD-build \
 && rm -rf /tmp/openPMD-api /tmp/openPMD-build

# PIConGPU at the tag, refused unless it is the pinned commit.
RUN git clone --depth 1 --branch "${PICONGPU_TAG}" https://github.com/ComputationalRadiationPhysics/picongpu.git /opt/picongpu \
 && test "$(git -C /opt/picongpu rev-parse HEAD)" = "${PICONGPU_COMMIT}" \
 && rm -rf /opt/picongpu/.git

# The environment of PIConGPU's dependencies.rst ("PIConGPU Source Code"). PIC_BACKEND makes
# pic-build target the RTX 4080 SUPER without a flag. Single-rank runs as the caller's uid without
# capabilities: no CUDA-aware MPI (dependencies.rst), no CMA single-copy (needs ptrace).
ENV PICONGPU_TAG=${PICONGPU_TAG} \
    PICONGPU_COMMIT=${PICONGPU_COMMIT} \
    PICSRC=/opt/picongpu \
    PIC_EXAMPLES=/opt/picongpu/share/picongpu/examples \
    PIC_BACKEND=cuda:${CUDA_ARCH} \
    PATH=/opt/picongpu:/opt/picongpu/bin:/opt/picongpu/src/tools/bin:${PATH} \
    PYTHONPATH=/opt/picongpu/lib/python \
    CMAKE_PREFIX_PATH=/opt/openPMD-api \
    LD_LIBRARY_PATH=/opt/openPMD-api/lib \
    OMPI_MCA_mpi_leave_pinned=0 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_MCA_rmaps_base_oversubscribe=1

COPY picongpu/pm-versions.sh /usr/local/bin/pm-versions

# Smoke build in the configuration WP-017 runs: the Empty example switched to 2D and 64-bit
# precision, compiled for sm_${CUDA_ARCH}. nvcc needs no GPU, so a broken toolchain fails here
# instead of in the first runner job. The build is discarded.
RUN chmod 0755 /usr/local/bin/pm-versions \
 && pic-create "${PIC_EXAMPLES}/Empty" /tmp/smoke \
 && sed 's/#define SIMDIM DIM3/#define SIMDIM DIM2/' "${PICSRC}/include/picongpu/param/dimension.param" \
      > /tmp/smoke/include/picongpu/param/dimension.param \
 && sed 's/namespace precisionPIConGPU = precision32Bit;/namespace precisionPIConGPU = precision64Bit;/' \
      "${PICSRC}/include/picongpu/param/precision.param" > /tmp/smoke/include/picongpu/param/precision.param \
 && grep -q 'SIMDIM DIM2' /tmp/smoke/include/picongpu/param/dimension.param \
 && grep -q 'precisionPIConGPU = precision64Bit' /tmp/smoke/include/picongpu/param/precision.param \
 && cd /tmp/smoke && pic-build -j "$(nproc)" \
 && test -x /tmp/smoke/bin/picongpu \
 && cd / && rm -rf /tmp/smoke \
 && pm-versions > /opt/picongpu/VERSION

WORKDIR /run-dir
