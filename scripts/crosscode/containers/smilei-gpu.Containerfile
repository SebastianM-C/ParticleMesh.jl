# Smilei, GPU build (OpenACC through the NVIDIA HPC SDK, the nvhpc route of Smilei's own guide,
# doc/Sphinx/Use/install_linux_GPU.rst at v5.1), for runs on the RTX 4080 SUPER. The CPU image
# (smilei.Containerfile) stays the reference. Build context: scripts/crosscode/containers/ (see
# README.md there). The runner provides the driver at run time; the build needs no GPU.
#
# The guide's versions: NVIDIA HPC SDK 23.11 with CUDA 12.3, HDF5 1.14.2 built with the SDK's
# mpicc, GCC <= 12. Ubuntu 22.04 (GCC 11) is a platform the 23.11 SDK supports; Ubuntu 24.04's
# GCC 13 is outside the guide. The SDK comes from NVIDIA's signed apt repository (the same
# package as the guide's tarball, nvhpc_2023_2311_Linux_x86_64_cuda_12.3), so apt checks it.
FROM docker.io/library/ubuntu:22.04@sha256:8a2321a92b438238f93b4346fff61bc4259c99fdd4d9b52ee86f153fee0c344f

ARG SMILEI_TAG=v5.1
ARG SMILEI_COMMIT=438d43d8ab727d02dd8f4402c50393343e871b99
ARG HDF5_TAG=hdf5-1_14_2
ARG HDF5_COMMIT=2d926cf0426d2ed9403946071d9b05ac8b4dd778
ARG NVHPC_SERIES=23-11
ARG NVHPC_RELEASE=23.11
ARG CUDA_RELEASE=12.3
# sm_89 is the RTX 4080 SUPER.
ARG PM_SM=89

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONEXE=python3
# Smilei's CPU packages (doc/Sphinx/Use/install_linux.rst, Ubuntu) without MPI and HDF5, which
# come from the SDK and from source; cmake for HDF5; curl and gnupg for NVIDIA's repository.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc g++ gfortran git ca-certificates curl gnupg cmake file \
      python3-dev python3-numpy python3-h5py python3-matplotlib python3-pint \
 && rm -rf /var/lib/apt/lists/*

# NVIDIA HPC SDK (compilers, the bundled CUDA, its MPI); refused unless nvc++ and nvcc are the pinned releases.
RUN curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg \
 && echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' \
      > /etc/apt/sources.list.d/nvhpc.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends "nvhpc-${NVHPC_SERIES}=${NVHPC_RELEASE}" \
 && rm -rf /var/lib/apt/lists/* \
 && NV=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE} \
 && "$NV/compilers/bin/nvc++" --version | grep -q "nvc++ ${NVHPC_RELEASE}" \
 && "$NV/compilers/bin/nvcc" --version | grep -q "release ${CUDA_RELEASE}"

COPY smilei-gpu/nvidia_env.sh /opt/smilei-gpu/nvidia_env.sh
COPY smilei-gpu/acc-probe.cpp /opt/smilei-gpu/acc-probe.cpp
COPY smilei-gpu/pm-versions.sh /usr/local/bin/pm-versions

# The nvc++ target: cc89 if this nvc++ accepts it for OpenACC, else cc86, whose code runs on the
# 8.9 card (same major compute capability). Decided by compiling (not linking: the build has no
# driver) a probe; recorded in GPU_CC; the build stops if neither compiles. Every nvc++ call names
# the bundled toolkit (-gpu=cuda12.3): without a driver nvc++ otherwise asks for the CUDA of the
# driver or its fallback 11.8, which this single-CUDA package does not have.
RUN mkdir -p /opt/smilei \
 && export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/compilers/bin:$PATH \
 && for cc in cc89 cc86; do \
      if nvc++ -acc=gpu -gpu=${cc},cuda${CUDA_RELEASE} -c -o /tmp/acc-probe.o /opt/smilei-gpu/acc-probe.cpp; then \
        echo $cc > /opt/smilei/GPU_CC; break; \
      fi; \
    done \
 && test -s /opt/smilei/GPU_CC && rm -f /tmp/acc-probe.o && cat /opt/smilei/GPU_CC

# HDF5 with the SDK's mpicc, as in the guide (parallel on); tests and examples off to save build
# time. Fetched at the tag and refused unless it is the pinned commit.
RUN git clone --depth 1 --branch "${HDF5_TAG}" https://github.com/HDFGroup/hdf5.git /tmp/hdf5 \
 && test "$(git -C /tmp/hdf5 rev-parse HEAD)" = "${HDF5_COMMIT}" \
 && export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/comm_libs/mpi/bin:$PATH \
 && cmake -S /tmp/hdf5 -B /tmp/hdf5-build -DCMAKE_C_COMPILER="$(command -v mpicc)" \
      -DCMAKE_INSTALL_PREFIX=/opt/hdf5 -DCMAKE_BUILD_TYPE=Release -DHDF5_ENABLE_PARALLEL=ON \
      -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF \
 && cmake --build /tmp/hdf5-build -j"$(nproc)" \
 && cmake --install /tmp/hdf5-build \
 && rm -rf /tmp/hdf5 /tmp/hdf5-build \
 && test -f /opt/hdf5/lib/libhdf5.so

# Smilei at the tag, refused unless it is the pinned commit.
RUN git clone --depth 1 --branch "${SMILEI_TAG}" https://github.com/SmileiPIC/Smilei.git /opt/smilei/src \
 && test "$(git -C /opt/smilei/src rev-parse HEAD)" = "${SMILEI_COMMIT}"

# Two binaries from one source: `smilei` with the guide's flags (-gpu=...,fastmath) and
# `smilei-nofastmath` with the same flags without fastmath, so that runs can measure its effect.
# make config="gpu_nvidia" as in the guide (it implies noopenmp). At most 16 parallel compiles:
# nvc++ with OpenACC needs far more memory per process than g++.
WORKDIR /opt/smilei/src
RUN jobs=$(( $(nproc) < 16 ? $(nproc) : 16 )) \
 && for variant in fastmath nofastmath; do \
      ( export NVHPC_RELEASE=${NVHPC_RELEASE} PM_CUDA=${CUDA_RELEASE} PM_SM=${PM_SM} PM_GPU_CC=$(cat /opt/smilei/GPU_CC) \
               PM_FASTMATH=$([ $variant = fastmath ] && echo ,fastmath || true) \
        && . /opt/smilei-gpu/nvidia_env.sh \
        && make -j"$jobs" config="gpu_nvidia" BUILD_DIR=build_$variant \
        && name=$([ $variant = fastmath ] && echo smilei || echo smilei-nofastmath) \
        && cp smilei /usr/local/bin/$name && cp smilei_test /usr/local/bin/${name}_test \
        && rm -f smilei smilei_test ) || exit 1; \
    done \
 && rm -rf build_fastmath build_nofastmath \
 && export NVHPC_RELEASE=${NVHPC_RELEASE} PM_CUDA=${CUDA_RELEASE} PM_SM=${PM_SM} PM_GPU_CC=$(cat /opt/smilei/GPU_CC) PM_FASTMATH= \
 && . /opt/smilei-gpu/nvidia_env.sh \
 && make happi \
 && echo "${SMILEI_TAG} ${SMILEI_COMMIT} gpu" > /opt/smilei/VERSION

# The run-time environment: the SDK's compilers, MPI and CUDA libraries, HDF5; Smilei selects
# its GPU from SLURM_LOCALID and exits without it (src/Smilei.cpp). The SDK's default MPI
# (comm_libs/mpi) is HPC-X's Open MPI, which looks for its help files, components and runtime
# under NVIDIA's build prefix (/proj/nv/...) unless OPAL_PREFIX names where it is installed; the
# first GPU run failed in MPI_Init that way. /opt/nvhpc-mpi is a link to that prefix, made below.
# Single-rank runs as the caller's uid without capabilities: shared-memory transport only, no
# CMA single-copy (needs ptrace), no hcoll.
ENV SMILEI_TAG=${SMILEI_TAG} \
    SMILEI_COMMIT=${SMILEI_COMMIT} \
    PM_SM=${PM_SM} \
    PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/comm_libs/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/hdf5/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/comm_libs/mpi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE}/cuda/lib64 \
    SLURM_LOCALID=0 \
    OPAL_PREFIX=/opt/nvhpc-mpi \
    PYTHONPATH=/opt/smilei/src \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,vader \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_MCA_coll_hcoll_enable=0

# The MPI prefix: the Open MPI tree that the SDK's mpicc links against (its first library
# directory, whose parent holds share/openmpi; comm_libs/hpcx/bin/mpicc itself sits in a
# wrapper directory, not in that tree), linked as /opt/nvhpc-mpi. The step prints what it
# found before checking it. Then a singleton MPI start with MPI_THREAD_MULTIPLE, as Smilei
# makes it, which needs no GPU (run as root at build time, hence the two ALLOW variables).
COPY smilei-gpu/mpi-probe.c /opt/smilei-gpu/mpi-probe.c
RUN NV=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE} \
 && echo "mpicc: $(command -v mpicc) -> $(readlink -f "$(command -v mpicc)") ($(file -b "$(readlink -f "$(command -v mpicc)")"))" \
 && echo "mpicc --showme: $(env -u OPAL_PREFIX mpicc --showme 2>&1)" \
 && ls -l "$NV/comm_libs" \
 && find "$NV/comm_libs" -name help-mpi-runtime.txt \
 && libdir=$(env -u OPAL_PREFIX mpicc --showme:libdirs | tr ' ' '\n' | sed '/^$/d' | head -1) \
 && prefix=$(readlink -f "$libdir/..") && echo "MPI prefix: $prefix" \
 && test -f "$prefix/share/openmpi/help-mpi-runtime.txt" \
 && ln -s "$prefix" /opt/nvhpc-mpi \
 && mpicc -o /tmp/mpi-probe /opt/smilei-gpu/mpi-probe.c \
 && OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 /tmp/mpi-probe \
 && rm -f /tmp/mpi-probe

# Every shared library resolves except the driver's (libcuda, libnvidia-*), which the runner
# binds in at run time; then the toolchain record for the run manifests.
RUN chmod 0755 /usr/local/bin/pm-versions \
 && for b in smilei smilei-nofastmath; do \
      missing=$(ldd /usr/local/bin/$b | grep 'not found' | grep -v -E 'libcuda\.so|libnvidia' || true); \
      if [ -n "$missing" ]; then echo "$b: $missing"; exit 1; fi; \
    done \
 && pm-versions > /opt/smilei/VERSION.toml && cat /opt/smilei/VERSION.toml

WORKDIR /run-dir
