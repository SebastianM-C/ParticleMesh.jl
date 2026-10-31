# Smilei, CPU build with MPI, OpenMP, parallel HDF5 and the Python interface (happi).
# Build context: scripts/crosscode/containers/ (see README.md there). No upstream-maintained
# Smilei image is used: the recipe builds from the release tag on a digest-pinned base.
FROM docker.io/library/ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d

ARG SMILEI_TAG=v5.1
ARG SMILEI_COMMIT=438d43d8ab727d02dd8f4402c50393343e871b99

ENV DEBIAN_FRONTEND=noninteractive
# Packages as listed in Smilei's doc/Sphinx/Use/install_linux.rst (Ubuntu), without sphinx and ipython.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc g++ git ca-certificates \
      python3-dev python3-numpy python3-h5py python3-matplotlib python3-pint \
      libhdf5-openmpi-dev openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONEXE=python3 \
    HDF5_ROOT_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

# Fetch at the tag, then refuse to build anything but the pinned commit.
RUN git clone --depth 1 --branch "${SMILEI_TAG}" https://github.com/SmileiPIC/Smilei.git /opt/smilei \
 && test "$(git -C /opt/smilei rev-parse HEAD)" = "${SMILEI_COMMIT}"

WORKDIR /opt/smilei
# Ubuntu's OpenMPI is built with MPI_THREAD_MULTIPLE; if a rebuild on another base fails at
# MPI_Init_thread, add config=no_mpi_tm (Smilei makefile).
RUN make -j"$(nproc)" \
 && make happi \
 && cp smilei smilei_test /usr/local/bin/ \
 && echo "${SMILEI_TAG} ${SMILEI_COMMIT}" > /opt/smilei/VERSION \
 && rm -rf build

# Runs execute as the caller's uid without capabilities; happi is found through PYTHONPATH,
# not through the build user's site-packages.
ENV PYTHONPATH=/opt/smilei \
    OMPI_MCA_rmaps_base_oversubscribe=1
WORKDIR /run-dir
