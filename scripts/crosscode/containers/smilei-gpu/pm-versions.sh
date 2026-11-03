#!/usr/bin/env bash
# Print the toolchain of the smilei-gpu image as key = "value" lines (TOML), for a run's manifest.
# Installed as /usr/local/bin/pm-versions; the build also writes its output to /opt/smilei/VERSION.toml.
set -euo pipefail

q() { printf '%s = "%s"\n' "$1" "$2"; }

q smilei_tag "$SMILEI_TAG"
q smilei_commit "$SMILEI_COMMIT"
q nvhpc "$(nvc++ --version | sed -n 's/^nvc++ \([0-9.]*\).*/\1/p' | head -1)"
q nvcc "$(nvcc --version | sed -n 's/.*release \([0-9.]*\), V\([0-9.]*\).*/\2/p')"
q gcc "$(gcc -dumpfullversion)"
q mpi "$(mpirun --version 2>/dev/null | sed -n '1s/.*) //p')"
q hdf5 "$(sed -n 's/^#define H5_VERSION "\(.*\)"$/\1/p' /opt/hdf5/include/H5pubconf.h)"
q python "$(python3 -c 'import platform; print(platform.python_version())')"
q numpy "$(python3 -c 'import numpy; print(numpy.__version__)')"
q gpu_cc "$(cat /opt/smilei/GPU_CC)"
q nvcc_arch "sm_$PM_SM"
q binaries "smilei (guide flags, -gpu=...,fastmath), smilei-nofastmath (the same without fastmath)"
