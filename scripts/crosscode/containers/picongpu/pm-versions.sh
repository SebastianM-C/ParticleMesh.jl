#!/usr/bin/env bash
# Print the toolchain of the PIConGPU image as key = "value" lines (TOML), for a run's manifest.
# Installed as /usr/local/bin/pm-versions; the build also writes its output to /opt/picongpu/VERSION.
set -euo pipefail

q() { printf '%s = "%s"\n' "$1" "$2"; }

q picongpu_tag "$PICONGPU_TAG"
q picongpu_commit "$PICONGPU_COMMIT"
v=/opt/openPMD-api/include/openPMD/version.hpp
q openpmd_api "$(for p in MAJOR MINOR PATCH; do sed -n "s/^#define OPENPMDAPI_VERSION_$p \([0-9]*\)$/\1/p" $v; done | paste -sd.)"
q nvcc "$(nvcc --version | sed -n 's/.*release \([0-9.]*\), V\([0-9.]*\).*/\2/p')"
q gcc "$(gcc -dumpfullversion)"
q cmake "$(cmake --version | sed -n '1s/cmake version //p')"
q openmpi "$(mpirun --version 2>/dev/null | sed -n '1s/.*) //p')"
q hdf5 "$(dpkg-query -W -f='${Version}' libhdf5-openmpi-dev 2>/dev/null)"
q boost "$(dpkg-query -W -f='${Version}' libboost-program-options-dev 2>/dev/null)"
q cuda_arch "${PIC_BACKEND#cuda:}"
