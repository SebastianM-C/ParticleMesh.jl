#!/usr/bin/env bash
# Serialize GPU use between agents and worktrees on one machine.
# Usage: PM_TEST_GPU=1 scripts/gpu-lock.sh julia +1.13 --project -e 'using Pkg; Pkg.test()'
set -euo pipefail
LOCK="${PM_GPU_LOCK:-/tmp/particlemesh-gpu.lock}"
WAIT="${PM_GPU_LOCK_WAIT:-3600}"
exec flock -w "$WAIT" "$LOCK" "$@"
