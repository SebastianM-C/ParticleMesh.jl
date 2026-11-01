#!/usr/bin/env bash
# Serialize GPU use between agents and worktrees on one machine.
# Usage: PM_TEST_GPU=1 scripts/gpu-lock.sh julia +1.13 --project -e 'using Pkg; Pkg.test()'
# Waits at most PM_GPU_LOCK_WAIT seconds (default 14400, 4 hours) for the lock. If the lock is not
# free by then, the command does not run and the script exits 75 with a message saying so, so that
# a wait that ran out does not look like a failed run.
set -euo pipefail
LOCK="${PM_GPU_LOCK:-/tmp/particlemesh-gpu.lock}"
WAIT="${PM_GPU_LOCK_WAIT:-14400}"
exec 9>>"$LOCK"
if ! flock -w "$WAIT" 9; then
  echo "gpu-lock: lock wait exceeded: $LOCK was not free within ${WAIT}s, so the command did not run" >&2
  exit 75
fi
# The script keeps fd 9, and with it the lock, until the command ends; the command gets fd 9 closed.
rc=0
"$@" 9>&- || rc=$?
exit "$rc"
