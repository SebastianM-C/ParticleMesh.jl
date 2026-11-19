#!/usr/bin/env bash
# Spike-only: time each unrolling change of unroll/ on its own and all together, on one GPU,
# with the WP-045 step phases (scripts/optim/step_timing.jl: Float32, Float64 and duals in
# Reference, the plain types also in Fast) and the full-size 2D throughput (scripts/em2d/timing.jl).
# Run from the worktree root:  scripts/enzyme/spike/run_unroll_bench.sh [configs...]
# The baseline runs first and last, so that drift shows. Writes scripts/runs/unroll-bench-<stamp>/.
set -uo pipefail
ROOT=$(pwd)
OUT="$ROOT/scripts/runs/unroll-bench-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"
P=scripts/enzyme/spike/unroll
CONFIGS=("${@:-base gather atomic prefix pick gamma all base2}")
read -r -a CONFIGS <<< "${CONFIGS[*]}"
patches() {
  case "$1" in
    base|base2) echo "" ;;
    all) echo "$P/gather.jl;$P/atomic.jl;$P/prefix.jl;$P/pick.jl;$P/gamma.jl" ;;
    *) echo "$P/$1.jl" ;;
  esac
}
{
  echo "# unroll bench, commit $(git rev-parse HEAD), started $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$OUT/summary.txt"
for c in "${CONFIGS[@]}"; do
  for job in step timing; do
    if [ "$job" = step ]; then args=(scripts/optim/step_timing.jl cuda fast); else args=(scripts/em2d/timing.jl cuda 50); fi
    log="$OUT/$c-$job.log"
    t0=$(date +%s)
    PM_PATCH="$(patches "$c")" scripts/gpu-lock.sh julia +1.13 --project=scripts \
      scripts/enzyme/spike/with_patches.jl "${args[@]}" > "$log" 2>&1
    rc=$?
    dir=$(grep -o 'outputs in .*' "$log" | tail -1 | sed 's/outputs in //')
    echo "$c $job rc=$rc $(( $(date +%s) - t0 ))s dir=$dir" >> "$OUT/summary.txt"
  done
done
# One table of the step phases over all configurations, in run order.
cmp=()
while read -r c job rc t d; do
  [ "$job" = step ] && [ "$rc" = rc=0 ] && cmp+=("$c=${d#dir=}")
done < <(grep -v '^#' "$OUT/summary.txt" | grep ' step ')
[ ${#cmp[@]} -gt 0 ] && julia +1.13 --project=scripts scripts/optim/step_timing.jl compare "${cmp[@]}" > "$OUT/compare.log" 2>&1
echo "# finished $(date -u +%FT%TZ)" >> "$OUT/summary.txt"
