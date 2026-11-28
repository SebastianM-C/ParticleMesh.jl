#!/usr/bin/env bash
# WP-090's benchmark batch on one GPU machine (scripts/bench3d/bench3d.jl).
#
#   scripts/bench3d/run-batch.sh build <root>    # every picongpu-* directory's build.sh: CPU only, outside any GPU queue
#   scripts/bench3d/run-batch.sh run <root>      # every run of <root>/runs.txt in order, then compare into <root>/compare
#
# <root> is a directory written by `bench3d.jl prepare` (scripts/runs/bench3d-<tag>). PIC_RUNNER names the
# command that runs a script of a PIConGPU directory inside the PIConGPU container, the directory bound at
# /run-dir and the GPU in: `$PIC_RUNNER <dir> <script>`, for example a wrapper around podman run. A run whose
# timing.json exists is skipped, so the batch can be run again after a failure; a failed run is listed at
# the end and does not stop the others. The timing runs need the GPU to themselves.
set -uo pipefail
cmd=${1:?usage: run-batch.sh build|run <root>}
root=$(cd "${2:?usage: run-batch.sh build|run <root>}" && pwd)
repo=$(cd "$(dirname "$0")/../.." && pwd)
jl=(julia +1.13 --project="$repo/scripts" "$repo/scripts/bench3d/bench3d.jl")
: "${PIC_RUNNER:?set PIC_RUNNER to the command that runs <dir> <script> in the PIConGPU container}"
failed=""
echo "# bench3d $cmd $root at $(git -C "$repo" rev-parse HEAD), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/# gpu: /'
while read -r name; do
  [ -n "$name" ] || continue
  d="$root/$name"
  case "$cmd:$name" in
    build:picongpu-*)
      [ -x "$d/setup/bin/picongpu" ] && continue
      echo "# build $name: $(date -u +%H:%M:%S)"
      $PIC_RUNNER "$d" build.sh > "$d/runner-build.log" 2>&1 || failed="$failed $name" ;;
    run:pm-*)
      [ -f "$d/timing.json" ] && continue
      echo "# run $name: $(date -u +%H:%M:%S)"
      "${jl[@]}" run-pm "$d" cuda > "$d/run.log" 2>&1 || failed="$failed $name"
      tail -n 1 "$d/run.log" ;;
    run:picongpu-*)
      [ -f "$d/timing.json" ] && continue
      [ -x "$d/setup/bin/picongpu" ] || { failed="$failed $name(not-built)"; continue; }
      echo "# run $name: $(date -u +%H:%M:%S)"
      if $PIC_RUNNER "$d" run.sh > "$d/runner-run.log" 2>&1; then
        "${jl[@]}" read-picongpu "$d" > "$d/read.log" 2>&1 || failed="$failed $name(read)"
        grep -h "calculation  simulation time" "$d/picongpu.log"
      else
        failed="$failed $name"
      fi ;;
  esac
done < "$root/runs.txt"
if [ "$cmd" = run ]; then
  "${jl[@]}" compare "$root" "$root/compare" > "$root/compare.log" 2>&1 && "${jl[@]}" table "$root/compare/comparison.json"
fi
echo "# failed:${failed:- none}"
