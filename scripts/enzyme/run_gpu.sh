#!/usr/bin/env bash
# WP-050 on CUDA: the chain-rule check and the cost of a reverse pass, at the default device heap.
#     scripts/gpu-lock.sh scripts/enzyme/run_gpu.sh [cost cases...]
# A cost case is "<T> <cells> <ppc> <steps>" (cost.jl's arguments); without any, the list below.
# Each run is its own process (a CUDA error leaves its context unusable); a run that fails at the
# default heap is repeated once with a 1 GiB heap (PM_CUDA_HEAP), and the summary says so.
set -u
cd "$(dirname "$0")/../.."
out="scripts/runs/enzyme-gpu-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/spike/active_args.jl}"
{ echo "# commit $(git rev-parse HEAD), patches $PM_ENZYME_PATCH, started $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$out/summary.txt"
run() {  # run <log name> <command...>: once at the default heap, again at 1 GiB if that fails
    local name="$1"; shift
    echo "== $(date -u +%H:%M:%S) $name: $*" >> "$out/summary.txt"
    timeout 3000 "$@" > "$out/$name.log" 2>&1; local rc=$?
    echo "   exit $rc (default heap)" >> "$out/summary.txt"
    if [ $rc -ne 0 ]; then
        PM_CUDA_HEAP=1073741824 timeout 3000 "$@" > "$out/$name-heap1g.log" 2>&1; rc=$?
        echo "   exit $rc (1 GiB heap)" >> "$out/summary.txt"
    fi
    grep -hE 'verdict|relative difference|times:|^T = |^steps|reverse failed|^wrote|  wrote|^ERROR' "$out/$name"*.log | cut -c1-300 >> "$out/summary.txt"
}
for args in "fast 20 1 Float64 7" "fast 20 1 Float32 7" "fast 100 1 Float64 3"; do
    run "chain-${args// /-}" "${J[@]}" scripts/enzyme/chain_rule.jl $args
done
cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=("Float32 10x8 3x3 1,2,4,8,16,32,64" "Float64 10x8 3x3 1,2,4,8,16,32,64"
                                 "Float32 64x64 3x3 1,2,4,8,16,32" "Float32 256x256 2x2 1,2,4,8,16")
for c in "${cases[@]}"; do
    run "cost-${c// /-}" "${J[@]}" scripts/enzyme/cost.jl $c
done
echo "# finished $(date -u +%FT%TZ)" >> "$out/summary.txt"
