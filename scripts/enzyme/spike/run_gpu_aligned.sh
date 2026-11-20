#!/usr/bin/env bash
# WP-050 spike on CUDA, third stage: slot-aligned checks with a 1 GiB device heap.
#     scripts/gpu-lock.sh scripts/enzyme/spike/run_gpu_aligned.sh
set -u
cd "$(dirname "$0")/../../.."
out="scripts/runs/enzyme-gpu-aligned-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl PM_CUDA_HEAP=1073741824
git rev-parse HEAD > "$out/commit.txt"
run() {  # run <log name> <command...>
    local log="$out/$1.log"; shift
    echo "== $(date -u +%H:%M:%S) $*" | tee -a "$out/summary.txt"
    timeout 2400 "$@" > "$log" 2>&1
    echo "   exit $?" >> "$out/summary.txt"
    grep -E '^\[|^    input .*DIFFERS|^mode=|initial states|relative difference|verdict|times:|^ERROR' "$log" | cut -c1-400 >> "$out/summary.txt"
}
run pieces-all-seeds "${J[@]}" scripts/enzyme/spike/pieces.jl gather push solve kinetic
PM_SEED_FIELDS=1 run pieces-field-seeds "${J[@]}" scripts/enzyme/spike/pieces.jl current push+current step "two steps"
for args in "fast 1 1 Float64 7" "fast 20 1 Float64 7" "fast 20 1 Float32 7" "fast 100 1 Float64 3" "fast 20 2 Float64 11"; do
    run "chain-${args// /-}" "${J[@]}" scripts/enzyme/chain_rule.jl $args
done
echo "== done $(date -u +%H:%M:%S), logs in $out" | tee -a "$out/summary.txt"
