#!/usr/bin/env bash
# WP-050 spike on CUDA, second stage: each part of the step against duals (pieces.jl), then the
# chain-rule check through whole steps with the scalar-argument patch only.
#     scripts/gpu-lock.sh scripts/enzyme/spike/run_gpu_chain.sh
set -u
cd "$(dirname "$0")/../../.."
out="scripts/runs/enzyme-gpu-chain-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl
LIMIT="${PM_SPIKE_TIMEOUT:-2400}"
git rev-parse HEAD > "$out/commit.txt"
run() {  # run <log name> <command...>
    local log="$out/$1.log"; shift
    echo "== $(date -u +%H:%M:%S) $*" | tee -a "$out/summary.txt"
    timeout "$LIMIT" "$@" > "$log" 2>&1
    echo "   exit $?" >> "$out/summary.txt"
    grep -E '^\[|^    input .*DIFFERS|^mode=|relative difference|verdict|times:|dropped|^ERROR|illegal insertion' "$log" | cut -c1-400 >> "$out/summary.txt"
}
run pieces "${J[@]}" scripts/enzyme/spike/pieces.jl solve push kinetic current push+current step "two steps"
for args in "fast 1 1 Float64 7" "fast 20 1 Float64 7" "fast 20 1 Float32 7" "fast 100 1 Float64 3" "reference 20 1 Float64 7" "fast 20 2 Float64 11"; do
    run "chain-${args// /-}" "${J[@]}" scripts/enzyme/chain_rule.jl $args
done
echo "== done $(date -u +%H:%M:%S), logs in $out" | tee -a "$out/summary.txt"
