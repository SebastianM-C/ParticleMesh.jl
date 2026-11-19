#!/usr/bin/env bash
# WP-050 spike on CUDA: minimal kernels, the step's kernels one at a time, then the chain-rule
# check through whole steps. Run from the repository root through the GPU lock:
#     scripts/gpu-lock.sh scripts/enzyme/spike/run_gpu.sh
# Each run's log goes to scripts/runs/enzyme-gpu-<stamp>/; `summary.txt` collects the result lines.
set -u
cd "$(dirname "$0")/../../.."
out="scripts/runs/enzyme-gpu-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda
A=scripts/enzyme/spike/active_args.jl
U=scripts/enzyme/spike/unroll.jl
LIMIT="${PM_SPIKE_TIMEOUT:-2400}"
run() {  # run <log name> <command...>
    local log="$out/$1.log"; shift
    echo "== $(date -u +%H:%M:%S) $*" | tee -a "$out/summary.txt"
    timeout "$LIMIT" "$@" > "$log" 2>&1
    echo "   exit $?" >> "$out/summary.txt"
    grep -E '^\[|^mode=|relative difference|verdict|times:|dropped|^ERROR' "$log" | cut -c1-400 >> "$out/summary.txt"
}
git rev-parse HEAD > "$out/commit.txt"
"${J[@]}" -e 'using Pkg; Pkg.instantiate(); using CUDACore; println(CUDACore.name(CUDACore.device()), " functional=", CUDACore.functional())' > "$out/instantiate.log" 2>&1
tail -1 "$out/instantiate.log" | tee -a "$out/summary.txt"
run minimal "${J[@]}" scripts/enzyme/spike/minimal.jl
PM_ENZYME_PATCH="$A" run kernels-fast-active "${J[@]}" scripts/enzyme/spike/kernels.jl
PM_ENZYME_PATCH="$A;$U" run kernels-fast-active-unroll "${J[@]}" scripts/enzyme/spike/kernels.jl
PM_MODE=reference PM_ENZYME_PATCH="$A;$U" run kernels-reference-active-unroll "${J[@]}" scripts/enzyme/spike/kernels.jl
# whole steps: the smallest patch set that passes one step is used for the longer runs
P="$A"
PM_ENZYME_PATCH="$P" run chain-fast-1-Float64 "${J[@]}" scripts/enzyme/chain_rule.jl fast 1 1 Float64 7
if ! grep -q "verdict pass" "$out/chain-fast-1-Float64.log"; then
    P="$A;$U"
    PM_ENZYME_PATCH="$P" run chain-fast-1-Float64-unroll "${J[@]}" scripts/enzyme/chain_rule.jl fast 1 1 Float64 7
fi
echo "patch set for the longer runs: $P" >> "$out/summary.txt"
for args in "fast 20 1 Float64 7" "fast 20 1 Float32 7" "reference 20 1 Float64 7" "fast 100 1 Float64 3" "fast 20 2 Float64 11"; do
    PM_ENZYME_PATCH="$P" run "chain-${args// /-}" "${J[@]}" scripts/enzyme/chain_rule.jl $args
done
echo "== done $(date -u +%H:%M:%S), logs in $out" | tee -a "$out/summary.txt"
