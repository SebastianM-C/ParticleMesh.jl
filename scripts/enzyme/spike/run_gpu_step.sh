#!/usr/bin/env bash
# WP-050 spike on CUDA: which part of step! makes Enzyme abort, one process per check.
#     scripts/gpu-lock.sh scripts/enzyme/spike/run_gpu_step.sh
set -u
cd "$(dirname "$0")/../../.."
out="scripts/runs/enzyme-gpu-step-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/spike/active_args.jl
git rev-parse HEAD > "$out/commit.txt"
for t in "kinetic" "check_params" "step without kinetic" "step without check_params"; do
    log="$out/kernel-${t// /-}.log"
    echo "== $(date -u +%H:%M:%S) $t" | tee -a "$out/summary.txt"
    ONLY="$t" timeout 2400 "${J[@]}" scripts/enzyme/spike/kernels.jl > "$log" 2>&1
    echo "   exit $?" >> "$out/summary.txt"
    grep -E '^\[|^ERROR|illegal insertion|UNREACHABLE' "$log" | cut -c1-400 >> "$out/summary.txt"
done
echo "== done $(date -u +%H:%M:%S), logs in $out" | tee -a "$out/summary.txt"
