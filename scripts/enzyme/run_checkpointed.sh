#!/usr/bin/env bash
# WP-080 on CUDA: checkpointed reverse mode through the 2D and 3D step (checkpointed.jl), each run
# its own process (a CUDA error leaves its context unusable), at the default device heap.
#     scripts/gpu-lock.sh scripts/enzyme/run_checkpointed.sh [runs...]
# A run is checkpointed.jl's arguments as one string, "check dim=2 ..." or "cost dim=3 ...";
# without any, the list below. Logs and a summary go to scripts/runs/enzyme-checkpointed-gpu-<stamp>/.
set -u
cd "$(dirname "$0")/../.."
out="scripts/runs/enzyme-checkpointed-gpu-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no -t 2 --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/spike/active_args.jl}"
{ echo "# commit $(git rev-parse HEAD), patches $PM_ENZYME_PATCH, started $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; } > "$out/summary.txt"
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(
    "check dim=2 T=Float64 mode=reference steps=32 c=4 plain=1"
    "check dim=2 T=Float32 mode=fast steps=32 c=4 plain=1"
    "check dim=3 T=Float64 mode=reference steps=16 c=3 plain=1"
    "check dim=3 T=Float32 mode=fast steps=16 c=3 plain=1"
    "check dim=2 T=Float64 mode=reference steps=400 c=10 plain=0"
    "check dim=3 T=Float64 mode=reference steps=200 c=8 plain=0"
    "cost dim=2 T=Float32 mode=fast cells=1024x1024 ppc=2x2 steps=8,32,128,512 plain=16 dual=128 c=16"
    "cost dim=2 T=Float32 mode=reference cells=1024x1024 ppc=2x2 steps=8,32,128,512 plain=16 dual=128 c=16"
    "cost dim=3 T=Float32 mode=fast cells=96x96x96 ppc=2x2x2 steps=8,32,128 plain=8 dual=32 c=8"
    "cost dim=3 T=Float32 mode=reference cells=96x96x96 ppc=2x2x2 steps=8,32,128 plain=8 dual=32 c=8")
i=0
for r in "${runs[@]}"; do
    i=$((i + 1))
    name="$(printf '%02d' $i)-${r// /-}"
    echo "== $(date -u +%H:%M:%S) $r" >> "$out/summary.txt"
    timeout 5400 "${J[@]}" scripts/enzyme/checkpointed.jl $r > "$out/$name.log" 2>&1
    echo "   exit $? at $(date -u +%H:%M:%S)" >> "$out/summary.txt"
    grep -hE '^check|^cost|first call|relative difference|chain rule|host storage|first calls|^  steps|failed|^wrote|^ERROR' "$out/$name.log" | cut -c1-400 >> "$out/summary.txt"
done
echo "# finished $(date -u +%FT%TZ)" >> "$out/summary.txt"
