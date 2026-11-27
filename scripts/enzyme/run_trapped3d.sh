#!/usr/bin/env bash
# WP-084, WP-086 on CUDA: Enzyme's checkpointed reverse pass through the 3D trapped-beam case
# (trapped3d.jl), each run its own process (a CUDA error leaves its context unusable), at the
# default device heap and again at 1 GiB if that fails: if the process errors, or its reverse pass
# fails for another reason than running out of the card's memory (trapped3d.jl then exits 2).
# nvidia-smi samples the card's memory beside every run; `pool used` in the logs is CUDA's memory
# pool, without the device malloc heap that Enzyme's tapes use, which only these samples show.
#     scripts/enzyme/run_trapped3d.sh [runs...]
# A run is trapped3d.jl's arguments as one string ("grad ppl=16 ..."); without any, the list
# below. Logs and a summary go to scripts/runs/enzyme-trapped3d-gpu-<stamp>/.
set -u
cd "$(dirname "$0")/../.."
out="scripts/runs/enzyme-trapped3d-gpu-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no -t 2 --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/active_reduce.jl}"
export JULIA_LOAD_PATH="@:$(pwd):@stdlib"
{ echo "# commit $(git rev-parse HEAD), patches $PM_ENZYME_PATCH, started $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader; } > "$out/summary.txt"
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(
    "grad ppl=16 dual=1,2,3,4 dual4=0 c=12 host=1"
    "grad ppl=16 dual= dual4=0 c=6 host=0 primal=0"
    "grad ppl=32 dual=1,2,3,4 dual4=0 c=12 host=1")
i=0
for r in "${runs[@]}"; do
    i=$((i + 1))
    name="$(printf '%02d' $i)-${r// /-}"
    for heap in default 1073741824; do
        echo "== $(date -u +%H:%M:%S) $r (heap $heap)" >> "$out/summary.txt"
        nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$out/$name-$heap.nvsmi.csv" 2>&1 &
        smi=$!
        if [ "$heap" = default ]; then
            timeout 14400 "${J[@]}" scripts/enzyme/trapped3d.jl $r > "$out/$name-$heap.log" 2>&1; rc=$?
        else
            PM_CUDA_HEAP=$heap timeout 14400 "${J[@]}" scripts/enzyme/trapped3d.jl $r > "$out/$name-$heap.log" 2>&1; rc=$?
        fi
        kill $smi 2>/dev/null; wait $smi 2>/dev/null
        echo "   exit $rc at $(date -u +%H:%M:%S)" >> "$out/summary.txt"
        grep -hE '^\[|dJ/d|^wrote|^ERROR|Out of|out of memory' "$out/$name-$heap.log" | cut -c1-400 >> "$out/summary.txt"
        [ $rc -eq 0 ] && break
    done
done
echo "# finished $(date -u +%FT%TZ)" >> "$out/summary.txt"
