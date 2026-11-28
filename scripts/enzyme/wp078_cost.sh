#!/usr/bin/env bash
# WP-078's per-method cost on the λ₀/32 trapped-beam case: a plain primal run, Dual{1}, Dual{4}
# and the reverse gradient (C = 20, device checkpoints), each in its own process
# (`trapped.jl cost`), with nvidia-smi sampling the card's memory every 100 ms beside each.
# Run in a worktree of this branch, with the card to itself (no other process in nvidia-smi):
#     scripts/enzyme/wp078_cost.sh <worktree> <out dir> [julia threads, default 2]
# The reverse gets faster with more Julia threads, for a reason not measured: on the RTX 5090 its
# compiled reverse phase took 40.2 s at -t 1, 29.9 s at -t 2 and 23.5 s at -t 16.
# The result.toml and manifest.toml of each method land in <out dir>/enzyme-trapped-cost-<method>-*.
set -u
W=$1; OUT=$2; NT=${3:-2}
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
cd "$W"
git rev-parse HEAD > "$OUT/commit.txt"
nvidia-smi > "$OUT/nvidia-smi-start.txt" 2>&1
echo "[$(stamp)] instantiate"
julia +1.13 --project -e 'using Pkg; Pkg.instantiate(workspace = true)' > "$OUT/instantiate.log" 2>&1
# The runs in reports/wp-078-reverse-mode-at-production-size.md also loaded
# scripts/enzyme/spike/unrolled_species.jl, at b3e46201. Since WP-086 the engine carries its
# flag sum and ramp (through sin rather than a sinpi rule), so only active_reduce.jl is patched in.
P="scripts/enzyme/active_reduce.jl"
for m in primal dual1 dual4 reverse; do
    reps=3; [ "$m" = reverse ] && reps=2
    echo "[$(stamp)] cost: $m"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 100 > "$OUT/smi-$m.csv" 2>&1 &
    SMI=$!
    PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_ENZYME_PATCH="$P" JULIA_LOAD_PATH="@:$W:@stdlib" \
        timeout 5400 julia +1.13 -t $NT --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl cost method=$m reps=$reps \
        > "$OUT/cost-$m.log" 2>&1
    echo "[$(stamp)] cost: $m exit $?"
    kill $SMI; wait $SMI 2>/dev/null
done
cp -r scripts/runs/enzyme-trapped-cost-* "$OUT/" 2>/dev/null
nvidia-smi > "$OUT/nvidia-smi-end.txt" 2>&1
echo "[$(stamp)] done"
