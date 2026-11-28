#!/usr/bin/env bash
# WP-088 on CUDA: trapped.jl runs (the reverse gradient at several segment lengths, the
# one-call-against-chained-calls check) on the λ₀/32 trapped-beam case, each in its own process
# (a CUDA error leaves its context unusable), with nvidia-smi sampling the card beside each.
# Run in a worktree of this branch, through the machine's GPU queue or lock:
#     scripts/enzyme/wp088_batch.sh <worktree> <out dir> [run ...]
# A run is trapped.jl's arguments as one string; without any, the list below. Logs, results
# and manifests land in <out dir>.
set -u
W=$1; OUT=$2; shift 2
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
cd "$W"
git rev-parse HEAD > "$OUT/commit.txt"
git log --oneline -3 >> "$OUT/commit.txt"
nvidia-smi > "$OUT/nvidia-smi-start.txt" 2>&1
echo "[$(stamp)] instantiate"
julia +1.13 --project -e 'using Pkg; Pkg.instantiate(workspace = true)' > "$OUT/instantiate.log" 2>&1
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(
    "grad C=20 ckpt=device wp=WP-088"
    "grad C=40 ckpt=device wp=WP-088"
    "grad C=40 ckpt=device inner=1 wp=WP-088"
    "segcheck s0=900 ns=1,2,5,10,20,30,40 ckpt=host wp=WP-088")
i=0
for r in "${runs[@]}"; do
    i=$((i + 1))
    name="$(printf '%02d' $i)-${r// /-}"
    name="${name//=/}"
    echo "[$(stamp)] $r"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 500 > "$OUT/$name.smi.csv" 2>&1 &
    SMI=$!
    PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/active_reduce.jl}" JULIA_LOAD_PATH="@:$W:@stdlib" \
        timeout 5400 julia +1.13 --startup=no -t 2 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl $r \
        > "$OUT/$name.log" 2>&1
    echo "[$(stamp)] $r: exit $?"
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
done
# then the per-piece dot-product checks in the step's context (wp088_checks.sh), unless WP088_CHECKS=0
[ "${WP088_CHECKS:-1}" = 1 ] && bash scripts/enzyme/wp088_checks.sh "$W" "$OUT"
cp -r scripts/runs/enzyme-trapped-* "$OUT/" 2>/dev/null
nvidia-smi > "$OUT/nvidia-smi-end.txt" 2>&1
echo "[$(stamp)] done"
