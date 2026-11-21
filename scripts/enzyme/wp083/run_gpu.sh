#!/usr/bin/env bash
# WP-083 on a GPU, in a worktree of this branch (claude/project-thread-xjh4di), made beforehand:
#     git fetch origin claude/project-thread-xjh4di
#     git worktree add --detach <worktree> origin/claude/project-thread-xjh4di
# Usage: run_gpu.sh <worktree> <out dir>. Runs T-1140 and T-1141, then Enzyme's reverse through the
# trapped-beam case's step! (antenna, window, PMLs, ionization) against duals, with the CUDA package
# loaded (Enzyme's CUDA extension) and only active_reduce.jl, the refill's flag sum and the sinpi
# rule left of WP-078's patches: no antenna rule, copy kernels, unrolled loops, four-kernel
# ionization or freed-array skip. Needs the card to itself (nvidia-smi first).
set -u
W=$1; OUT=$2
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
cd "$W"
git rev-parse HEAD > "$OUT/commit.txt"
nvidia-smi > "$OUT/nvidia-smi-start.txt" 2>&1
echo "[$(stamp)] instantiate"
julia +1.13 --project -e 'using Pkg; Pkg.instantiate(workspace = true)' > "$OUT/instantiate.log" 2>&1
echo "[$(stamp)] T-1140"
PM_TEST_GPU=1 julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/wp083/t1140_gpu.jl > "$OUT/t1140.log" 2>&1
echo "[$(stamp)] T-1140 exit $?"
P="scripts/enzyme/active_reduce.jl;scripts/enzyme/spike/unrolled_species.jl"
run_t() {
    local name=$1; shift
    echo "[$(stamp)] trapped: $name"
    PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_ENZYME_PATCH="$P" JULIA_LOAD_PATH="@:$W:@stdlib" \
        timeout 5400 julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl "$@" > "$OUT/$name.log" 2>&1
    echo "[$(stamp)] trapped: $name exit $?"
}
# while the antenna emits (the pulse's peak crosses the plane near t = 54): 1, 4 and 16 steps (#548: ≤ 20)
run_t steps-t50 steps tmid=50 ks=1,4,16
# the whole run's gradient in the four free parameters, segments of 20 steps, device checkpoints, against Dual{4}
run_t grad-c20 grad C=20 ckpt=device
cp -r scripts/runs/enzyme-trapped-* "$OUT/" 2>/dev/null
echo "[$(stamp)] done"
