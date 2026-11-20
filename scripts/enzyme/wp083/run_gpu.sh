#!/usr/bin/env bash
# WP-083 on a GPU, in two worktrees prepared by prepare.sh (git steps, outside any sandbox):
#   A = this branch (claude/project-thread-xjh4di): T-1140 and T-1141 only.
#   B = WP-078's study branch (claude/project-thread-kjjfr2-gpu) with this branch's sincos change
#       applied, its antenna reverse rule left out: Enzyme's reverse through the trapped-beam case's
#       step! (antenna, window, PMLs, ionization) against duals.
# Usage: run_gpu.sh <worktree A> <worktree B> <out dir>; through scripts/gpu-lock.sh on a shared GPU.
set -u
A=$1; B=$2; OUT=$3
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
echo "[$(stamp)] A: instantiate"
(cd "$A" && julia +1.13 --project -e 'using Pkg; Pkg.instantiate(workspace = true)') > "$OUT/a-instantiate.log" 2>&1
echo "[$(stamp)] A: T-1140"
(cd "$A" && PM_TEST_GPU=1 julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/wp083/t1140_gpu.jl) > "$OUT/t1140.log" 2>&1
echo "[$(stamp)] A: exit $?"
echo "[$(stamp)] B: instantiate"
(cd "$B" && julia +1.13 --project -e 'using Pkg; Pkg.instantiate(workspace = true)') > "$OUT/b-instantiate.log" 2>&1
P="scripts/enzyme/active_reduce.jl;$A/scripts/enzyme/wp083/unrolled_species_noantenna.jl"
run_b() {
    local name=$1; shift
    echo "[$(stamp)] B: $name"
    (cd "$B" && PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_ENZYME_PATCH="$P" JULIA_LOAD_PATH="@:$B:@stdlib" \
        timeout 5400 julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl "$@") > "$OUT/$name.log" 2>&1
    echo "[$(stamp)] B: $name exit $?"
}
# while the antenna emits (the pulse's peak crosses the plane near t = 54): 1, 4 and 16 steps (#548: ≤ 20)
run_b steps-t50 steps tmid=50 ks=1,4,16
# the whole run's gradient in the four free parameters, segments of 20 steps, against Dual{4}
run_b grad-c20 grad C=20 ckpt=device
(cd "$B" && ls -d scripts/runs/enzyme-trapped-* 2>/dev/null | tail -2 | xargs -I{} cp -r {} "$OUT/")
echo "[$(stamp)] done"
