#!/usr/bin/env bash
# WP-088 on CUDA: the per-piece dot-product checks of the λ₀/32 trapped-beam step
# (spike/trapped_pieces.jl ... check), each piece where the step runs it (PM_CHECK_CTX=1), each in
# its own process. Run in a worktree of this branch, through the machine's GPU queue or lock:
#     scripts/enzyme/wp088_checks.sh <worktree> <out dir> [piece[:v regex] ...]
# Without pieces, the list below; `ionize:pending` puts the random tangent on the arrays whose path
# matches `pending` only (PM_CHECK_V).
set -u
W=$1; OUT=$2; shift 2
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
cd "$W"
git rev-parse HEAD > "$OUT/checks-commit.txt"
pieces=("$@")
[ ${#pieces[@]} -eq 0 ] && pieces=(ionize ionize:pending ionize_nobirth current step)
for pc in "${pieces[@]}"; do
    name="${pc%%:*}"; v=""
    [ "$pc" != "$name" ] && v="${pc#*:}"
    log="$OUT/check-$name${v:+-$v}.log"
    echo "[$(stamp)] check $name ${v:+(v on $v)}"
    PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_CHECK_CTX=1 PM_CHECK_V="$v" PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/active_reduce.jl}" \
        JULIA_LOAD_PATH="@:$W:@stdlib" timeout 3600 julia +1.13 --startup=no -t 2 --project=lib/ParticleMeshElectromagnetic/test \
        scripts/enzyme/spike/trapped_pieces.jl "$name" 180 32 2 check > "$log" 2>&1
    echo "[$(stamp)] check $name: exit $?"
done
echo "[$(stamp)] checks done"
