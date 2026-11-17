#!/bin/bash
# WP-041 on a GPU: the independent checks and timings of #469 (the factored stencil), #470 (the sort
# by its keys and the sort from the last order), #476 (nearest-node bins) and #480 (the segment
# bounds from the sorted keys), one job at a time:
#
#   bash run_gpu.sh <dir> <stage>...      # the stages in the order given; `checks`, `timings` and `all` are lists of them
#
# <dir> takes the trees, one git worktree per commit, made by the `setup` stage from the checkout
# this script lies in: wp041-base (before #469), wp041-s469 (with it), wp041-s470 (with #470),
# wp041-old3 (with #476, without #480) and wp041-bins (the head of #480; BINS=<commit> for another).
# The study's scripts (scripts/deposition of this branch) are copied into each tree.
# Results go to scripts/runs/wp041-gpu/ of this script's own tree: logs/<stage>.log, one line per
# stage in `status` (the last one is ALL-DONE), and the `report` stage's report-*.md, which is what
# to send back. While a run is going, `WP041_INNER=1 bash run_gpu.sh <dir> report` writes the reports
# of what has finished without touching `status`. A failed `setup` or `tests_env` ends the run.
# A stage that uses the GPU runs through scripts/gpu-lock.sh (LOCK=<wrapper> for another, LOCK=env
# for none). BACKEND=cpu runs the same on the CPU backend (a dry run), THREADS is julia's thread
# count (8), JULIA the launcher ("julia +1.13").
set -u
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
H=$(dirname "$SELF")
TREE=$(cd "$H/../../.." && pwd)
W=$(mkdir -p "$1" && cd "$1" && pwd); shift
R=$TREE/scripts/runs/wp041-gpu
BASE=2cfcf7a
S469=a324826
S470=dc47d9b
OLD3=6a729fd
BINS=${BINS:-3989e4c}
BACKEND=${BACKEND:-cuda}
THREADS=${THREADS:-8}
LOCK=${LOCK:-$TREE/scripts/gpu-lock.sh}
J=${JULIA:-julia +1.13}
GPU=1; [ "$BACKEND" = cuda ] || GPU=0
DEP='engine: patch,atomics,factored stencil'
SORT='engine sort,engine resort,permutation form'
CHECKS="gpu_info flux_check proto_bits flux_tests resort_check paths_check bins_check module_bins resort_bins paths_bins segs_check core_suite em_tests"
TIMINGS="bench_bins bench_s470 sort_parts bench_s469 bench_base bench3d_s469 bench3d_s470 bench_bins2 bench_s470b"
mkdir -p "$R/logs"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$R/status"; }
t() { echo "$W/wp041-$1"; }

setup() {
    cd "$TREE" && git fetch -q origin main && git fetch -q origin refs/pull/480/head || return 1
    git rev-parse HEAD > "$R/study-commit.txt"
    : > "$R/tree-commits.txt"
    for w in base:$BASE s469:$S469 s470:$S470 old3:$OLD3 bins:$BINS; do
        d=$(t ${w%%:*})
        [ -d "$d" ] || git worktree add -q --detach "$d" ${w##*:} || return 1
        mkdir -p "$d/scripts/deposition" && cp "$TREE"/scripts/deposition/*.jl "$d/scripts/deposition/"
        echo "${w%%:*} $(cd "$d" && git rev-parse HEAD)" >> "$R/tree-commits.txt"
        $J --project="$d/scripts" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || return 1
    done
}
tests_env() {
    for p in s469/lib/ParticleMeshCore bins/lib/ParticleMeshElectromagnetic; do
        $J --project="$W/wp041-$p/test" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || return 1
    done
    cd "$(t bins)" && $J --project -e 'using Pkg; Pkg.instantiate()'
}
gpu_info() {
    [ $GPU = 1 ] || { echo "backend $BACKEND"; return 0; }
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader | tee "$R/gpu.txt"
    $J --project="$(t bins)/scripts" -e 'using CUDACore; println(CUDACore.functional(), " ", CUDACore.name(CUDACore.device()))'
}
check() { $J -t $THREADS --project="$(t $1)/scripts" "$H/$2" "$(t $1)" $BACKEND "${@:3}"; }      # tree, script, more arguments
# #469: the flux on states the suite does not use, the prototype's bits, and T-300, T-302, T-840 on the device
flux_check() { check s469 flux_check.jl; }
proto_bits() { check s469 proto_bits.jl; }
flux_tests() { PM_TEST_GPU=$GPU $J -t $THREADS --project="$(t s469)/lib/ParticleMeshCore/test" "$H/gpu_flux_tests.jl" "$(t s469)" $BACKEND; }
# #470: the sort from the last order against the CPU's full sort, and the module's step along the ways a state reaches it
resort_check() { check s470 resort_check.jl; }
paths_check() { check s470 state_paths_check.jl; }
# #476 and #480: the deposits and the sort by nearest-node bins against the cells' and against the CPU's, the module's
# step with and without the bins, the sort from the last order by bins, the state paths with the bins as the default,
# and the full sort against a host reference
bins_check() { check bins bins_check.jl; }
module_bins() { check bins module_bins_check.jl; }
resort_bins() { for b in nodes mixed centres; do check bins resort_check.jl $b || return 1; done; }
paths_bins() { check bins state_paths_check.jl; }
segs_check() { check bins segs_check.jl > "$R/segs_$BACKEND.txt"; rc=$?; tail -n 3 "$R/segs_$BACKEND.txt"; return $rc; }
# the engine's whole suite on the head of #480 with its GPU test sets required (T-190 to T-198, and T-876, the 3D
# reruns of #471), and the module's GPU test sets that step in 2D there (the module bins by node)
core_suite() { cd "$(t bins)" && PM_TEST_GPU=$GPU JULIA_NUM_THREADS=$THREADS $J --project=lib/ParticleMeshCore -e 'using Pkg; Pkg.test()'; }
em_tests() {
    PM_TEST_GPU=$GPU PM_EM_SET=sort $J -t $THREADS --project="$(t bins)/lib/ParticleMeshElectromagnetic/test" "$H/em_gpu_tests.jl" "$(t bins)" \
        $([ $GPU = 1 ] || echo cpu)
}
bench() {       # tree, rows, sizes, elements
    cd "$(t $1)" && PM_BENCH_ONLY="$2" $J -t $THREADS --project=scripts scripts/deposition/bench2d.jl $BACKEND $3 $4
}
SIZES=${SIZES:-S,M,L}
bench_bins() { bench bins "$DEP,$SORT" $SIZES f32,f64,d32x3; }      # the rows by cell, and by node bins on both axes and on y alone
bench_s470() { bench s470 "$SORT" $SIZES f32,f64,d32x3; }
bench_s469() { bench s469 "$DEP,$SORT" $SIZES f32,f64,d32x3; }
bench_base() { bench base "$DEP" $SIZES f32,f64,d32x3; }
bench_bins2() { bench bins "$DEP,$SORT" $SIZES f32,f64; }           # a second pass, for the spread
bench_s470b() { bench s470 "$SORT" $SIZES f32,f64; }
# #480: the sort's parts timed alone, before and after it
sort_parts() {
    for w in old3 bins; do for size in ${PARTS:-224x64 896x256 1792x512}; do
        echo "== $w $size"; check $w sort_parts.jl $size 4 || return 1
    done; done
}
bench3d() {     # tree, cells, elements
    cd "$(t $1)" && PM_BENCH_ONLY="$SORT" $J -t $THREADS --project=scripts scripts/deposition/bench3d.jl $BACKEND $2 8 $3
}
bench3d_s469() { bench3d s469 ${CUBE:-64x64x64} f32,f64,d32 && { [ -n "${CUBE:-}" ] || bench3d s469 128x128x128 f32; }; }
bench3d_s470() { bench3d s470 ${CUBE:-64x64x64} f32,f64,d32 && { [ -n "${CUBE:-}" ] || bench3d s470 128x128x128 f32; }; }
# What to send back: the checks' outcome in one file, each timing log in one of its own, home directory replaced by ~
report() {
    clean() { sed "s#$HOME#~#g"; }
    {
        echo "# WP-041 GPU run: checks"; echo; echo '```'
        cat "$R/gpu.txt" 2>/dev/null; echo "study $(cat "$R/study-commit.txt")"; cat "$R/tree-commits.txt"; echo; cat "$R/status"; echo '```'
        for s in $CHECKS; do
            f=$R/logs/$s.log; [ -f "$f" ] || continue
            echo; echo "## $s"; echo '```'
            grep -n "FAILED\|DIFFER\|ABOVE\|ERROR\|Error\|Test Failed\|Broken\|Exception\|gpu.jl\|GPU tests" "$f" | cut -c1-300 | head -n 30
            echo "[last lines]"; tail -n ${TAIL:-30} "$f" | cut -c1-400; echo '```'
        done
    } | clean > "$R/report-1-checks.md"
    for s in $TIMINGS; do
        f=$R/logs/$s.log; [ -f "$f" ] || continue
        { echo "# WP-041 GPU run: $s ($(cat "$R/gpu.txt" 2>/dev/null))"; echo; cut -c1-400 "$f"; } | clean > "$R/report-2-$s.md"
    done
    ls -la "$R"/report-*.md
}

{       # one group, so that the shell has read the whole file before it runs a stage
if [ -n "${WP041_INNER:-}" ]; then "$@"; exit $?; fi
stages=()
for s in "$@"; do
    case $s in
        checks) stages+=($CHECKS) ;;
        timings) stages+=($TIMINGS) ;;
        all) stages+=(setup tests_env $CHECKS $TIMINGS report) ;;
        *) stages+=($s) ;;
    esac
done
for stage in "${stages[@]}"; do
    st "$stage started"
    case $stage in
        setup | tests_env | report) $stage > "$R/logs/$stage.log" 2>&1 ;;
        *) if [ $GPU = 1 ] && [ "$LOCK" != env ]; then
               WP041_INNER=1 "$LOCK" bash "$SELF" "$W" $stage > "$R/logs/$stage.log" 2>&1
           else
               $stage > "$R/logs/$stage.log" 2>&1
           fi ;;
    esac
    rc=$?
    st "$stage rc=$rc"
    case $stage in setup | tests_env) [ $rc = 0 ] || { st "ABORTED: $stage failed"; break; } ;; esac
done
st "ALL-DONE"
exit 0
}
