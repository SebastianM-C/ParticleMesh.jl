#!/usr/bin/env bash
# The second version of the grid and particle budget on one CUDA GPU, one stage after the other:
# gridppc.jl's v2 campaign (2D, solvers × gathers), gridppc3d.jl's g3 campaign (3D), then the
# timing2 campaign with one process per job (run_timing.sh). Each stage logs into <out>, and
# <out>/status.txt has a line at each start and end; `pack` writes <out>/results-v2.tar.gz without
# the frame binaries and <out>/frames-v2.tar.gz with them.
#
#   scripts/resolution/run_v2_gpu.sh <out dir> [stage...]     stages: v2 g3 timing2 (default: all three)
#   scripts/resolution/run_v2_gpu.sh <out dir> pack
set -u
out=$(mkdir -p "$1" && cd "$1" && pwd)
shift
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "pack" ]; then
    tar czf "$out/results-v2.tar.gz" -C "$out" --exclude='*.frames.bin' --exclude='*.tar.gz' .
    find "$out" -name '*.frames.bin' -printf '%P\n' | tar czf "$out/frames-v2.tar.gz" -C "$out" -T -
    ls -la "$out"/*.tar.gz
    exit 0
fi
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(v2 g3 timing2)
for s in "${stages[@]}"; do
    echo "== $s started $(date -u +%FT%TZ) at $(git rev-parse --short HEAD)" >> "$out/status.txt"
    case "$s" in
        v2) julia +1.13 --project=scripts -t "${PM_THREADS:-8}" scripts/resolution/gridppc.jl campaign v2 "$out/v2" cuda >> "$out/v2.log" 2>&1 ;;
        g3) julia +1.13 --project=scripts -t "${PM_THREADS:-8}" scripts/resolution/gridppc3d.jl campaign g3 "$out" cuda >> "$out/g3.log" 2>&1 ;;
        timing2) scripts/resolution/run_timing.sh "$out" timing2 ;;
        *) echo "unknown stage $s" >> "$out/status.txt"; continue ;;
    esac
    echo "== $s exit $? $(date -u +%FT%TZ)" >> "$out/status.txt"
done
echo "# all finished $(date -u +%FT%TZ)" >> "$out/status.txt"
