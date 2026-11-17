#!/usr/bin/env bash
# Run campaigns of gridppc.jl one after the other on a CUDA GPU, each into <out>/<campaign>, with
# a log per campaign and a status file. A file <out>/PAUSE makes the running campaign wait between
# jobs. `pack` writes <out>/results.tar.gz without the frame binaries.
#
#   scripts/resolution/run_gpu.sh <out dir> <campaign>...
#   scripts/resolution/run_gpu.sh <out dir> pack
set -u
out=$(mkdir -p "$1" && cd "$1" && pwd)
shift
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "pack" ]; then
    tar czf "$out/results.tar.gz" -C "$out" --exclude='*.frames.bin' --exclude='results.tar.gz' .
    ls -la "$out/results.tar.gz"
    exit 0
fi
for c in "$@"; do
    echo "== $c started $(date -u +%FT%TZ)" >> "$out/status.txt"
    julia +1.13 --project=scripts -t "${PM_THREADS:-8}" scripts/resolution/gridppc.jl campaign "$c" "$out/$c" cuda >> "$out/$c.log" 2>&1
    echo "== $c exit $? $(date -u +%FT%TZ)" >> "$out/status.txt"
done
echo "# all finished $(date -u +%FT%TZ)" >> "$out/status.txt"
