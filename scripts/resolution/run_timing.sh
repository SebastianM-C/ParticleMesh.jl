#!/usr/bin/env bash
# Wall times and device memory of gridppc.jl's `timing` campaign on a CUDA GPU: each job in a
# julia process of its own (`gridppc.jl timed`, which runs the job twice and keeps the second run),
# one after the other, and only while no other process computes on the card. A job whose JSON is
# in <out>/timing is not run again. Next to the results: a log per job, status.txt, and
# gpu-samples.csv with every compute process's memory once a second (time, pid, MiB), from which
# a job's peak can be read independently of the driver's own count.
#
#   scripts/resolution/run_timing.sh <out dir>
#
# PM_IDLE_MIB (default 300): another compute process holding more than this keeps a job waiting;
# PM_IDLE_WAIT (default 21600 s): after that long the script stops with exit 75, to be resubmitted.
set -u
out=$(mkdir -p "$1" && cd "$1" && pwd)
cd "$(dirname "$0")/../.."
dir="$out/timing"
mkdir -p "$dir"
idle_mib=${PM_IDLE_MIB:-300}
idle_wait=${PM_IDLE_WAIT:-21600}

( while true; do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$(date -u +%FT%TZ), /"
      sleep 1
  done >> "$out/gpu-samples.csv" ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

# the largest memory another compute process holds on the card, in MiB
others() {
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1
}

echo "== timing started $(date -u +%FT%TZ)" >> "$out/status.txt"
for id in $(julia +1.13 --project=scripts scripts/resolution/gridppc.jl list timing | tail -n +2); do
    [ -f "$dir/$id.json" ] && continue
    waited=0
    while m=$(others); [ "${m:-0}" -gt "$idle_mib" ]; do
        if [ "$waited" -ge "$idle_wait" ]; then
            echo "== card busy for ${idle_wait} s, stopped before $id $(date -u +%FT%TZ)" >> "$out/status.txt"
            exit 75
        fi
        sleep 30
        waited=$((waited + 30))
    done
    echo "$id start $(date -u +%FT%TZ) waited ${waited}s" >> "$out/status.txt"
    julia +1.13 --project=scripts -t "${PM_THREADS:-8}" scripts/resolution/gridppc.jl timed timing "$id" "$dir" cuda > "$out/$id.log" 2>&1
    echo "$id exit $? $(date -u +%FT%TZ)" >> "$out/status.txt"
done
echo "# all finished $(date -u +%FT%TZ)" >> "$out/status.txt"
