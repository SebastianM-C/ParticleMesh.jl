#!/usr/bin/env bash
# WP-080 on CUDA: bisect3d.jl for every piece of the 3D step, each in its own process (an Enzyme
# compiler crash takes the process with it), plus the 2D step as the control.
#     scripts/gpu-lock.sh scripts/enzyme/run_bisect3d.sh [pieces...]
# Logs and a summary go to scripts/runs/enzyme-bisect3d-gpu-<stamp>/.
set -u
cd "$(dirname "$0")/../.."
out="scripts/runs/enzyme-bisect3d-gpu-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
J=(julia +1.13 --startup=no -t 2 --project=lib/ParticleMeshElectromagnetic/test)
export PM_BACKEND=cuda PM_ENZYME_PATCH="${PM_ENZYME_PATCH:-scripts/enzyme/spike/active_args.jl}"
echo "# commit $(git rev-parse HEAD), patches $PM_ENZYME_PATCH, started $(date -u +%FT%TZ)" > "$out/summary.txt"
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=("step dim=2" solve fold kinetic push sort deposit current step_nopush step "step T=Float64 mode=reference")
i=0
for r in "${runs[@]}"; do
    i=$((i + 1))
    name="$(printf '%02d' $i)-${r// /-}"
    timeout 2400 "${J[@]}" scripts/enzyme/bisect3d.jl $r > "$out/$name.log" 2>&1
    code=$?
    echo "== $r: exit $code at $(date -u +%H:%M:%S); $(grep -h '^bisect:.*ok in' "$out/$name.log" | cut -c1-200)" >> "$out/summary.txt"
    [ $code -ne 0 ] && grep -m1 -hE 'signal|ERROR' "$out/$name.log" | cut -c1-300 >> "$out/summary.txt"
done
echo "# finished $(date -u +%FT%TZ)" >> "$out/summary.txt"
