#!/usr/bin/env bash
# usage: b3run.sh <tag> <PM_WA_OFF> <load_cuda 0|1> <trapped.jl args...>  (WP-078 CUDA follow-up, RTX 5090)
tag=$1; off=$2; cuda=$3; shift 3
WT=~/pm-wp078-cuda
O=~/pm-wp078-cuda-out; mkdir -p $O
cd $WT
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
echo "$tag start $(date -u +%H:%M:%S) others: $others free: $free" >> $O/status
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 1000 > $O/mem-$tag.log &
SMI=$!
s=$(date +%s)
env PM_WA_OFF="$off" PM_LOAD_CUDA=$cuda PM_GPU_OTHERS="$others" PM_GPU_FREE="$free" PM_ENZYME_ALL_OVERWRITTEN=0 PM_ENZYME_NO_CONST=0 \
  PM_BACKEND=cuda PM_CUDA_HEAP=1073741824 PM_ENZYME_PATCH="scripts/enzyme/active_reduce.jl;scripts/enzyme/spike/unrolled_species.jl" \
  JULIA_LOAD_PATH="@:$WT:$HOME/pm-wp078-cuda-env:@stdlib" \
  julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/trapped.jl "$@" > $O/run-$tag.log 2>&1
rc=$?
kill $SMI
peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>p) p=$2+0} END {print p}' $O/mem-$tag.log)
echo "$tag exit $rc $(( $(date +%s) - s ))s peak_MiB $peak" >> $O/status
