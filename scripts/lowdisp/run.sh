#!/usr/bin/env bash
# run.sh <rundir>: Smilei on 4 OpenMP threads, timing into container.txt
set -u
cd "$1"
start=$(date +%s)
OMP_NUM_THREADS=${THREADS:-4} /opt/smilei/smilei namelist.py > smilei.log 2>&1
st=$?
echo "# end: exit $st, $(( $(date +%s) - start )) s" > container.txt
