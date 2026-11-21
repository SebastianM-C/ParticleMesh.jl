#!/usr/bin/env bash
# R6 and R7 again, each once the card has at least 26 GB free (the grid/ppc campaign shares it)
R=~/pm-wp078-cuda-out/b3run.sh
waitfree() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" -ge 26000 ]; do sleep 60; done; }
waitfree; bash $R r6b-nofreedskip "copy,freedskip" 1 grad C=20 ckpt=host
waitfree; bash $R r7b-minimal "copy,unroll,ionize4,freedskip" 1 grad C=20 ckpt=host
echo queue3-done >> ~/pm-wp078-cuda-out/status
