#!/usr/bin/env bash
R=~/pm-wp078-cuda-out/b3run.sh
bash $R r4-nounroll "copy,unroll" 1 grad C=20 ckpt=host
bash $R r5-noionize4 "copy,ionize4" 1 grad C=20 ckpt=host
bash $R r6-nofreedskip "copy,freedskip" 1 grad C=20 ckpt=host
bash $R r7-minimal "copy,unroll,ionize4,freedskip" 1 grad C=20 ckpt=host
echo queue2-done >> ~/pm-wp078-cuda-out/status
