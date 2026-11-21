#!/usr/bin/env bash
R=~/pm-wp078-cuda-out/b3run.sh
bash $R r1-base "" 1 grad C=20 ckpt=host
bash $R r2-nocopy "copy" 1 grad C=20 ckpt=host
bash $R r3-nocopy-noflagsum "copy,flagsum" 1 grad C=20 ckpt=host
bash $R r0-nocuda-base "" 0 grad C=20 ckpt=host
echo queue1-done >> ~/pm-wp078-cuda-out/status
