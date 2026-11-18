#!/usr/bin/env bash
# The plasma campaign: a0 scale from each config's vacuum run, relative to the r64x4 M4 launch.
S=/home/user/lowdisp/src
cd /home/user/runs
PY=/usr/bin/python3.12
ion() { case $1 in 16) echo 32x16;; 32) echo 16x8;; 48) echo 8x7;; 64) echo 8x4;; esac; }
scale() { $PY -c "
import sys; sys.path.insert(0,'$S'); import calib
_, r, _ = calib.calib('vac-r$1x$2-$3-$4'); _, ref, _ = calib.calib('vac-r64x4-M4-yee')
print((ref / r) ** 0.5)"; }
run() { # ppl ny solver dt seed
  local d=pl-r$1x$2-$3-$4-s$5
  [ -f $d/container.txt ] && return
  python3 $S/gen.py $d --ppl $1 --ny $2 --solver $3 --dt $4 --a0scale $(scale $1 $2 $3 $4) --ionppc $(ion $1) --seed $5 > /dev/null
  bash $S/run.sh $d
  $PY $S/analyze.py $d >> campaign.log
}
for r in "16 1" "32 2" "48 3" "64 4"; do set -- $r
  for c in "Yee yee" "M4 yee" "Lehe 0.92" "Lehe yee"; do set -- $1 $2 $c; run $1 $2 $3 $4 0; done
  if [ $1 -le 32 ]; then for s in 1 2; do run $1 $2 Yee yee $s; run $1 $2 M4 yee $s; done; fi
done
echo CAMPAIGN_DONE >> campaign.log
