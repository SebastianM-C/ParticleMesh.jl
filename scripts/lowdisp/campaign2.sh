#!/usr/bin/env bash
# Extension: a finer reference (r96x4) for Yee and M4, run after campaign.sh.
S=/home/user/lowdisp/src
cd /home/user/runs
PY=/usr/bin/python3.12
until grep -q CAMPAIGN_DONE campaign.log; do sleep 20; done
for s in Yee M4; do d=vac-r96x4-$s-yee; [ -f $d/container.txt ] || { python3 $S/gen.py $d --ppl 96 --ny 4 --solver $s --dt yee --vacuum > /dev/null; bash $S/run.sh $d; }; done
for s in Yee M4; do
  d=pl-r96x4-$s-yee-s0
  [ -f $d/container.txt ] && continue
  sc=$($PY -c "
import sys; sys.path.insert(0,'$S'); import calib
_, r, _ = calib.calib('vac-r96x4-$s-yee'); _, ref, _ = calib.calib('vac-r64x4-M4-yee'); print((ref / r) ** 0.5)")
  python3 $S/gen.py $d --ppl 96 --ny 4 --solver $s --dt yee --a0scale $sc --ionppc 7x3 --seed 0 > /dev/null
  bash $S/run.sh $d
  $PY $S/analyze.py $d >> campaign.log
done
echo CAMPAIGN2_DONE >> campaign.log
