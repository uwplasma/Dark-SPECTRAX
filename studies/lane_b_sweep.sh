#!/bin/sh
# Lane B sweep driver: one configuration per process, each under the shared heavy lock, 15 min cap per run.
cd "$(dirname "$0")/.."
for Nn in 64 128; do for vq in 0.1 0.03 0.01; do for o in 2 3; do for nu in 0.25 0.5 1 2 4; do
  f=studies/lane_b/h05_vq${vq}_Nn${Nn}_nu${nu}_o${o}.npz
  [ -f "$f" ] && continue
  until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 20; done
  gtimeout 900 .venv/bin/python studies/lane_b_closure.py $vq $Nn $nu $o >> studies/lane_b/sweep.log 2>&1 || echo "FAIL $vq $Nn $nu $o rc=$?" >> studies/lane_b/sweep.log
  rmdir /tmp/spectrax-heavy.lock
done; done; done; done
echo DONE >> studies/lane_b/sweep.log
