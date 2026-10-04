#!/bin/bash
# Lane C controls: unseeded homogeneous runs and the independent grid check, under the shared lock.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
lock() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; }
mkdir -p studies/lane_c/logs
for vq in 0.01 0.1; do
  [ -f studies/lane_c/runs/vq${vq}_Nn64_r-1.json ] && continue
  lock; PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/lane_c_run.py --vq $vq --Nn 64 --real -1 > studies/lane_c/logs/vq${vq}_Nn64_r-1.log 2>&1
  echo "$vq 64 -1 exit=$?" >> studies/lane_c/logs/batch2.txt; rmdir /tmp/spectrax-heavy.lock
done
lock; PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/lane_c_grid.py > studies/lane_c/logs/grid.log 2>&1
echo "grid exit=$?" >> studies/lane_c/logs/batch2.txt; rmdir /tmp/spectrax-heavy.lock
echo DONE >> studies/lane_c/logs/batch2.txt
