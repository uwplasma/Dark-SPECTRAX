#!/bin/bash
# Lane C mobile-ion grid checks, under the shared lock (each <= 15 min).
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
one() { lock() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; }; lock
  /opt/local/bin/gtimeout 900 $PY studies/lane_c_grid_mobile.py "$@" >> studies/lane_c/logs/gridm.log 2>&1
  echo "$* exit=$?" >> studies/lane_c/logs/batch3.txt; rmdir /tmp/spectrax-heavy.lock; }
one 0.1 700 16 4096 0.02
one 0.03 1000 16 4096 0.02
one 0.1 700 16 8192 0.02
one 0.1 700 32 4096 0.02
echo DONE >> studies/lane_c/logs/batch3.txt
