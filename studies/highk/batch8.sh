#!/bin/bash
# highk batch 8: linear test of the grid reference (k5 seed only), dt 0.02 and 0.005
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       /opt/local/bin/gtimeout 900 $PY studies/highk_grid_linear.py "$@" >> studies/highk/logs/batch8.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch8.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
go 0.1 500 16 4096 0.02 5
go 0.1 500 16 4096 0.005 5
