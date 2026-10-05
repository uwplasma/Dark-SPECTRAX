#!/bin/bash
# highk batch 5: mobile-ion grid references missing from lane C (Nx 8 at both drives, Nx 32 at 0.03); records go to
# studies/lane_c/ (the grid script's output directory). Each run under the owned heavy lock, 900 s cap.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
mkdir -p studies/highk/logs studies/lane_c
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       /opt/local/bin/gtimeout 900 $PY studies/lane_c_grid_mobile.py "$@" >> studies/highk/logs/batch5.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch5.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
#done go 0.1 700 8 4096 0.02
#done go 0.03 1000 8 4096 0.02
go 0.03 1000 32 4096 0.02
