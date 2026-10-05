#!/bin/bash
# highk batch 7: Nx = 32 closure runs on a window that fits the 900 s cap (the T = 700/1000 runs hit the cap).
# Nx 8/16/32, v_q/v_te = 0.1 (T 700) and 0.03 (T 1000). Each run under the owned heavy lock, 900 s cap.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
mkdir -p studies/highk/logs
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py "$@" >> studies/highk/logs/batch7.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch7.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
go --Nx 32 --Nn 64 --T 500 --field-nu 1
go --vq 0.03 --Nx 32 --Nn 64 --T 500 --field-nu 1
