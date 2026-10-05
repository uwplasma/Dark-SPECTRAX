#!/bin/bash
# highk batch 1: diagnostic Nx=16 runs at v_q/v_te = 0.1 (each under the owned heavy lock, 900 s cap)
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
mkdir -p studies/highk/logs
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py "$@" >> studies/highk/logs/batch2.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch2.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }




go --Nx 16 --Nn 64 --T 300 --no-remap --seed-scale 0.01
go --vq 0 --Nx 16 --Nn 64 --T 300 --no-remap
go --vq 0.01 --Nx 16 --Nn 64 --T 300 --no-remap
