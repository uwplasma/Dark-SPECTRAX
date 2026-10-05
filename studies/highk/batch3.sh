#!/bin/bash
# highk batch 1: diagnostic Nx=16 runs at v_q/v_te = 0.1 (each under the owned heavy lock, 900 s cap)
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
mkdir -p studies/highk/logs
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py "$@" >> studies/highk/logs/batch3.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch3.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }




go --Nx 8 --Nn 64 --T 480 --no-remap
go --Nx 12 --Nn 64 --T 480 --no-remap
go --Nx 16 --Nn 32 --T 480 --no-remap
