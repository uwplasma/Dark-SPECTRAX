#!/bin/bash
# highk batch 6: field-scaled AW top-mode closure (--field-nu c), LANES contract: c halved/doubled, Nn doubled,
# Nx 8/16/32, v_q/v_te = 0.1 (T 700) and 0.03 (T 1000). Each run under the owned heavy lock, 900 s cap.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
mkdir -p studies/highk/logs
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py "$@" >> studies/highk/logs/batch6.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch6.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
go --Nx 16 --Nn 64 --T 700 --field-nu 1 --tag _v2
go --Nx 16 --Nn 64 --T 700 --field-nu 0.5
go --Nx 16 --Nn 64 --T 700 --field-nu 2
go --Nx 16 --Nn 128 --T 700 --field-nu 1
go --Nx 8 --Nn 64 --T 700 --field-nu 1
go --Nx 32 --Nn 64 --T 700 --field-nu 1
go --vq 0.03 --Nx 16 --Nn 64 --T 1000 --field-nu 1
go --vq 0.03 --Nx 16 --Nn 64 --T 1000 --field-nu 0.5
go --vq 0.03 --Nx 16 --Nn 64 --T 1000 --field-nu 2
go --vq 0.03 --Nx 16 --Nn 128 --T 1000 --field-nu 1
go --vq 0.03 --Nx 8 --Nn 64 --T 1000 --field-nu 1
go --vq 0.03 --Nx 32 --Nn 64 --T 1000 --field-nu 1
