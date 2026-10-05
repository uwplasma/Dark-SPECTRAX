#!/bin/bash
# highk batch 9 (after rebase on main): Nx = 32 closure runs with the parent noise floor 1e-14 (without it, Nx 32 takes
# 57x the Nx 16 steps by t = 40 because atol 1e-14 sits below FFT round-off), and Nx 16 with the same floor as a check
# that the floor does not change the Nx 16 result. Each run under the owned heavy lock, 900 s cap.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py "$@" >> studies/highk/logs/batch9.log 2>&1
       echo "exit $? : $*" >> studies/highk/logs/batch9.log
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
go --Nx 32 --Nn 64 --T 700 --field-nu 1 --noise-floor 1e-14
go --Nx 16 --Nn 64 --T 700 --field-nu 1 --noise-floor 1e-14
go --vq 0.03 --Nx 32 --Nn 64 --T 1000 --field-nu 1 --noise-floor 1e-14
