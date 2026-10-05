#!/bin/bash
# nxconv batch 1: tests, closure-equivalence check against the highk Nx16 record, grid validation, Hermite pilots.
# Every command holds the owned heavy lock with a 900 s cap; long runs are chunked (rerun = resume).
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch1.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1
       echo "exit $? $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
go $PY -m pytest -q -n 2 tests
go $PY studies/nxconv_run.py --vq 0.1 --Nx 16 --Nn 64 --T 300 --tag _equiv
go $PY studies/nxconv_grid_validate.py
go $PY studies/nxconv_run.py --vq 0.1 --Nx 64 --Nn 64 --T 1000 --chunk 200 --noise-floor 1e-14
