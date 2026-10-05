#!/bin/bash
# nxconv batch 7: trimmed remainder of batch 5 (the unmasked grid is aliasing-unstable, so its 0.03 runs and Nx 64 PFC are
# dropped; the masked grid runs are in batch 6). Resumes the Nv 8192 masked Nx 64 run from its checkpoint.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch7.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
grid() { for i in $(seq 1 30); do go $PY studies/nxconv_grid.py "$@" --chunk 200 --frame osc --vmax 32; [ $? -eq 3 ] || return; done; }
grid --vq 0.1 --T 700 --Nx 64 --Nv 8192 --xshift mask23
grid --vq 0.1 --T 1000 --Nx 32 --Nv 4096 --xshift mask23
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vshift pfc --xshift pfc
