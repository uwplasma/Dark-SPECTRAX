#!/bin/bash
# nxconv batch 10: Nv 8192 masked grid Nx 64 at v_q 0.1 to t = 1000 (chunked restart; supersedes the T = 700 record)
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch10.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
grid() { for i in $(seq 1 30); do go $PY studies/nxconv_grid.py "$@" --chunk 200 --frame osc --vmax 32 --xshift mask23; [ $? -eq 3 ] || return; done; }
grid --vq 0.1 --T 1000 --Nx 64 --Nv 8192 --dt 0.02
