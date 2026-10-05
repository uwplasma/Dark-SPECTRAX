#!/bin/bash
# nxconv batch 5 (replaces the rest of batch 4, reordered): batch 4 showed the lab-x-spectral grid at Nx 64 going
# negative EARLIER (506.5) than Nx 32 (541.5), with |E_k5|^2 jumping 1e5x over 500-550: test grid aliasing (2/3 mask,
# as the Hermite runs) and v resolution at Nx 64 first, then the remaining batch 4 items. Resumable (checkpoints).
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch5.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
grid() { for i in $(seq 1 30); do go $PY studies/nxconv_grid.py "$@" --chunk 200 --frame osc --vmax 32; [ $? -eq 3 ] || return; done; }
grid --vq 0.1 --T 700 --Nx 64 --Nv 4096 --xshift mask23
grid --vq 0.1 --T 700 --Nx 64 --Nv 8192
grid --vq 0.1 --T 700 --Nx 64 --Nv 8192 --xshift mask23
grid --vq 0.1 --T 1000 --Nx 32 --Nv 4096 --xshift mask23
grid --vq 0.03 --T 1000 --Nx 32 --Nv 4096
grid --vq 0.03 --T 1000 --Nx 64 --Nv 4096
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vshift pfc --xshift pfc
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --xshift mask23
grid --vq 0.1 --T 700 --Nx 64 --Nv 4096 --vshift pfc --xshift pfc
