#!/bin/bash
# nxconv batch 4: grid negativity comes from the x step, not v interpolation (batch 3: PFC-in-v and Nv 8192 change
# nothing). Grid Nx convergence (osc frame, +-32 v_te, Nv 4096, spectral) at 0.1 and 0.03 to t = 1000; x-scheme
# brackets at Nx 16 (positivity-preserving PFC in x and v; 2/3 mask as the Hermite runs); Nv 8192 spot check at Nx 64.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch4.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
grid() { for i in $(seq 1 30); do go $PY studies/nxconv_grid.py "$@" --chunk 200 --frame osc --vmax 32; [ $? -eq 3 ] || return; done; }
grid --vq 0.1 --T 1000 --Nx 32 --Nv 4096
grid --vq 0.1 --T 1000 --Nx 64 --Nv 4096
grid --vq 0.03 --T 1000 --Nx 32 --Nv 4096
grid --vq 0.03 --T 1000 --Nx 64 --Nv 4096
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vshift pfc --xshift pfc
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --xshift mask23
grid --vq 0.1 --T 1000 --Nx 128 --Nv 4096
grid --vq 0.1 --T 1000 --Nx 64 --Nv 4096 --vshift pfc --xshift pfc
grid --vq 0.1 --T 1000 --Nx 64 --Nv 8192
grid --vq 0.03 --T 1000 --Nx 128 --Nv 4096
