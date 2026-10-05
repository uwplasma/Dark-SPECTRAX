#!/bin/bash
# nxconv batch 3: grid reference v-resolution / dt / interpolation study at v_q/v_te = 0.1, Nx 16, t <= 700
# (lane C's lab-frame spectral grid lost positivity at 457.5 there), then the chosen scheme at Nx 32 / 64 to t = 1000.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch3.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; return $rc; }
grid() { for i in $(seq 1 12); do go $PY studies/nxconv_grid.py "$@" --chunk 250; [ $? -eq 3 ] || return; done; }
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vmax 64 --frame lab --vshift spectral
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vmax 32 --frame osc --vshift spectral
grid --vq 0.1 --T 700 --Nx 16 --Nv 8192 --vmax 32 --frame osc --vshift spectral
grid --vq 0.1 --T 700 --Nx 16 --Nv 4096 --vmax 32 --frame osc --vshift pfc
grid --vq 0.1 --T 700 --Nx 16 --Nv 8192 --vmax 32 --frame osc --vshift pfc
grid --vq 0.1 --T 700 --Nx 16 --Nv 8192 --vmax 32 --frame osc --vshift filtered
grid --vq 0.1 --T 700 --Nx 16 --Nv 8192 --vmax 32 --frame osc --vshift pfc --dt 0.01
