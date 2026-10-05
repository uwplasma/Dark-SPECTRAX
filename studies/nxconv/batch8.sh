#!/bin/bash
# nxconv batch 8: Hermite c = 1 fails earlier with Nx (t_pos 681.5 / 618 / 555.5 at Nx 32 / 64 / 128, top-mode growth),
# early negativity, worse with Nx), so run it at 0.1 / 0.03 for Nx 32/64/128 to t = 1000 and Nv 8192 at Nx 64;
# the same at 0.03, and a dt check of the masked grid.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch8.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
grid() { for i in $(seq 1 30); do go $PY studies/nxconv_grid.py "$@" --chunk 200 --frame osc --vmax 32 --xshift mask23; [ $? -eq 3 ] || return; done; }
herm() { for i in 1 2 3 4 5 6 7 8; do go $PY studies/nxconv_run.py "$@" --T 1000 --chunk 240 --noise-floor 1e-14 || return
         tail -1 $LOG | grep -q "nothing to do" && return; done; }
herm --vq 0.1 --Nx 64 --Nn 64 --field-nu 4
grid --vq 0.1 --T 1000 --Nx 64 --Nv 4096
grid --vq 0.03 --T 1000 --Nx 64 --Nv 4096
grid --vq 0.03 --T 1000 --Nx 32 --Nv 4096
grid --vq 0.1 --T 1000 --Nx 128 --Nv 4096
herm --vq 0.03 --Nx 64 --Nn 64 --field-nu 4
grid --vq 0.03 --T 1000 --Nx 128 --Nv 4096
herm --vq 0.1 --Nx 32 --Nn 64 --field-nu 4
herm --vq 0.1 --Nx 64 --Nn 64 --field-nu 8
herm --vq 0.1 --Nx 64 --Nn 64 --field-nu 2
herm --vq 0.1 --Nx 64 --Nn 128 --field-nu 4
herm --vq 0.1 --Nx 128 --Nn 64 --field-nu 4
herm --vq 0.03 --Nx 32 --Nn 64 --field-nu 4
herm --vq 0.03 --Nx 128 --Nn 64 --field-nu 4
herm --vq 0.03 --Nx 64 --Nn 64 --field-nu 8
grid --vq 0.1 --T 700 --Nx 32 --Nv 4096 --dt 0.01
