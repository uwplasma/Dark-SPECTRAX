#!/bin/bash
# nxconv batch 6: (a) the 2/3-masked grid is the usable reference (unmasked grid: aliasing-driven top-mode growth and
# early negativity, worse with Nx), so run it at 0.1 / 0.03 for Nx 32/64/128 to t = 1000 and Nv 8192 at Nx 64;
# (b) Hermite Nx 64 at 0.1 loses positivity at 618 through top-retained-mode (k = 21) growth with c = 1: try c = 4.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch6.log
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
