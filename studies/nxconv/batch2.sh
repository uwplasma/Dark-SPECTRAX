#!/bin/bash
# nxconv batch 2: tests; grid validation (fixed Landau box); Hermite contract runs (Nx 32/64/128, Nn 64, c = 1,
# noise floor 1e-14) with Nn 128 and c 1/2, 2 spot checks at Nx 64; each command is re-invoked (resume) until done.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=.venv/bin/python
LOG=studies/nxconv/logs/batch2.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       /opt/local/bin/gtimeout 900 "$@" >> $LOG 2>&1; rc=$?
       echo "exit $rc $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; sleep 3; return $rc; }
herm() { for i in 1 2 3 4 5 6 7 8; do go $PY studies/nxconv_run.py "$@" --T 1000 --chunk 250 --noise-floor 1e-14 || return
         tail -1 $LOG | grep -q "nothing to do" && return; done; }
go $PY -m pytest -q -n 2 tests
go $PY studies/nxconv_grid_validate.py
for vq in 0.1 0.03; do
  for nx in 32 64 128; do herm --vq $vq --Nx $nx --Nn 64; done
done
for vq in 0.1 0.03; do
  herm --vq $vq --Nx 64 --Nn 128
  herm --vq $vq --Nx 64 --Nn 64 --field-nu 0.5
  herm --vq $vq --Nx 64 --Nn 64 --field-nu 2
done
