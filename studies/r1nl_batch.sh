#!/bin/bash
# r1nl: run one case chunked (200 units) until done.  Usage: studies/r1nl_batch.sh ARGS...   (args of r1nl_run.py)
# Env: PY (python), CUDA_VISIBLE_DEVICES / JAX_PLATFORMS as set by the caller.
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=studies/r1nl/log.txt; mkdir -p studies/r1nl
for i in $(seq 1 12); do
  echo "start $(date +%T) $*" >> $LOG
  nice -n 10 $PY studies/r1nl_run.py "$@" --chunk 200 > /tmp/r1nl_last.$$ 2>&1; rc=$?
  tail -n 1 /tmp/r1nl_last.$$ | cut -c1-300 >> $LOG
  echo "exit $rc $(date +%T) $*" >> $LOG
  grep -q "nothing to do" /tmp/r1nl_last.$$ && break
  [ $rc -eq 0 ] || break
done
rm -f /tmp/r1nl_last.$$
