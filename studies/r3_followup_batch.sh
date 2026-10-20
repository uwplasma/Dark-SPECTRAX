#!/bin/bash
# R3 follow-up F (r3_followup_gates.py). usage: bash studies/r3_followup_batch.sh <python>
PY=${1:-python}
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
run() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
        $PY studies/r3_kin.py "$@"; rm -rf /tmp/spectrax-heavy.lock; }
for nx in 1024 1536 2048 3072; do
  run 0.3 4 none x --Nx $nx
  run 0.3 4 45 y --Nx $nx
done
for v in 0.005 0.02; do
  run 0.3 4 none x --Nx 3072 --vte $v
  run 0.3 4 45 y --Nx 3072 --vte $v
done
