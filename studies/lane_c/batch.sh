#!/bin/bash
# Lane C run matrix, serialized under the shared heavy-run lock, <=15 min per run.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
one() {  # vq Nn real
  f=studies/lane_c/runs/vq$1_Nn$2_r$3.json
  [ -f "$f" ] && return
  until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 20; done
  PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY studies/lane_c_run.py --vq $1 --Nn $2 --real $3 > studies/lane_c/logs/vq$1_Nn$2_r$3.log 2>&1
  echo "$1 $2 $3 exit=$?" >> studies/lane_c/logs/batch.txt
  rmdir /tmp/spectrax-heavy.lock
}
trap "rmdir /tmp/spectrax-heavy.lock 2>/dev/null" INT TERM
mkdir -p studies/lane_c/logs
for vq in 0.001 0.002 0.005 0.01 0.02 0.03 0.05 0.1; do one $vq 64 0; one $vq 128 0; done
for vq in 0.01 0.03 0.1; do one $vq 256 0; done
for r in 1 2; do for vq in 0.01 0.03 0.1; do one $vq 64 $r; one $vq 128 $r; done; done
echo DONE >> studies/lane_c/logs/batch.txt
