#!/bin/bash
# SW Hermite H05 prototype runs (v_q 0.1, nu = 0, no closure, to t = 1000). One thread each, nice 10.
# Usage: studies/sw/batch.sh
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for args in "--Nx 16 --Nn 64" "--Nx 16 --Nn 128" "--Nx 32 --Nn 64" "--Nx 32 --Nn 128" "--Nx 32 --Nn 256"; do
  nice -n 10 $PY studies/sw/sw_run.py --vq 0.1 --T 1000 --dt 0.02 $args
done
