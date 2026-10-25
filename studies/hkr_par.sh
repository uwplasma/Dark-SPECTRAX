#!/bin/bash
# Lock-free lane runner for office CPU (same hkr_grid.py arguments as hkr_batch.sh main). One lane = sequential jobs.
# Usage: PY=python CORES=5-7 studies/hkr_par.sh "NX SEED T" "NX SEED T" ...   (resumes from checkpoints; rc 3 = more chunks)
cd "$(dirname "$0")/.."
PY=${PY:-python}; LOG=studies/hkr/log_par.txt; mkdir -p studies/hkr
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for job in "$@"; do
  set -- $job
  for i in $(seq 1 400); do
    echo "start $(date +%T) : $job" >> $LOG
    nice -n 10 taskset -c "${CORES:-5-7}" $PY studies/hkr_grid.py --vq 0.03 --T $3 --Nx $1 --Nv 2048 --vmax 16 --seed $2 --chunk 40 --workers ${WORKERS:-1} >> $LOG 2>&1
    rc=$?; echo "exit $rc $(date +%T) : $job" >> $LOG
    [ $rc -eq 3 ] || break
  done
done
echo "LANE_DONE $(date +%T)" >> $LOG
