#!/bin/bash
# hkr batch: every chunk (<= ~15 min) runs under the owned heavy lock; rerunning resumes from the checkpoint.
# Usage: studies/hkr_batch.sh validate | main NX SEED [NV VMAX] | list
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=studies/hkr/log.txt
mkdir -p studies/hkr
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
grid() { local chunk=$1; shift
  for i in $(seq 1 200); do
    until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
    echo "start $(date +%T) : $*" >> $LOG
    nice -n 10 $PY studies/hkr_grid.py "$@" --chunk $chunk --workers ${WORKERS:-2} >> $LOG 2>&1
    rc=$?; rm -rf /tmp/spectrax-heavy.lock; echo "exit $rc $(date +%T) : $*" >> $LOG
    [ $rc -eq 3 ] || return $rc
  done; }
case "$1" in
 validate)
  grid 1e9 --vq 0.03 --T 100 --Nx 64 --Nv 4096 --vmax 32 --seed det
  grid 1e9 --vq 0 --T 40 --Nx 8 --Nv 2048 --vmax 16 --landau 0.5 --seed det
  grid 1e9 --vq 0 --T 8 --Nx 8 --Nv 2048 --vmax 16 --landau 1.0 --seed det ;;
 main) grid ${CHUNK:-40} --vq 0.03 --T ${T:-1000} --Nx $2 --Nv ${4:-2048} --vmax ${5:-16} --seed $3 ;;
esac
