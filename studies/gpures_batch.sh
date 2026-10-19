#!/bin/bash
# gpures: GPU 0 only, x64 (darkspectrax enables it), chunked + resumable; polls GPU 0 for other users before each chunk.
# Usage: studies/gpures_batch.sh c08 | run VQ NX NN
cd "$(dirname "$0")/.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
LOG=studies/gpures/log.txt; mkdir -p studies/gpures
busy() { # GPU 0 used by another process (>20% util while none of our jobs runs)
  u=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits); [ "${u:-0}" -gt 20 ]; }
wait_idle() { while busy; do echo "gpu0 busy $(date +%T)" >> $LOG; sleep 60; done; }
common="--field-nu 4 --noise-floor 1e-14"
case "$1" in
 c08) wait_idle
      JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 nice -n 10 $PY studies/nxconv_run.py --vq 0.1 --Nx 32 --Nn 64 $common --T 100 --tag _c08cpu >> $LOG 2>&1
      nice -n 10 $PY studies/nxconv_run.py --vq 0.1 --Nx 32 --Nn 64 $common --T 100 --tag _c08gpu >> $LOG 2>&1 ;;
 run) for i in $(seq 1 100); do wait_idle; echo "start $(date +%T) $*" >> $LOG
        nice -n 10 $PY studies/nxconv_run.py --vq $2 --Nx $3 --Nn $4 $common --T 1000 --chunk 100 --tag _gpu >> $LOG 2>&1
        rc=$?; echo "exit $rc $(date +%T) $*" >> $LOG
        grep -q "nothing to do" <(tail -n 3 $LOG) && break; [ $rc -eq 0 ] || break; done ;;
esac
