#!/bin/bash
# H08 batch: controls, collisionless baselines, collisional runs and refinement pairs (rerun = resume, chunks of 200).
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
PY=.venv/bin/python
LOG=studies/h08/batch.log
go() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
       echo "start $(date +%T) : $*" >> $LOG
       for i in 1 2 3 4 5 6; do nice -n 10 $PY studies/h08.py run "$@" >> $LOG 2>&1; done
       echo "exit $? $(date +%T) : $*" >> $LOG
       [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; }
for vq in 0.01 0.03; do
  for nu in 1e-3 1e-2; do go --vq $vq --nu $nu --Nx 8 --control; done
  for nu in 0 1e-3 1e-2; do go --vq $vq --nu $nu --Nx 32; go --vq $vq --nu $nu --Nx 64; done
  for nu in 1e-3 1e-2; do go --vq $vq --nu $nu --Nx 32 --Nn 128; done
done
$PY studies/h08.py analyze >> $LOG 2>&1
