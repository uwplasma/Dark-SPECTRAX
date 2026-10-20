#!/bin/bash
# stream production: exact Hermite streaming + field-scaled closure in the integrating factor (SPECTRAX #83), H05 v_q 0.1 Nx 64 Nn 128,
# closure c (4 or 64), noise floor 1e-14, CPU (JAX_PLATFORMS=cpu, 2 threads, nice), x64, chunk 100 to t = 500.  TAG=_ls (default) runs
# --streaming; TAG=_li2 with LAW=--lawson is the same-session closure-only control.  Usage: studies/stream_batch.sh C
cd "$(dirname "$0")/.."
PY=${PY:-python}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2"
C=$1; TAGV=${TAG:-_ls}; MODE=${LAW---streaming}; LOG=studies/stream/log_c$C$TAGV.txt; mkdir -p studies/stream
for i in $(seq 1 20); do
  nice -n 10 $PY studies/nxconv_run.py --vq 0.1 --Nx 64 --Nn 128 --field-nu $C --noise-floor 1e-14 --T ${TMAX:-500} --chunk 100 --tag "$TAGV" $MODE >> $LOG 2>&1
  rc=$?; echo "exit $rc $(date +%T)" >> $LOG
  tail -n 2 $LOG | grep -q "nothing to do" && break
  [ $rc -eq 0 ] || break
done
