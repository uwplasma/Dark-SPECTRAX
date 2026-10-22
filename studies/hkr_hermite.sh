#!/bin/bash
# hkr Hermite lane: v_q/v_te = 0.03, Nn 64, field-scaled closure c = 64, Lawson, noise floor 1e-14, pump frame + remaps,
# Nx in "$@". Each chunk (40 time units) runs under the owned heavy lock; rerunning resumes. SPECTRAX at the pinned
# commit must be importable (SPECTRAX_SRC=<checkout at 35231ab>).
cd "$(dirname "$0")/.."
LOG=studies/hkr/hermite_log.txt
export PYTHONPATH="${SPECTRAX_SRC:?}:$PWD" JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2"
for NX in "$@"; do
  for i in $(seq 1 100); do
    until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
    echo "start $(date +%T) Nx $NX" >> $LOG
    out=$(nice -n 10 python studies/nxconv_run.py --vq 0.03 --Nx $NX --Nn 64 --T 1000 --chunk 40 --field-nu 64 --lawson --tag _li --noise-floor 1e-14 2>&1)
    rc=$?; rm -rf /tmp/spectrax-heavy.lock; sleep 8
    echo "$out" | tail -c 700 >> $LOG
    [ $rc -ne 0 ] && break
    echo "$out" | grep -q '"status": "success"' || break
    echo "$out" | grep -q '"t_end": 1000.0' && break
  done
done
echo HERMITE_DONE >> $LOG
