#!/bin/bash
# stiffdiag production: H05 v_q 0.1 Nx 64 Nn 128, closure c (4 or 64), noise_floor 1e-14, GPU 0 only, x64.
# 100-unit chunks up to SW, then 20-unit chunks (checkpoints near the failure).  Usage: studies/stiffdiag_batch.sh C SW
cd "$(dirname "$0")/.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
C=$1; SW=$2; LOG=studies/stiffdiag/log_c$C.txt; mkdir -p studies/stiffdiag
busy() { u=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits); [ "${u:-0}" -gt 60 ]; }
for i in $(seq 1 120); do
  while busy; do echo "gpu0 busy $(date +%T)" >> $LOG; sleep 30; done
  t=$($PY -c "
import glob,json
ps=glob.glob('studies/nxconv/runs/vq0.1_Nx64_Nn128_c${C}_nf1e-14_sd.part*.json')
print(max([json.load(open(p))['t_end'] for p in ps]) if ps else 0)")
  ch=100; [ "$(echo "$t >= $SW" | bc)" = 1 ] && ch=20
  nice -n 10 $PY studies/nxconv_run.py --vq 0.1 --Nx 64 --Nn 128 --field-nu $C --noise-floor 1e-14 --T 1000 --chunk $ch --tag _sd >> $LOG 2>&1
  rc=$?; echo "exit $rc t=$t $(date +%T)" >> $LOG
  tail -n 2 $LOG | grep -q "nothing to do" && break
  [ $rc -eq 0 ] || break
done
