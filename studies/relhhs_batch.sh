#!/bin/bash
# relhhs production: one configuration "Nx Nn c" per call, chunks of 100 omega_pe^-1 until T = 1000 or a failed/short part (rerunning resumes).
# SPECTRAX at the pinned commit must be importable (SPECTRAX_SRC=<checkout>). Usage: [T=1000] [EXTRA="--shot-npc 103 --seed 0"] relhhs_batch.sh NX NN C
cd "$(dirname "$0")/.."
export PYTHONPATH="${SPECTRAX_SRC:?}:$PWD" JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
NX=$1; NN=$2; C=$3; TEND=${T:-1000}; TAG=$(echo "$EXTRA" | tr -d " -")
mkdir -p studies/relhhs
for i in $(seq 1 60); do
  out=$(nice -n 10 ${PYTHON:-python} studies/relhhs_run.py --Nx $NX --Nn $NN --field-nu $C --T $TEND --chunk 100 $EXTRA 2>&1)
  echo "$out" | tail -c 600 >> studies/relhhs/log_Nx${NX}_Nn${NN}_c${C}${TAG}.txt
  echo "$out" | grep -q '"status": "success"' || break
  echo "$out" | grep -q "\"t_end\": ${TEND}.0" && break
done
echo "DONE Nx$NX Nn$NN c$C" >> studies/relhhs/log_Nx${NX}_Nn${NN}_c${C}${TAG}.txt
