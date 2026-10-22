#!/bin/bash
# same-Nx grid references for the Hermite lane (Nx 128, 512), deterministic seed
cd "$(dirname "$0")/../.."
for job in "128 det 1000" "512 det 1000"; do
  set -- $job; T=$3 ./studies/hkr_batch.sh main $1 $2; echo "CHAIN2 done $job $(date +%T)" >> studies/hkr/log.txt
done
