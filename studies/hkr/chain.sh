#!/bin/bash
cd "$(dirname "$0")/../.."
for job in "64 noise" "640 noise" "64 det" "640 det" "256 noise" "256 det" "1024 noise" "1024 det"; do
  set -- $job; ./studies/hkr_batch.sh main $1 $2; echo "CHAIN done $job $(date +%T)" >> studies/hkr/log.txt
done
echo CHAIN_ALL_DONE >> studies/hkr/log.txt
