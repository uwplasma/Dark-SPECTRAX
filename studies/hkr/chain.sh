#!/bin/bash
# order after the first noise run showed positivity loss at t = 163 (Nx 64): det first to T = 1000, noise to T = 400
cd "$(dirname "$0")/../.."
for job in "640 det 1000" "256 det 1000" "640 noise 400" "256 noise 400" "1024 det 1000" "1024 noise 400"; do
  set -- $job; T=$3 ./studies/hkr_batch.sh main $1 $2; echo "CHAIN done $job $(date +%T)" >> studies/hkr/log.txt
done
echo CHAIN_ALL_DONE >> studies/hkr/log.txt
