#!/bin/bash
# relgrid driver (H04): runs gate C2, then the queue parts in order, each chunk under the owned heavy lock (10 ms poll: the lock is unfair to 2 s pollers).
cd "$(dirname "$0")/.." || exit 1
until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 0.01; done; echo $$ > /tmp/spectrax-heavy.lock/owner
python3 studies/relgrid_gates.py c2 > /dev/null 2>&1
rm -rf /tmp/spectrax-heavy.lock
bash studies/relgrid_batch.sh stage1c
bash studies/relgrid_batch2.sh rest
bash studies/relgrid_batch2.sh stage2
