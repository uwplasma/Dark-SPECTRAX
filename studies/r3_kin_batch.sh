#!/bin/bash
# R3 kinetic production batch (declared in r3_gates.py). Sequential, under the shared heavy-run lock.
# usage: bash studies/r3_kin_batch.sh <python> ; logs to stdout
PY=${1:-python}
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
run() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
        $PY studies/r3_kin.py "$@"; rm -rf /tmp/spectrax-heavy.lock; }
CASES="0.3,2,none,y 0.3,2,none,x 0.3,4,none,y 0.3,4,none,x 0.3,8,none,y 0.3,8,none,x 0.15,4,none,y 0.15,8,none,y 0.15,12,none,y
0.3,4,0,y 0.3,4,45,x 0.3,4,45,y 0.3,4,45,z 0.3,4,90,y 0.3,4,90,z"
run 0.3 4 none y --nopacket                       # K1
for c in $CASES; do IFS=, read eta w th pol <<< "$c"
  run $eta $w $th $pol                              # primary member (Nx 512, Nn 4)
  run $eta $w $th $pol --Nx 768 --Nn 6              # refinement member
done
run 0.3 4 none y --amp 3                            # K6
