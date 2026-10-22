#!/bin/bash
# relgrid production queue, part 2 (H04); same chunking/lock as relgrid_batch.sh. Run: bash studies/relgrid_batch2.sh rest|stage2
# (a separate file because relgrid_batch.sh was already executing when these stages were declared.)
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
chunked() {
  while true; do
    until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done
    echo $$ > /tmp/spectrax-heavy.lock/owner
    nice -n 10 $PY studies/relgrid.py "$@" --chunk 300 --workers 2
    rc=$?
    rm -rf /tmp/spectrax-heavy.lock
    [ $rc -ne 3 ] && break
  done
}
B="--Nx 32 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23"
case $1 in
rest)  # remainder of stage1b (nonrelativistic control at npc 2.05e5 resumes from its checkpoint), then the weak drive and the npc 2.05e4 pair
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 205000 --nonrel
  chunked --vq 0.001 --T 3000 $B
  chunked --vq 0.001 --T 3000 $B --nonrel
  chunked --vq 0.001 --T 3000 $B --noise shot --seed 1 --npc 184500000
  chunked --vq 0.001 --T 3000 $B --noise shot --seed 1 --npc 184500000 --nonrel
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 20500
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 20500 --nonrel
  ;;
stage2)  # refinement pairs (gate E, D2) at v_q = 0.03, relativistic det, T = 1000
  chunked --vq 0.03 --T 1000 --Nx 64 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 4096 --dt 0.01 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 8192 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 6144 --dt 0.02 --frame osc --umax 48 --xshift mask23
  ;;
esac
