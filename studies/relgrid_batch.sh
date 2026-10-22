#!/bin/bash
# relgrid production queue (H04). Each chunk (<= 300 omega_pe^-1, a few minutes) takes the owned heavy lock, so other agents interleave.
# Run: bash studies/relgrid_batch.sh stage1|stage2|stage3
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
chunked() {  # chunked <relgrid.py args...>
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
stage1)  # decisive: matched drive 0.03, relativistic vs nonrelativistic, deterministic and HHS-like noise; weak drive relativistic
  chunked --vq 0.03 --T 1000 $B
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 205
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 205 --nonrel
  chunked --vq 0.001 --T 2000 $B
  ;;
stage2)  # refinement pairs (gate E) at v_q = 0.03, T = 1000
  chunked --vq 0.03 --T 1000 --Nx 64 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 8192 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 4096 --dt 0.01 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 6144 --dt 0.02 --frame osc --umax 48 --xshift mask23
  ;;
stage3)  # weak drive to HHS timescales
  chunked --vq 0.001 --T 5000 $B
  chunked --vq 0.001 --T 5000 $B --noise shot --seed 1 --npc 184500
  chunked --vq 0.001 --T 5000 $B --nonrel
  chunked --vq 0.001 --T 5000 $B --noise shot --seed 1 --npc 184500 --nonrel
  ;;
esac
