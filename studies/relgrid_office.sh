#!/bin/bash
# relgrid (H04, agent relgrid2) office-CPU runner: same run set as relgrid_driver.sh (C2, stage1c, rest, stage2), no local heavy lock
# (office is shared: <= 4 worker threads per lane, 2 lanes, nice 10). Chunked at 300 omega_pe^-1; rerun resumes from checkpoints.
# Run: bash studies/relgrid_office.sh lane1|lane2
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
chunked() {
  while true; do
    nice -n 10 $PY studies/relgrid.py "$@" --chunk 300 --workers 4
    rc=$?
    [ $rc -ne 3 ] && break
  done
}
B="--Nx 32 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23"
case $1 in
lane1)
  OMP_NUM_THREADS=4 nice -n 10 $PY studies/relgrid_gates.py c2 > /dev/null 2>&1
  chunked --vq 0.03 --T 1000 --Nx 64 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23 --noise shot --seed 1 --npc 205000
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 205000 --nonrel
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 20500
  chunked --vq 0.03 --T 1000 $B --noise shot --seed 1 --npc 20500 --nonrel
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 4096 --dt 0.01 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 64 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 8192 --dt 0.02 --frame osc --umax 32 --xshift mask23
  chunked --vq 0.03 --T 1000 --Nx 32 --Nu 6144 --dt 0.02 --frame osc --umax 48 --xshift mask23
  ;;
lane2)
  chunked --vq 0.03 --T 1000 --Nx 128 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23 --noise shot --seed 1 --npc 205000
  chunked --vq 0.001 --T 3000 $B
  chunked --vq 0.001 --T 3000 $B --nonrel
  chunked --vq 0.001 --T 3000 $B --noise shot --seed 1 --npc 184500000
  chunked --vq 0.001 --T 3000 $B --noise shot --seed 1 --npc 184500000 --nonrel
  ;;
esac
