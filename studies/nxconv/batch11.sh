#!/bin/bash
# nxconv batch 11 (gridref): wider electron v-box for the v_q 0.1 masked grid reference to t = 1000.
# dv = 64 v_te / 8192 as the Nv 8192 / vmax 32 run: vmax 48 (Nv 12288), vmax 64 (Nv 16384); coarser dv pair vmax 64
# Nv 8192; Nx pair Nx 32 at vmax 64 Nv 16384; dt 0.01 check to t = 300. Two processes at a time, 2 fft workers each.
# Usage: studies/nxconv/batch11.sh pilot | a | b   (runs the listed group in the foreground, chunked + resumable)
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
LOG=studies/nxconv/logs/batch11.log
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
grid() { for i in $(seq 1 60); do echo "start $(date +%T) : $*" >> $LOG
         nice -n 10 $PY studies/nxconv_grid.py "$@" --chunk 100 --frame osc --vshift spectral --xshift mask23 >> $LOG 2>&1
         rc=$?; echo "exit $rc $(date +%T) : $*" >> $LOG; [ $rc -eq 3 ] || return; done; }
case "$1" in
 pilot) grid --vq 0.1 --T 50 --Nx 64 --Nv 16384 --vmax 64 --dt 0.02 ;;
 a) grid --vq 0.1 --T 1000 --Nx 64 --Nv 16384 --vmax 64 --dt 0.02
    grid --vq 0.1 --T 1000 --Nx 64 --Nv 8192 --vmax 64 --dt 0.02
    grid --vq 0.1 --T 300 --Nx 64 --Nv 16384 --vmax 64 --dt 0.01 ;;
 b) grid --vq 0.1 --T 1000 --Nx 64 --Nv 12288 --vmax 48 --dt 0.02
    grid --vq 0.1 --T 1000 --Nx 32 --Nv 16384 --vmax 64 --dt 0.02 ;;
esac
