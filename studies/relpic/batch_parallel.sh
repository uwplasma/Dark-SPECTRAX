#!/bin/bash
# Same 12 runs as batch.sh (identical arguments), run concurrently without the laptop lock (office CPU).
# Usage: DJIC=.. JIC=.. PY=python JOBS=5 batch_parallel.sh LOGDIR   (one core per job; skips tags with records)
export DJIC=${DJIC:?} JIC=${JIC:?} OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
here=$(cd "$(dirname "$0")" && pwd); logs=${1:?}; mkdir -p "$logs"; PY=${PY:-python}
one() {
  tag=$1; shift
  [ -f "$here/records/$tag/record.json" ] && exit 0
  nice -n 10 taskset -c "${CORES:-0-7}" $PY "$here/relpic_run.py" --tag "$tag" "$@" > "$logs/$tag.log" 2>&1
  if [ -f "$here/runs/$tag/record.json" ]; then
    mkdir -p "$here/records/$tag"; cp "$here/runs/$tag/history.npz" "$here/runs/$tag/record.json" "$here/records/$tag/"
  fi
  echo "finished $tag $(date +%H:%M)" >> "$logs/progress.txt"
}
export -f one; export here logs PY
printf '%s\n' \
 "R_s0 --T 2000 --seed 0" "N_s0 --T 2000 --seed 0 --nonrel" "R_s1 --T 2000 --seed 1" "N_s1 --T 2000 --seed 1 --nonrel" \
 "R_s2 --T 2000 --seed 2" "N_s2 --T 2000 --seed 2 --nonrel" "Z --T 1000 --seed 0 --ratio 0" "D --T 1000 --seed 0 --dt 0.01" \
 "P4 --T 1000 --seed 0 --ppc 412" "X2 --T 1000 --seed 0 --cells 2000 --dt 0.01" "Q_s0 --T 1000 --seed 0 --shape 5" \
 "Q_s1 --T 1000 --seed 1 --shape 5" | xargs -P "${JOBS:-5}" -I{} bash -c 'one {}'
echo ALLDONE >> "$logs/progress.txt"
