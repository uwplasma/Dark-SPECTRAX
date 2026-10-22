#!/bin/bash
# relpic production batch (runs declared in gates.py). One run per lock acquisition.
# Usage: DJIC=<Dark-JAX-in-Cell> JIC=<JAX-in-Cell at 2d693cb> batch.sh LOGDIR
export DJIC=${DJIC:?} JIC=${JIC:?}
here=$(cd "$(dirname "$0")" && pwd)
logs=${1:?}; mkdir -p "$logs"
run() {  # TAG then relpic_run.py arguments
  tag=$1; shift
  [ -f "$here/records/$tag/record.json" ] && return
  until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 0.005; done
  echo $$ > /tmp/spectrax-heavy.lock/owner
  nice -n 5 python "$here/relpic_run.py" --tag "$tag" "$@" > "$logs/$tag.log" 2>&1
  rm -rf /tmp/spectrax-heavy.lock
  if [ -f "$here/runs/$tag/record.json" ]; then
    mkdir -p "$here/records/$tag"
    cp "$here/runs/$tag/history.npz" "$here/runs/$tag/record.json" "$here/records/$tag/"
  fi
  echo "finished $tag $(date +%H:%M)" >> "$logs/progress.txt"
}
run R_s0 --T 2000 --seed 0
run N_s0 --T 2000 --seed 0 --nonrel
run Z    --T 1000 --seed 0 --ratio 0
run D    --T 1000 --seed 0 --dt 0.01
run P4   --T 1000 --seed 0 --ppc 412
run X2   --T 1000 --seed 0 --cells 2000 --dt 0.01
run Q_s0 --T 1000 --seed 0 --shape 5
run R_s1 --T 2000 --seed 1
run N_s1 --T 2000 --seed 1 --nonrel
run R_s2 --T 2000 --seed 2
run N_s2 --T 2000 --seed 2 --nonrel
run Q_s1 --T 1000 --seed 1 --shape 5
echo ALLDONE >> "$logs/progress.txt"
