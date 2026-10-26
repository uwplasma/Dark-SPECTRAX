#!/bin/bash
# picconv launcher: one tag of conv_gates.LADDER on given cores.  Usage: DJIC=.. JIC=.. PY=.. batch_conv.sh TAG CORES LOGDIR
export JAX_PLATFORMS=cpu
here=$(cd "$(dirname "$0")" && pwd); tag=$1; cores=$2; logs=${3:?}; mkdir -p "$logs"
args=$($PY -c "import sys;sys.path.insert(0,'$here');import conv_gates as G;print(G.LADDER['$tag'][0])")
n=$(echo "$cores" | tr ',' '\n' | awk -F- '{if(NF==2)s+=$2-$1+1;else s+=1}END{print s}')
[ -f "$here/records/$tag/record.json" ] && exit 0
OMP_NUM_THREADS=$n nice -n 10 taskset -c "$cores" $PY "$here/relpic_run.py" --tag "$tag" $args > "$logs/$tag.log" 2>&1
if [ -f "$here/runs/$tag/record.json" ]; then
  mkdir -p "$here/records/$tag"; cp "$here/runs/$tag/history.npz" "$here/runs/$tag/record.json" "$here/records/$tag/"
fi
echo "finished $tag $(date +%H:%M)" >> "$logs/progress.txt"
