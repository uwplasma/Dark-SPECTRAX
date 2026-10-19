#!/bin/bash
# rclosure stage F (declared in studies/rclosure_gates.py): H05 v_q 0.1 Nx 64 Nn 128 with c in {16, 64, 256}; GPU 0, chunked.
cd "$(dirname "$0")/.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p studies/rclosure
for c in 16 64 256; do
  for i in $(seq 1 12); do
    out=$(nice -n 10 $PY studies/nxconv_run.py --vq 0.1 --Nx 64 --Nn 128 --field-nu $c --noise-floor 1e-14 --T 1000 --chunk 100 --tag _rcf 2>&1 | tail -n 2)
    echo "$c chunk $i: $out" >> studies/rclosure/stagef.log
    echo "$out" | grep -q "nothing to do" && break
  done
done
