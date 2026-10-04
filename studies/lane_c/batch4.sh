#!/bin/bash
# Lane C x-resolution check: spectra of the strong-drive growth, Hermite Nx 8 vs 16, grid Nx 16 vs 32.
cd "$(dirname "$0")/../.."
PY=../adapt-dark/.venv/bin/python
lock() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; }
go() { lock; PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 $PY "$@" >> studies/lane_c/logs/batch4.log 2>&1
  echo "$* exit=$?" >> studies/lane_c/logs/batch4.txt; rmdir /tmp/spectrax-heavy.lock; }
go studies/lane_c_grid_mobile.py 0.1 700 16 4096 0.02
go studies/lane_c_run.py --vq 0.1 --Nn 64 --real 0
go studies/lane_c_run.py --vq 0.1 --Nn 64 --real 0 --Nx 16
go studies/lane_c_grid_mobile.py 0.1 700 32 4096 0.02
go studies/lane_c_grid_mobile.py 0.03 1000 16 4096 0.02
echo DONE >> studies/lane_c/logs/batch4.txt
