#!/bin/bash
# highk: single sequential queue of the remaining jobs (each job takes and releases the owned heavy lock)
cd "$(dirname "$0")/../.."
lockrun() { until mkdir /tmp/spectrax-heavy.lock 2>/dev/null; do sleep 2; done; echo $$ > /tmp/spectrax-heavy.lock/owner
            "$@"; local rc=$?
            [ "$(cat /tmp/spectrax-heavy.lock/owner)" = "$$" ] && rm -rf /tmp/spectrax-heavy.lock; return $rc; }
PY=../adapt-dark/.venv/bin/python
lockrun bash -c 'cd ../highk && PYTHONPATH=$PWD /opt/local/bin/gtimeout 900 ../adapt/.venv/bin/python -m pytest -q -n 2 tests > ../highk-pytest.log 2>&1; echo "exit $?" >> ../highk-pytest.log'
for a in "0.1 500 16 4096 0.02 5" "0.1 500 16 4096 0.005 5"; do
  lockrun bash -c "/opt/local/bin/gtimeout 900 $PY studies/highk_grid_linear.py $a >> studies/highk/logs/batch8.log 2>&1; echo \"exit \$? : $a\" >> studies/highk/logs/batch8.log"; done
for a in "--Nx 32 --Nn 64 --T 500 --field-nu 1" "--vq 0.03 --Nx 32 --Nn 64 --T 500 --field-nu 1"; do
  lockrun bash -c "PYTHONPATH=\$PWD /opt/local/bin/gtimeout 900 $PY studies/highk_run.py $a >> studies/highk/logs/batch7.log 2>&1; echo \"exit \$? : $a\" >> studies/highk/logs/batch7.log"; done
lockrun bash -c "/opt/local/bin/gtimeout 900 $PY studies/lane_c_grid_mobile.py 0.03 1000 32 4096 0.02 >> studies/highk/logs/batch5.log 2>&1; echo \"exit \$? : 0.03 1000 32\" >> studies/highk/logs/batch5.log"
echo QUEUE_DONE >> studies/highk/logs/queue.log
