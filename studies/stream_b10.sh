#!/bin/bash
# stream B10: same-session closure-only Lawson control (record M_mi100_ord_j10_li2.json = copy of the --lawson record) and the
# streaming run (record M_mi100_ord_j10_ls.json). CPU, 2 threads, nice.
cd "$(dirname "$0")/.."
PY=${PY:-python}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
cp studies/b10/M_mi100_ord_j10_li.json studies/b10/M_mi100_ord_j10_li_expint_session.json
nice -n 10 $PY studies/b10.py run M_mi100_ord_j10 --lawson && mv studies/b10/M_mi100_ord_j10_li.json studies/b10/M_mi100_ord_j10_li2.json && mv studies/b10/M_mi100_ord_j10_li_expint_session.json studies/b10/M_mi100_ord_j10_li.json
nice -n 10 $PY studies/b10.py run M_mi100_ord_j10 --streaming
