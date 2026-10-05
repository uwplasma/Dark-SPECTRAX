"""Before/after for the run_adaptive fixes: rerun a recorded Lane C case with the current code and compare.

"Before" is the committed record studies/lane_c/runs/<case>.json/.npz (per-segment recompilation, no positivity
stop, plain PID). "After" is the current run_adaptive (one compilation, positivity stop) with optional noise_floor.
Usage: python studies/consolidate/before_after.py VQ NN REAL [FLOOR]  -> studies/consolidate/before_after_<case>.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lane_c_run import NSAVE, SEG, diagnostics, model, seeds  # noqa: E402

vq, Nn, real = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
nf = float(sys.argv[4]) if len(sys.argv) > 4 else None
case = f"vq{vq:g}_Nn{Nn}_r{real}"
old = json.loads((HERE.parent / "lane_c" / "runs" / f"{case}.json").read_text())
oz = np.load(HERE.parent / "lane_c" / "runs" / f"{case}.npz")
m = model(vq, Nn)
y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(real)))
tic = time.perf_counter()
o = ds.run_adaptive(m, y0, 1000.0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000, noise_floor=nf)
wall = time.perf_counter() - tic
good = np.isfinite(o["t"]) & np.all(np.isfinite(o["W"]), axis=1)
for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
    o[k] = o[k][good]
K = diagnostics(m, o)[0]
n = min(o["t"].size, oz["t"].size)
assert np.allclose(o["t"][:n], oz["t"][:n])
dKn, dKo = K[:n, 0] - K[0, 0], oz["K"][:n, 0] - oz["K"][0, 0]
win = o["t"][:n] > 50
run_max = np.maximum.accumulate(np.abs(dKo))
rec = {"case": case, "noise_floor": nf,
       "before": {k: old[k] for k in ("status", "failure_reason", "t_reached", "steps", "rejected", "compile_time",
                                      "run_time", "wall_time", "repository_commit")},
       "after": {"status": o["status"], "failure_reason": o["failure_reason"], "t_reached": o["t_reached"],
                 "steps": o["num_steps"], "rejected": o["num_rejected"], "compile_time": o["compile_time"],
                 "run_time": o["run_time"], "wall_time": wall, "repository_commit": ds._simulation._git_sha()},
       "common_saves": int(n), "max_rel_diff_dK_e": float(np.max(np.abs(dKn - dKo)[win] / run_max[win])),
       "max_rel_diff_W_ext": float(np.max(np.abs(o["W"][:n, 2] - oz["W"][:n, 2]) / np.abs(oz["W"][:n, 2]).max()))}
(HERE / f"before_after_{case}{'' if nf is None else f'_floor{nf:g}'}.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec), flush=True)
