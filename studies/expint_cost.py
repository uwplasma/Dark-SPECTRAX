"""expint per-step cost (ungated): plain vs Lawson Dopri8 on the same stored state, same machine, compile excluded.

Takes the checkpoint at the end of the 200-300 baseline chunk of the stiffdiag record (c = 64, Nx 64, Nn 128), integrates
T = 4 time units with the field-scaled closure c = 64 both ways (rtol 1e-10, atol 1e-14, noise floor 1e-14), twice each
(second timing reported), and prints steps, time and time per step.  Run on the same GPU as the production runs.
"""
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from lane_c_run import model  # noqa: E402

R = Path(__file__).resolve().parent / "stiffdiag" / "runs"


def main(c=64, part=2, T=4.0):
    f = R / f"vq0.1_Nx64_Nn128_c{c}_nf1e-14_sd.part{part}"
    d = np.load(f"{f}.npz")
    y = {k: jnp.asarray(d[f"y_{k}"]) for k in ("Ck", "Fk", "Dk", "W", "B")}
    y["W"], y["B"] = y["W"].astype(complex), y["B"].astype(complex)
    t0 = json.loads(Path(f"{f}.json").read_text())["t_end"]
    m = model(0.1, 128, 64)
    rhs_fn, rate_fn = ds.field_closure(m, float(c))
    _sim.rhs = rhs_fn
    out = {"c": c, "t0": t0, "T": T}
    for name, kw in (("plain", {}), ("lawson", {"integrating_factor": rate_fn})):
        cache = {}
        for rep in range(2):
            tic = time.perf_counter()
            o = ds.run(m, y, T, n_save=9, t0=t0, rtol=1e-10, atol=1e-14, noise_floor=1e-14, cache=cache, max_steps=400_000, **kw)
            wall = time.perf_counter() - tic
        out[name] = {"status": o["status"], "steps": o["num_steps"], "run_time": o["run_time"], "wall": wall,
                     "time_per_step": o["run_time"] / o["num_steps"]}
    out["cost_per_step_ratio"] = out["lawson"]["time_per_step"] / out["plain"]["time_per_step"]
    out["steps_ratio"] = out["plain"]["steps"] / out["lawson"]["steps"]
    Path(__file__).resolve().parent.joinpath("expint").mkdir(exist_ok=True)
    (Path(__file__).resolve().parent / "expint" / "cost.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
