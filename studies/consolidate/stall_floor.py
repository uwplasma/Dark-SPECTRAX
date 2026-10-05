"""Fix check for the unseeded pump-frame stall: plain PID vs noise_floor, H05 inputs (lane_c_run.model).

Usage: python studies/consolidate/stall_floor.py VQ NN NX T REAL FLOOR...   (FLOOR 'none' = plain PID)
Prints status, t reached, steps, rejected, and W_ext(T) vs the exact uniform two-fluid oscillator.
"""

import json
import sys

import numpy as np

import darkspectrax as ds
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from lane_c_run import SEG, model, seeds, w_tot  # noqa: E402

vq, Nn, Nx, T, real = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
for f in sys.argv[6:]:
    m = model(vq, Nn, Nx)
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(real)))
    nf = None if f == "none" else float(f)
    o = ds.run_adaptive(m, y0, T, SEG, n_save_segment=41, rtol=1e-10, atol=1e-14, max_steps=200_000, noise_floor=nf)
    good = np.isfinite(o["t"]) & np.all(np.isfinite(o["W"]), axis=1)
    t, W = o["t"][good], o["W"][good, 2]
    eps, E0 = 1 / 1836, vq * np.sqrt(1e-3)
    A, mu = E0 * (1 + eps), 1 / (1 + eps)  # exact undamped resonant oscillator from rest (Lane C reference)
    r, rd = -(A / (2 * w_tot)) * t * np.sin(w_tot * t), -(A / (2 * w_tot)) * (np.sin(w_tot * t) + w_tot * t * np.cos(w_tot * t))
    Wl = 0.5 * mu * (rd ** 2 + w_tot ** 2 * r ** 2)
    print(json.dumps({"floor": f, "real": real, "vq": vq, "Nn": Nn, "Nx": Nx, "status": o["status"],
                      "failure_reason": o["failure_reason"], "t_reached": float(t[-1]), "steps": o["num_steps"],
                      "rejected": o["num_rejected"], "compile": round(o["compile_time"], 2),
                      "run": round(o["run_time"], 2), "W_ext_end": float(W[-1]),
                      "max_rel_W_vs_linear": float(np.max(np.abs(W - Wl)[t > 10] / Wl.max()))}), flush=True)
