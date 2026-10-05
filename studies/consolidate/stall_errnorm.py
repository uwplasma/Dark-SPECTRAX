"""Probe: one Dopri8 step of the H05 pump-frame RHS from t=0; per-leaf error / (atol + rtol |y|) (Lane C defect 1)."""

import sys

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import darkspectrax as ds
from darkspectrax._model import rhs
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from lane_c_run import model, seeds  # noqa: E402

rtol, atol = 1e-10, 1e-14
for real in (-1, 0):
    m = model(0.1, 64)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(real)))
    term = diffrax.ODETerm(lambda t, y, a: rhs(t, y, m))
    s = diffrax.Dopri8()
    for dt in (1e-3, 1e-2, 1e-1):
        st = s.init(term, 0.0, dt, y, None)
        y1, err, _, _, _ = s.step(term, 0.0, dt, y, None, st, made_jump=False)
        out = {}
        for key in y:
            e = np.asarray(err[key])
            sc = atol + rtol * np.maximum(np.abs(np.asarray(y[key])), np.abs(np.asarray(y1[key])))
            r = np.abs(e) / sc
            i = np.unravel_index(np.argmax(r), r.shape)
            out[key] = (f"max {r.max():.2e} at {i} |y|={np.abs(np.asarray(y1[key]))[i]:.2e}", f"ms {np.mean(r**2):.2e}")
        tot = np.sqrt(sum(np.sum((np.abs(np.asarray(err[k])) / (atol + rtol * np.abs(np.asarray(y1[k])))) ** 2)
                          for k in y) / sum(np.asarray(y[k]).size for k in y))
        print(f"r{real} dt={dt} rms={tot:.2e}", out, flush=True)
