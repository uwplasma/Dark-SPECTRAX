"""Probe: why unseeded H05 pump-frame runs exhaust the step budget (Lane C defect 1).

Runs 20-omega_pe^-1 segments of the unseeded (r-1) and seeded (r0) H05 case at v_q/v_te = 0.1, Nn = 64, and reports
steps per segment plus the magnitude of the state components that the PID error norm sees relative to atol.
Usage: python studies/consolidate/stall_probe.py [max_steps] [n_segments]
"""

import sys

import jax.numpy as jnp
import numpy as np

import darkspectrax as ds
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from lane_c_run import model, seeds  # noqa: E402

max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
nseg = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for real in (-1, 0):
    m = model(0.1, 64)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(real)))
    t = 0.0
    for k in range(nseg):
        o = ds.run(m, y, 20.0, n_save=3, t0=t, rtol=1e-10, atol=1e-14, max_steps=max_steps)
        Ck = o["Ck"][-1]
        H = m.Nn
        mag = {}
        for s, name in enumerate("ei"):
            c = Ck[s * H:(s + 1) * H]
            mag[name] = dict(C000=abs(c[0, 0, 0, 0]), k0_n_gt0=np.abs(c[1:, 0, 0, 0]).max(),
                             kne0=np.abs(c[:, :, 1:, :]).max())
        print(f"r{real} seg{k} t0={t} status={o['status']} steps={o['num_steps']} rej={o['num_rejected']} "
              f"W={np.abs(o['W'][-1]).max():.3e} B={np.abs(o['B'][-1]).max():.3e} Fk={np.abs(o['Fk'][-1]).max():.3e}",
              {n: {kk: f"{v:.2e}" for kk, v in d.items()} for n, d in mag.items()}, flush=True)
        if o["status"] != "success":
            break
        t += 20.0
        y = {"Ck": jnp.asarray(Ck), "Fk": jnp.asarray(o["Fk"][-1]), "Dk": jnp.asarray(o["Dk"][-1]),
             "W": jnp.asarray(o["W"][-1], complex), "B": jnp.asarray(o["B"][-1], complex)}
