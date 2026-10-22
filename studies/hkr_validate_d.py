"""hkr V3: grid Landau response against the exact Volterra solution (gate GATES_V3 in hkr_gates.py)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hkr_gates as G  # noqa: E402

D = Path(__file__).resolve().parent


def volterra(kl, tmax, h):
    """n(t)/n(0) at t = i h, trapezoid; time in 1/omega_pe, k v_te = kl (lambda_De = v_te)."""
    n_t = int(round(tmax / h))
    t = np.arange(n_t + 1) * h
    g = lambda tau: tau * np.exp(-0.5 * (kl * tau) ** 2)  # noqa: E731  (omega_pe = 1, k v_te = kl, v in units of v_te)
    n = np.empty(n_t + 1)
    n[0] = 1.0
    for i in range(1, n_t + 1):
        w = np.full(i + 1, h)
        w[0] = w[-1] = h / 2
        gg = g(t[i] - t[:i + 1])
        # n_i (1 + w_i g(0)) with g(0) = 0 -> explicit
        n[i] = np.exp(-0.5 * (kl * t[i]) ** 2) - (w[:i] * gg[:i] * n[:i]).sum() - w[0] * 0
    return t, n


out = {}
for kl, tmax in G.GATES_V3["tmax"].items():
    d = np.load(D / f"hkr/grid/h_vq0_Nx8_Nv2048_v16_dt0.02_det_landau{kl:g}_fine.npz")
    t, e = d["t"], np.sqrt(d["Ek2"][:, 1])
    m = t <= tmax
    res = {}
    for h in (0.005, 0.0025):
        tv, nv = volterra(kl, tmax, h)
        ex = np.interp(t[m], tv, nv)
        res[h] = ex
    # the grid stores |E_1| = |n_1|/k; normalize both to their t = 0 value (the Poisson field of the seed)
    e0 = e[0]
    en = e[m] / e0
    sign = np.sign(res[0.0025])  # |E| has no sign: compare |exact|
    dg = float(np.max(np.abs(en - np.abs(res[0.0025]))))
    rich = float(np.max(np.abs(np.abs(res[0.005]) - np.abs(res[0.0025]))))
    out[f"V3_kl{kl:g}"] = {"max_abs_diff_over_E0": dg, "volterra_h_diff": rich, "tmax": tmax,
                           "pass": bool(dg <= G.GATES_V3["tol"])}
(D / "hkr/validation_d.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
