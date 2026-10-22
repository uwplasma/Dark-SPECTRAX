"""hkr V1b/V2b: follow-up validation gates (declared in hkr_gates.py after the first evaluation)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hkr_gates as G  # noqa: E402

D = Path(__file__).resolve().parent
out = {}
a = np.load(D / "hkr/grid/h_vq0.03_Nx64_Nv4096_v32_dt0.02_det.npz")
b = np.load(D / "nxconv/grid/g_vq0.03_Nx64_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz")
n = int(np.searchsorted(a["t"], 100.0, side="right"))
diff = np.abs(a["dK_e"][:n] - b["dK_e"][:n]).max() / np.abs(b["dK_e"][:n]).max()
out["V1b"] = {"max_abs_over_max": float(diff), "pass": bool(diff <= G.GATES_V1B["V1b_max_abs_over_max"])}
for kl, tmax in ((0.5, 40.0), (1.0, 8.0)):
    d = np.load(D / f"hkr/grid/h_vq0_Nx8_Nv2048_v16_dt0.02_det_landau{kl:g}.npz")
    m = (d["t"] >= 1.0) & (d["t"] <= tmax)
    t, e = d["t"][m], np.sqrt(d["Ek2"][m, 1])
    w0, g0 = G.lan_root(kl)

    def res(p):
        return (p[0] * np.exp(p[1] * t) * np.abs(np.cos(p[2] * t + p[3])) - e) / e.max()
    best = min((least_squares(res, [e[0] * 1.5, g0 * 0.9, w0 * 1.02, p0]) for p0 in np.linspace(-1.5, 1.5, 7)),
               key=lambda s: s.cost)
    g, w = best.x[1], best.x[2]
    out[f"V2b_kl{kl:g}"] = {"gamma": float(g), "gamma_exact": float(g0), "omega": float(w), "omega_exact": float(w0),
                            "rms_resid": float(np.sqrt(2 * best.cost / len(t))),
                            "pass": bool(abs(g / g0 - 1) <= G.GATES["V2_rate_rel"] and abs(w / w0 - 1) <= G.GATES["V2_freq_rel"])}
(D / "hkr/validation_b.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
