"""hkr V1c/V2c: validation with the gates declared in hkr_gates.py (GATES_V2C). Landau runs use --save-every 0.02."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hkr_gates as G  # noqa: E402

D = Path(__file__).resolve().parent
out = {}
a = np.load(D / "hkr/grid/h_vq0.03_Nx64_Nv4096_v32_dt0.02_det.npz")
b = np.load(D / "nxconv/grid/g_vq0.03_Nx64_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz")
n = int(np.searchsorted(a["t"], 100.0, side="right"))
diff = np.abs(a["dK_e"][:n] - b["dK_e"][:n]).max() / np.abs(b["dK_e"][:n]).max()
out["V1c"] = {"max_abs_over_max": float(diff), "pass": bool(diff <= G.GATES_V2C["V1c_max_abs_over_max"])}
for kl, (t0, t1) in G.GATES_V2C["windows"].items():
    d = np.load(D / f"hkr/grid/h_vq0_Nx8_Nv2048_v16_dt0.02_det_landau{kl:g}_fine.npz")
    t, e = d["t"], np.sqrt(d["Ek2"][:, 1])
    pk = np.nonzero((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]))[0] + 1
    pk = pk[(t[pk] > t0) & (t[pk] < t1)]
    gam = np.polyfit(t[pk], np.log(e[pk]), 1)[0]
    om = np.pi / np.mean(np.diff(t[pk]))
    w0, g0 = G.lan_root(kl)
    out[f"V2c_kl{kl:g}"] = {"gamma": float(gam), "gamma_exact": float(g0), "omega": float(om), "omega_exact": float(w0),
                            "npeaks": int(len(pk)),
                            "pass": bool(abs(gam / g0 - 1) <= G.GATES["V2_rate_rel"] and abs(om / w0 - 1) <= G.GATES["V2_freq_rel"])}
(D / "hkr/validation_c.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
