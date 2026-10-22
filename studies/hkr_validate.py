"""hkr V1/V2 validation against the gates in hkr_gates.py. Run after: hkr_grid.py runs listed in hkr_batch.sh validate."""
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
n = min(n, len(a["t"]), len(b["t"]))
rel = np.abs(a["dK_e"][:n] - b["dK_e"][:n]) / np.maximum(np.abs(b["dK_e"][:n]), 1e-30)
out["V1"] = {"max_rel_dKe": float(rel[1:].max()), "pass": bool(rel[1:].max() <= G.GATES["V1_max_rel_dKe"]), "t_to": float(a["t"][n - 1])}
for kl, tmax in ((0.5, 40.0), (1.0, 8.0)):
    d = np.load(D / f"hkr/grid/h_vq0_Nx8_Nv2048_v16_dt0.02_det_landau{kl:g}.npz")
    t, e1 = d["t"], np.sqrt(d["Ek2"][:, 1])
    m = t <= tmax
    tt, ee = t[m], e1[m]
    pk = np.nonzero((ee[1:-1] > ee[:-2]) & (ee[1:-1] > ee[2:]))[0] + 1
    gam = np.polyfit(tt[pk], np.log(ee[pk]), 1)[0]
    om = np.pi / np.mean(np.diff(tt[pk])) if len(pk) > 1 else np.nan  # |E| peaks every half period
    w0, g0 = G.lan_root(kl)
    out[f"V2_kl{kl:g}"] = {"gamma": float(gam), "gamma_exact": float(g0), "omega": float(om), "omega_exact": float(w0),
                           "pass": bool(abs(gam / g0 - 1) <= G.GATES["V2_rate_rel"] and abs(om / w0 - 1) <= G.GATES["V2_freq_rel"]),
                           "npeaks": int(len(pk))}
(D / "hkr/validation.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
