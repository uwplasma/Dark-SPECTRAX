"""nxconv: validation of studies/nxconv_grid.py on its own (before it is used as a reference).

1. Linear Landau root: no drive, electrons + mobile ions (m_i/m_e = 1836), k lambda_D in {0.3, 0.5}, amplitude 1e-4.
   Fit omega and gamma of E_k1(t) (zero crossings + log-envelope over 5 < t < 30) for each v-shift scheme and frame,
   against the exact two-species root (studies/refs/code/disp.py, scipy wofz).
2. Conservation on the H05 setup at v_q/v_te = 0.1 (short window t <= 100): mass, momentum, energy defect
   (K + U - W_ext) relative to W_ext, lab vs oscillating frame.
3. Unseeded H05 control at 0.1 (t <= 200): W_ext against the exact uniform two-fluid law (cycle average
   (1 + eps) E0^2 (t^2 + w^-2)/8 and the instantaneous mu/2 (r'^2 + w^2 r^2)).
Run: python studies/nxconv_grid_validate.py -> studies/nxconv/grid_validation.json
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / "refs" / "code"))
sys.path.insert(0, str(D))
import nxconv_grid as g  # noqa: E402
from disp import root  # noqa: E402

G = D / "nxconv" / "grid"
res = {}


def load(tag):
    return np.load(G / f"{tag}.npz"), json.loads((G / f"{tag}.json").read_text())


vte, mi = np.sqrt(1e-3), 1836.0
vti = vte / np.sqrt(mi)
for kld in (0.3, 0.5):
    k = kld / vte
    exact = root(k, [(1.0, vte, 0.0), (1 / mi, vti, 0.0)], np.sqrt(1 + 3 * kld ** 2) - 0.05j)
    for vs, fr in (("spectral", "lab"), ("filtered", "lab"), ("pfc", "lab"), ("pfc", "osc")):
        args = ["--vq", "0", "--T", "30", "--Nx", "8", "--Nv", "512", "--Nvi", "256", "--dt", "0.02",
                "--vmax", "8", "--frame", fr, "--vshift", vs, "--landau", str(kld), "--save-every", "0.02"]
        g.main(args)
        tag = f"g_vq0_Nx8_Nv512_dt0.02_{fr}_v8_{vs}_landau{kld:g}"
        d, _ = load(tag)
        t, E = d["t"], np.sqrt(d["Ek2"][:, 1])
        sel = (t > 3) & (t < 25)  # gamma fit window
        # |E_k1| ~ |cos(w t + p)| e^{gamma t}: fit gamma on local maxima, omega from the peak spacing (half period)
        tt, ee = t[sel], E[sel]
        pk = np.nonzero((ee[1:-1] > ee[:-2]) & (ee[1:-1] > ee[2:]))[0] + 1
        gam = np.polyfit(tt[pk], np.log(ee[pk]), 1)[0]
        om = np.pi / np.mean(np.diff(tt[pk]))
        res[f"landau_k{kld:g}_{vs}_{fr}"] = {"omega": om, "gamma": gam, "exact_omega": exact.real,
                                            "exact_gamma": exact.imag, "rel_err_gamma": abs(gam / exact.imag - 1),
                                            "note": "peaks of |E_k1| sampled every 0.02"}
for fr in ("lab", "osc"):
    for vs in ("spectral", "pfc"):
        g.main(["--vq", "0.1", "--T", "100", "--Nx", "16", "--Nv", "4096", "--dt", "0.02", "--vmax",
                "64" if fr == "lab" else "24", "--frame", fr, "--vshift", vs])
        tag = f"g_vq0.1_Nx16_Nv4096_dt0.02_{fr}_v{64 if fr == 'lab' else 24}_{vs}"
        d, j = load(tag)
        res[f"conservation_vq0.1_T100_{fr}_{vs}"] = {k: j[k] for k in (
            "max_energy_defect_over_W", "max_mass_drift", "max_momentum", "min_f_rel", "wall_time")}
eps = 1 / mi
for dt in ("0.02", "0.01"):
    g.main(["--vq", "0.1", "--T", "200", "--Nx", "8", "--Nv", "2048", "--dt", dt, "--vmax", "16",
          "--frame", "osc", "--vshift", "pfc", "--unseeded"])
    d, j = load(f"g_vq0.1_Nx8_Nv2048_dt{dt}_osc_v16_pfc_unseeded")
    w = np.sqrt(1 + eps)
    A, mu, t = 0.1 * vte * (1 + eps), 1 / (1 + eps), d["t"]
    r = -(A / (2 * w)) * t * np.sin(w * t)
    rd = -(A / (2 * w)) * (np.sin(w * t) + w * t * np.cos(w * t))
    Wl = mu / 2 * (rd ** 2 + w ** 2 * r ** 2)
    sel = t > 20
    res[f"unseeded_vq0.1_T200_osc_pfc_dt{dt}"] = {
        "max_rel_dev_W_vs_exact_two_fluid": float(np.max(np.abs(d["W_ext"][sel] / Wl[sel] - 1))),
        "max_energy_defect_over_W": j["max_energy_defect_over_W"]}
(D / "nxconv" / "grid_validation.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
print(json.dumps(res, indent=1, default=float))
