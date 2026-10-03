"""Ordinary-plasma cross-check against the independent continuum references in studies/refs.

Reruns the Hermite side of B00 (linear Landau) and B02 (warm two-stream) through darkspectrax
in mode="ordinary" on the pinned parent, with the same electrostatic inputs and fit rules as the
reference records, and compares with the semi-Lagrangian grid solver and the wofz roots stored there.
Electrostatic units (omega_pe = 1, v_te = 1) map to parent units with v_t/c = beta:
a = sqrt(2) beta v_t, u -> beta u, L -> beta L, E_es = E_code / beta (see studies/refs/code/sx.py).

Run: python studies/compare_refs.py   -> studies/refs_rerun/{run.json, run.npz}
"""

import json
import sys
from pathlib import Path

import numpy as np

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "refs" / "code"))
from fit import fit_growth, fit_pair  # noqa: E402  (reference fit code, unchanged)

beta = 0.1


def run_es(species, L_es, Nx, Nn, tmax, nt, seeds):
    a = [np.sqrt(2) * beta * vt for _, vt, _ in species]
    model = ds.Model(Nx=Nx, Nn=Nn, Lx=L_es * beta, qs=(-1.0,) * len(species),
                     Omega_cs=(1.0,) * len(species), alpha_s=tuple(np.repeat(a, 3)),
                     u_s=tuple(v for _, _, u in species for v in (beta * u, 0.0, 0.0)),
                     rho_background=1.0)
    pert = [(s, (m, 0, 0), fr * dn) for s, (fr, _, _) in enumerate(species) for m, dn in seeds.items()]
    y = ds.consistent_fields(model, ds.maxwellian(model, [fr for fr, _, _ in species], pert))
    out = ds.run(model, y, tmax, n_save=nt, rtol=1e-10, atol=1e-16)
    return out, out["Fk"][:, 0, 0, :, 0] / beta


rec = {"case_id": "refs_rerun", "parent_commit": ds.PARENT_COMMIT,
       "repository_commit": ds._simulation._git_sha(),
       "command": "python studies/compare_refs.py", "beta": beta, "B00": {}, "B02": {}}
arrays = {}
b00 = json.loads((HERE / "refs/B00/run.json").read_text())["results"]
for key, k, tmax in (("k0.5", 0.5, 45.0), ("k0.3", 0.3, 160.0)):
    ref = b00[key]
    for Nn in (128, 256):
        win = tuple(ref["spectrax"][f"N{Nn}"]["fit_window"])
        out, E = run_es([(1.0, 1.0, 0.0)], 2 * np.pi / k, 4, Nn, tmax, int(round(tmax / 0.1)) + 1, {1: 5e-5})
        w, g, res = fit_pair(out["t"], E[:, 1], *ref["root_wofz"], win)
        gb = ref["grid_best"]
        rec["B00"][f"{key}_N{Nn}"] = {
            "status": out["status"], "fit_window": win, "w": w, "g": g, "fit_resid": res,
            "root": ref["root_wofz"], "grid": [gb["w"], gb["g"]], "grid_unc": [gb["unc_w"], gb["unc_g"]],
            "dw_root": w - ref["root_wofz"][0], "dg_root": g - ref["root_wofz"][1],
            "dw_grid": w - gb["w"], "dg_grid": g - gb["g"],
            "spectrax_6781d80": [ref["spectrax"][f"N{Nn}"]["w"], ref["spectrax"][f"N{Nn}"]["g"]],
            "compile_time": out["compile_time"], "run_time": out["run_time"]}
        arrays[f"B00_{key}_N{Nn}_t"], arrays[f"B00_{key}_N{Nn}_E"] = out["t"], E[:, 1]
        print(key, Nn, rec["B00"][f"{key}_N{Nn}"], flush=True)

b02 = json.loads((HERE / "refs/B02/run.json").read_text())["results"]
ms = (1, 2, 3, 4)
for vkey, vt in (("vt0.1", 0.1), ("vt0.3", 0.3)):
    ref = b02[vkey]
    out, E = run_es([(0.5, vt, 1.0), (0.5, vt, -1.0)], 2 * np.pi / 0.2, 16, 32, 60.0, 1201,
                    {m: 0.5e-12 for m in ms})
    for m in ms[1:]:  # k = 0.2 is contaminated by a nonlinear beat in the 4-mode reference run
        kk = f"k{0.2 * m:.1f}"
        A = np.abs(E[:, m])
        big = np.nonzero(A > 1e-5)[0]
        win = (25.0, min(60.0, out["t"][big[0]] if len(big) else 60.0))
        g = fit_growth(out["t"], A, win)
        root = ref["roots"][kk]["warm_root"][1]
        grid = ref["grid"]["Nv2048_dt0.0125"]["fits"][kk]["g"]
        rec["B02"][f"{vkey}_{kk}"] = {"status": out["status"], "window": win, "g": g, "warm_root": root,
                                      "grid": grid, "dg_root": g - root, "dg_grid": g - grid,
                                      "spectrax_6781d80_N32": ref["spectrax"]["N32"]["fits"][kk]["g"]}
        print(vkey, kk, rec["B02"][f"{vkey}_{kk}"], flush=True)
    arrays[f"B02_{vkey}_t"], arrays[f"B02_{vkey}_E"] = out["t"], E

d = HERE / "refs_rerun"
d.mkdir(exist_ok=True)
(d / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(d / "run.npz", **arrays)
