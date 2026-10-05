"""Re-measure the B01 trapping rebound under the hypercollision closure with a detector-free envelope-ratio metric.

Retracted claim (studies/refs/B01, c-refs 2026-10-03): "nu = 1 at N = 256/512 removes the trapping rebound (no envelope
extrema)". That used find_peaks(prominence 0.3) on log env, which also finds no extremum in the nu = 0, N = 512 run.
Metrics (fixed before running), envelope = running max of |E_k1| over ~one plasma period (same as studies/b01_b06.py):
- R_fixed = log(env(65.6) / env(31.45)) at the grid's extremum times (Lane B's metric); retained = R / R_grid.
- R_window = log(max env on [50, 80] / min env on [20, 45]): extremum times free, no prominence threshold.
Sources: fresh runs (N = 256, 512; nu = 0, 1; parent order-2 spectrum, the c-refs closure) plus the committed
studies/b01_b06/run.npz arrays (ordinary and dark N = 512, nu = 0, 1; N = 1024, nu = 0).
Run: python studies/consolidate/b01_rebound.py -> studies/consolidate/b01_rebound.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
S = HERE.parent
beta, t1, t2 = 0.1, 31.45, 65.6


def env_of(t, E):
    return maximum_filter1d(np.abs(E), size=int(round(2 * np.pi / 1.1 / (t[1] - t[0]))), mode="nearest")


def metrics(t, E):
    env = env_of(t, E)
    rf = float(np.log(np.interp(t2, t, env) / np.interp(t1, t, env)))
    a, b = (t >= 20) & (t <= 45), (t >= 50) & (t <= 80)
    rw = float(np.log(env[b].max() / env[a].min()))
    return {"R_fixed": rf, "R_window": rw, "t_env_min_20_45": float(t[a][np.argmin(env[a])]),
            "t_env_max_50_80": float(t[b][np.argmax(env[b])])}


g = np.load(S / "refs/B01/grid_k0.3_eps0.05.npz")
res = {"grid": metrics(g["t"], g["E"])}
for Nn in (256, 512):
    for nu in (0.0, 1.0):
        m = ds.Model(Nx=16, Nn=Nn, Lx=2 * np.pi / 0.3 * beta, alpha_s=(np.sqrt(2) * beta,) * 3, rho_background=1.0, nu=nu)
        y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 0.025)]))
        out = ds.run(m, y, 100.0, n_save=2001, rtol=1e-10, atol=1e-14)
        res[f"fresh_ordinary_N{Nn}_nu{nu:g}"] = {"status": out["status"],
                                                 **metrics(out["t"], out["Fk"][:, 0, 0, 1, 0] / beta)}
rec = np.load(S / "b01_b06/run.npz")
for key in ("ordinary_N512_nu0.0", "ordinary_N512_nu1.0", "ordinary_N1024_nu0.0", "dark_N512_nu0.0",
            "dark_N512_nu1.0", "dark_N1024_nu0.0"):
    res[f"record_{key}"] = metrics(rec[f"B01_{key}_t"], rec[f"B01_{key}_E"])
for k, v in res.items():
    if k != "grid":
        v["retained_fixed"] = v["R_fixed"] / res["grid"]["R_fixed"]
        v["retained_window"] = v["R_window"] / res["grid"]["R_window"]
res["repository_commit"], res["parent_commit"] = ds._simulation._git_sha(), ds.PARENT_COMMIT
(HERE / "b01_rebound.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
