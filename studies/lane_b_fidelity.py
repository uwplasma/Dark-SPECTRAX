"""Lane B: physical-fidelity cost of the order >= 2 hypercollision closure on B01 and B06.

Same inputs, kick and windows as studies/b01_b06.py (ordinary Vlasov, beta = 0.1), against the grid references in
studies/refs. Closure: nu in {0, 0.25, 0.5, 1, 2, 4} x order in {2, 3}, spectrum installed in Model.p before tracing.
Metrics fixed before running:
- B01 (k = 0.3, eps = 0.05, t <= 100): trapping rebound R = log(env(t_max^grid) / env(t_min^grid)) with grid
  t_min = 31.45, t_max = 65.6; erased fraction = 1 - R / R_grid; first t with |log env/env_grid| > 0.1.
- B06 (k1 = 1, k2 = 1.5 kick at tau = 10, echo at k3 = 0.5): echo amplitude relative to the grid (amp/amp_grid - 1)
  and echo time; max k1 recurrence after t = 20.

Run: python studies/lane_b_fidelity.py {B01|B06} N -> studies/lane_b/fidelity_{case}_N{N}.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import maximum_filter1d
from spectrax._initialization import hypercollision_spectrum

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
beta = 0.1
NUS, ORDERS = (0.25, 0.5, 1.0, 2.0, 4.0), (2, 3)


def model(k0, Nx, Nn, nu, order):
    m = ds.Model(Nx=Nx, Nn=Nn, Lx=2 * np.pi / k0 * beta, alpha_s=(np.sqrt(2) * beta,) * 3, rho_background=1.0, nu=nu)
    m.p["collision_matrix"] = hypercollision_spectrum(Nn, 1, 1, order=order)
    return m


def env_of(t, E):
    w = int(round(2 * np.pi / 1.1 / (t[1] - t[0])))
    return maximum_filter1d(np.abs(E), size=w, mode="nearest")


def b01(Nn):
    g = np.load(HERE / "refs/B01/grid_k0.3_eps0.05.npz")
    tg, envg = g["t"], env_of(g["t"], g["E"])
    t1, t2 = 31.45, 65.6
    Rg = float(np.log(np.interp(t2, tg, envg) / np.interp(t1, tg, envg)))
    res = {"grid": {"t_min": t1, "t_max": t2, "R": Rg}}
    for nu, order in [(0.0, 2)] + [(n, o) for o in ORDERS for n in NUS]:
        m = model(0.3, 16, Nn, nu, order)
        y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 0.025)]))
        out = ds.run(m, y, 100.0, n_save=2001, rtol=1e-10, atol=1e-14)
        t, E = out["t"], out["Fk"][:, 0, 0, 1, 0] / beta
        env = env_of(t, E)
        R = float(np.log(np.interp(t2, t, env) / np.interp(t1, t, env)))
        dev = np.abs(np.log(env / np.interp(t, tg, envg)))
        res[f"nu{nu:g}_o{order}"] = {
            "status": out["status"], "R": R, "erased_fraction": 1 - R / Rg,
            "env_ratio_t65p6": float(np.interp(t2, t, env) / np.interp(t2, tg, envg)),
            "first_t_logenv_dev_gt_0p1": float(t[np.argmax(dev > 0.1)]) if dev.max() > 0.1 else None,
            "steps": out["num_steps"], "rejected": out["num_rejected"], "run_time": out["run_time"]}
        print(Nn, nu, order, res[f"nu{nu:g}_o{order}"], flush=True)
    return res


def kick(Ck, Nn, Nx, L, a, m2, d2):
    Nf = 128
    Cf = np.zeros((Nn, Nf // 2 + 1), complex)
    Cf[:, :Nx // 2 + 1] = Ck[:, 0, :, 0]
    Cx = np.fft.irfft(Cf, n=Nf, axis=1, norm="forward")
    xf = np.arange(Nf) * L / Nf
    R = np.diag(np.sqrt(2 * np.arange(1, Nn)), -1)
    s = beta * d2 * np.cos(m2 * 2 * np.pi * xf / L) / a
    for j in range(Nf):
        Cx[:, j] = expm(s[j] * R) @ Cx[:, j]
    out = Ck.copy()
    out[:, 0, :, 0] = np.fft.rfft(Cx, axis=1, norm="forward")[:, :Nx // 2 + 1]
    return out


def b06(Nn):
    k0, m1, m2, tau, eps1, d2, tmax = 0.25, 4, 6, 10.0, 0.01, 0.05, 45.0
    m3 = m2 - m1
    ge = np.load(HERE / "refs/B06/grid_Nv1024_dt0.0125.npz")
    tg, Eg = ge["t"], np.abs(ge["E"][:, 2])
    ig = np.argmax(Eg * (tg > tau + 5))
    res = {"grid": {"t_echo": float(tg[ig]), "amp": float(Eg[ig])}}
    for nu, order in [(0.0, 2)] + [(n, o) for o in ORDERS for n in NUS]:
        m = model(k0, 32, Nn, nu, order)
        y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (m1, 0, 0), eps1 / 2)]))
        a1 = ds.run(m, y, tau, n_save=201, rtol=1e-10, atol=1e-14)
        yk = {"Ck": kick(a1["Ck"][-1], Nn, 32, m.Lx, m.alpha_s[0], m2, d2), "Fk": a1["Fk"][-1], "Dk": a1["Dk"][-1],
              "W": a1["W"][-1].astype(complex)}
        a2 = ds.run(m, yk, tmax - tau, n_save=701, rtol=1e-10, atol=1e-14, t0=tau)
        t = np.concatenate([a1["t"], a2["t"][1:]])
        E3 = np.abs(np.concatenate([a1["Fk"][:, 0, 0, m3, 0], a2["Fk"][1:, 0, 0, m3, 0]])) / beta
        E1 = np.abs(np.concatenate([a1["Fk"][:, 0, 0, m1, 0], a2["Fk"][1:, 0, 0, m1, 0]])) / beta
        i = np.argmax(E3 * (t > tau + 5))
        res[f"nu{nu:g}_o{order}"] = {
            "status": [a1["status"], a2["status"]], "t_echo": float(t[i]), "amp": float(E3[i]),
            "amp_rel_to_grid": float(E3[i] / res["grid"]["amp"] - 1),
            "max_k1_after_t20": float(E1[t > 20].max()),
            "steps": a1["num_steps"] + a2["num_steps"], "run_time": a1["run_time"] + a2["run_time"]}
        print(Nn, nu, order, res[f"nu{nu:g}_o{order}"], flush=True)
    return res


if __name__ == "__main__":
    case, Nn = sys.argv[1], int(sys.argv[2])
    tic = time.perf_counter()
    res = {"B01": b01, "B06": b06}[case](Nn)
    res.update(case=case, Nn=Nn, wall=time.perf_counter() - tic, repository_commit=ds._simulation._git_sha(),
               parent_commit=ds.PARENT_COMMIT)
    (HERE / "lane_b").mkdir(exist_ok=True)
    (HERE / "lane_b" / f"fidelity_{case}_N{Nn}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
