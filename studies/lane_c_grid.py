"""Lane C independent grid check: collisionless Hermite pump frame (Nn 64, nu 0) vs the grid Vlasov-Ampere
primitives of studies/refs/code/slv.py, fixed ions, seed r0 electron part, t <= 300, identical to the gate's
check in studies/h05_pump_frame.py but for v_q/v_te in {0.01, 0.03, 0.1}. 10% of the running max on dK_e, W_ext.

Run: python studies/lane_c_grid.py -> studies/lane_c/grid.json, grid.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds
from darkspectrax import _simulation as _sim

sys.path.insert(0, str(Path(__file__).resolve().parent / "refs" / "code"))
from slv import VP  # noqa: E402

OUT = Path(__file__).resolve().parent / "lane_c"
vte = np.sqrt(1e-3)
a_e = np.sqrt(2) * vte
Lx, w_tot, Tc = 40.0, np.sqrt(1 + 1 / 1836), 300.0
seeds = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j)]


def disagreement(t, a, b, tol=0.1):
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


def grid(E0, Nv=1024, dt=0.02):
    s = VP(Lx, 16, 16 * vte, Nv)
    x, v = s.x, s.v
    k1 = 2 * np.pi / Lx
    f = (np.exp(-v[None, :] ** 2 / (2 * vte ** 2)) / np.sqrt(2 * np.pi * vte ** 2)
         * (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x))[:, None])
    rho = 1.0 - f.sum(axis=1) * s.dv
    Ek = np.fft.fft(rho) / s.Nx
    Ek[s.kx != 0] /= 1j * s.kx[s.kx != 0]
    Ek[0] = 0.0
    cur = lambda f: np.fft.fft(-(f * v[None, :]).sum(axis=1) * s.dv) / s.Nx  # noqa: E731
    Wg, tg, Kg, Wh = 0.0, [], [], []
    K0 = 0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx
    every = int(round(1.0 / dt))
    for n in range(int(round(Tc / dt))):
        t = n * dt
        if n % every == 0:
            tg.append(t), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0), Wh.append(Wg)
        f = s.adv_x(f, dt / 2)
        Jpre, n_x = cur(f), f.sum(axis=1) * s.dv
        Ed = E0 * np.cos(w_tot * (t + dt / 2))
        J = Jpre
        for _ in range(4):
            Emid = Ek - dt / 2 * J
            force = np.fft.ifft(Emid * s.Nx).real + Ed
            J = Jpre + 0.5 * dt * np.fft.fft(n_x * force) / s.Nx
        Wg += dt * J[0].real * Ed
        f = s.shift_v(f, -force * dt)
        Ek = Ek - dt * J
        f = s.adv_x(f, dt / 2)
    tg.append(Tc), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0), Wh.append(Wg)
    return np.array(tg), np.array(Kg), np.array(Wh), float(np.abs(f[:, [0, -1]]).max() / f.max())


rec, arrays = {"repository_commit": _sim._git_sha(), "parent_commit": ds.PARENT_COMMIT,
               "command": "python studies/lane_c_grid.py"}, {}
for vq in (0.01, 0.03, 0.1):
    E0 = vq * vte
    m = ds.Model(Nx=8, Nn=64, Lx=Lx, qs=(-1.0,), Omega_cs=(1.0,), alpha_s=(a_e,) * 3, rho_background=1.0,
                 mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=w_tot, frame="pump")
    tic = time.perf_counter()
    out = ds.run_adaptive(m, ds.consistent_fields(m, ds.maxwellian(m, [1.0], seeds)), Tc, 20.0, n_save_segment=21,
                          rtol=1e-10, atol=1e-14, max_steps=200_000)
    wh = time.perf_counter() - tic
    good = np.isfinite(out["t"])
    th = np.round(out["t"][good], 6)
    K = np.array([float(ds.energies(m, {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i],
                                        "W": out["W"][i].astype(complex), "B": out["B"][i].astype(complex)})["K"])
                  for i in np.flatnonzero(good)])
    dKh, Wxh = K - K[0], out["W"][good, 2]
    res = {}
    for Nv, dt in ((1024, 0.02), (2048, 0.01)):  # grid self-refinement
        tic = time.perf_counter()
        tg, Kg, Wg, edge = grid(E0, Nv, dt)
        keep, sel = np.isin(np.round(tg, 6), th), np.isin(th, np.round(tg, 6))
        tg, Kg, Wg = tg[keep], Kg[keep], Wg[keep]
        h, w = dKh[sel], Wxh[sel]
        res[f"Nv{Nv}_dt{dt:g}"] = {
            "wall_time": time.perf_counter() - tic, "edge_f_over_max": edge,
            "first_disagreement_10pct": {"dK_e": disagreement(tg, h, Kg), "W_ext": disagreement(tg, w, Wg)},
            "max_diff_over_running_max_t_gt_50": {
                nm: float(np.max((np.abs(a - b) / np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))))[tg > 50]))
                for nm, a, b in (("dK_e", h, Kg), ("W_ext", w, Wg))},
            "W_ext_final": {"hermite": float(w[-1]), "grid": float(Wg[-1])},
            "dK_e_final": {"hermite": float(h[-1]), "grid": float(Kg[-1])}}
        arrays.update({f"vq{vq:g}_Nv{Nv}_t": tg, f"vq{vq:g}_Nv{Nv}_dK": Kg, f"vq{vq:g}_Nv{Nv}_W": Wg})
    arrays.update({f"vq{vq:g}_herm_t": th, f"vq{vq:g}_herm_dK": dKh, f"vq{vq:g}_herm_W": Wxh})
    rec[f"vq{vq:g}"] = {"hermite": {"Nn": 64, "status": out["status"], "failure_reason": out["failure_reason"],
                                     "t_reached": float(th[-1]), "events": len(out["events"]), "wall_time": wh},
                        "grid": res}
    print(vq, json.dumps(rec[f"vq{vq:g}"], default=float), flush=True)
OUT.mkdir(exist_ok=True)
(OUT / "grid.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(OUT / "grid.npz", **arrays)
