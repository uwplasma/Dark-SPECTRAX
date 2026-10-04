"""Lane A: H05 with basis policy + filter (no physical or hyper collisions), common strong-drive evaluation.

Setup identical to studies/h05_pump_frame.py (v_te = sqrt(1e-3) c, m_i/m_e = 1836, L = 40, Nx = 8, mobile ions,
same seeds, drive E0 cos(omega t), omega = sqrt(1 + 1/1836) = 1.000272, pump frame, checks every 20 omega_pe^-1).
Lane A regularization (fixed before the runs):
- width: continuous growth a_dot = max(0, (c sigma_e - a)/tau), tau = 10, from the measured k = 0 rms speed
  (Model.width_floor = c), never shrinking; remaps are shift-only (keep_width=True).
- filter: exponential filter in RHS form, -rate (n/(N-1))**16 C for n > 2 (SPECTRAX PR #61), nu = 0.
Base: Nn = 32, c = 1.3, rate = 1. Refinements: Nn 64; rate 0.5 and 2; c 1.1 and 1.5.
Resolved time t_res: first t > 50 where dK_e or W_ext differs by > 10% of the running max between the base and
any refinement (reported per observable and per refinement). Positivity: first time the reconstructed kinetic
energy or the k = 0 x-temperature of any species goes negative.

Run: python studies/lane_a_h05.py <vq/vte> -> studies/lane_a/vq<r>.{json,npz};
     python studies/lane_a_h05.py grid <vq/vte> <t_max> -> fixed-ion Hermite vs grid comparison.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds

OUT = Path(__file__).resolve().parent / "lane_a"
vte = np.sqrt(1e-3)
a_e, a_i = np.sqrt(2) * vte, np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx, w_tot = 40.0, np.sqrt(1 + 1 / 1836)
seeds = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j), (1, (1, 0, 0), 5e-4)]
SEG, NSAVE, TMAX, TAU, ORDER = 20.0, 11, 1000.0, 10.0, 16
BASE = dict(Nn=32, c=1.3, rate=1.0)
REFINE = {"Nn64": dict(Nn=64), "rate0.5": dict(rate=0.5), "rate2": dict(rate=2.0), "c1.1": dict(c=1.1),
          "c1.5": dict(c=1.5)}


def model(E0, Nn, c, rate, ions=True):
    kw = dict(qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836), alpha_s=(a_e,) * 3 + (a_i,) * 3, u_s=(0.0,) * 6) if ions else \
        dict(qs=(-1.0,), Omega_cs=(1.0,), alpha_s=(a_e,) * 3, rho_background=1.0)
    return ds.Model(Nx=8, Nn=Nn, Lx=Lx, mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=w_tot,
                    frame="pump", width_floor=c, width_tau=TAU, filter_rate=rate, filter_order=ORDER, **kw)


def thermal(m, out):
    K, T = [], []
    for i in range(out["t"].size):
        st = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex),
              "B": out["B"][i].astype(complex)}
        K.append(np.asarray(ds.energies(m, st)["K_species"]))
        n, M, M2 = (np.asarray(x)[..., 0, 0, 0].real for x in ds.moments(m, st["Ck"], st["B"]))
        T.append(M2[:, 0, 0] - M[:, 0] ** 2 / n)
    return np.array(K), np.array(T)


def go(m, T, ions=True):
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0] if ions else [1.0], seeds if ions else seeds[:2]))
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000,
                          trigger={"keep_width": True})
    out["wall"] = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    out["ledger_defect"] = out["ledger_defect"][:, good]
    K, Tx = thermal(m, out)
    return out, K, Tx


def disagreement(t, a, b, tol=0.1):
    """First t > 50 with |a - b| > tol * running max of max(|a|, |b|) (as in studies/h05_pump_frame.py)."""
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


def first_negative(t, x):
    bad = np.any(x <= 0, axis=1)
    return float(t[np.argmax(bad)]) if bad.any() else None


def scan(ratio):
    E0 = float(ratio) * vte
    rec = {"label": f"Lane A H05 v_q/v_te = {ratio}", "parent_commit": ds.PARENT_COMMIT,
           "repository_commit": ds._simulation._git_sha(), "command": f"python studies/lane_a_h05.py {ratio}",
           "base": BASE, "refinements": REFINE, "tau": TAU, "filter_order": ORDER, "omega": w_tot, "runs": {}}
    arrays, series = {}, {}
    for name, ch in {"base": {}, **REFINE}.items():
        p = {**BASE, **ch}
        m = model(E0, p["Nn"], p["c"], p["rate"])
        out, K, Tx = go(m, TMAX)
        dK = K - K[0]
        series[name] = (out["t"], dK[:, 0], out["W"][:, 2])
        scale = max(np.abs(out["W"][:, 2]).max(), 1e-300)
        rec["runs"][name] = {
            **p, "status": out["status"], "failure_reason": out["failure_reason"], "t_reached": float(out["t"][-1]),
            "remap_events": len(out["events"]), "cost": out["cost"], "wall_time": out["wall"],
            "max_ledger_over_W_ext": float(np.abs(out["ledger_defect"]).max() / scale),
            "final_a_e_over_a0": float(out["B"][-1, 1, 0] / a_e),
            "W_ext_final": float(out["W"][-1, 2]), "dK_electron_final": float(dK[-1, 0]),
            "first_negative_K": first_negative(out["t"], K), "first_negative_Tx": first_negative(out["t"], Tx)}
        arrays.update({f"{name}_t": out["t"], f"{name}_dK": dK, f"{name}_Wext": out["W"][:, 2],
                       f"{name}_B": out["B"], f"{name}_Tx": Tx})
        print(name, rec["runs"][name], flush=True)
    t0, k0, w0 = series["base"]
    A = {}
    for name in REFINE:
        t1, k1, w1 = series[name]
        n = min(t0.size, t1.size)
        A[name] = {"dK_e": disagreement(t0[:n], k0[:n], k1[:n]), "W_ext": disagreement(t0[:n], w0[:n], w1[:n]),
                   "common_end": float(t0[n - 1])}
    for obs in ("dK_e", "W_ext"):
        A[f"t_res_{obs}"] = min(v[obs] if v[obs] is not None else v["common_end"] for v in A.values()
                                if isinstance(v, dict))
    A["t_res"] = min(A["t_res_dK_e"], A["t_res_W_ext"])
    negs = [x for r in rec["runs"].values() for x in (r["first_negative_K"], r["first_negative_Tx"]) if x is not None]
    A["first_negative"] = min(negs, default=None)
    rec["agreement"] = A
    print("agreement", A, flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"vq{ratio}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(OUT / f"vq{ratio}.npz", **arrays)


def grid(ratio, Tc, vmax_over_vte=16, Nv=1024):
    """Fixed-ion comparison of the base lane-A Hermite run with the slv_proca.py primitives (same drive)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "refs" / "code"))
    from slv import VP

    E0 = float(ratio) * vte
    m = model(E0, BASE["Nn"], BASE["c"], BASE["rate"], ions=False)
    out, K, _ = go(m, Tc, ions=False)
    s = VP(Lx, 16, vmax_over_vte * vte, Nv)
    x, v = s.x, s.v
    k1 = 2 * np.pi / Lx
    f = (np.exp(-v[None, :] ** 2 / (2 * vte ** 2)) / np.sqrt(2 * np.pi * vte ** 2)
         * (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x))[:, None])
    rho = 1.0 - f.sum(axis=1) * s.dv
    Ek = np.fft.fft(rho) / s.Nx
    Ek[s.kx != 0] /= 1j * s.kx[s.kx != 0]
    Ek[0] = 0.0
    cur = lambda f: np.fft.fft(-(f * v[None, :]).sum(axis=1) * s.dv) / s.Nx  # noqa: E731
    dt, Wg, tg, Kg, Wh, edge = 0.02, 0.0, [], [], [], []
    K0g = 0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx
    tic = time.perf_counter()
    for n in range(int(round(Tc / dt))):
        t = n * dt
        if n % 50 == 0:
            tg.append(t), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0g), Wh.append(Wg)
            edge.append(np.abs(f[:, [0, -1]]).max() / f.max())
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
    tg.append(Tc), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0g), Wh.append(Wg)
    edge.append(np.abs(f[:, [0, -1]]).max() / f.max())
    tg, Kg, Wh, edge = np.array(tg), np.array(Kg), np.array(Wh), np.array(edge)
    th = np.round(out["t"], 6)
    keep = np.isin(np.round(tg, 6), th)
    tg, Kg, Wh, edge = tg[keep], Kg[keep], Wh[keep], edge[keep]
    sel = np.isin(th, np.round(tg, 6))
    dKh, Wxh = (K[:, 0] - K[0, 0])[sel], out["W"][sel, 2]
    rel = {name: (np.abs(h - g) / np.maximum.accumulate(np.maximum(np.abs(h), np.abs(g))))
           for name, h, g in (("dK_e", dKh, Kg), ("W_ext", Wxh, Wh))}
    rec = {"label": f"Lane A fixed-ion grid comparison v_q/v_te = {ratio}", "parent_commit": ds.PARENT_COMMIT,
           "repository_commit": ds._simulation._git_sha(),
           "command": f"python studies/lane_a_h05.py grid {ratio} {Tc:g}", "t_max": Tc,
           "grid": {"Nx": 16, "Nv": Nv, "vmax_over_vte": vmax_over_vte, "dt": dt,
                    "wall_time": time.perf_counter() - tic, "max_edge_f_over_max": float(edge.max())},
           "hermite": {**BASE, "status": out["status"], "failure_reason": out["failure_reason"],
                       "events": len(out["events"]), "cost": out["cost"], "wall_time": out["wall"],
                       "final_a_e_over_a0": float(out["B"][-1, 1, 0] / a_e)},
           "dK_e_final": {"hermite": float(dKh[-1]), "grid": float(Kg[-1])},
           "W_ext_final": {"hermite": float(Wxh[-1]), "grid": float(Wh[-1])},
           "first_disagreement_10pct": {"dK_e": disagreement(tg, dKh, Kg), "W_ext": disagreement(tg, Wxh, Wh)},
           "max_diff_over_running_max_t_gt_50": {k: float(r[tg > 50].max()) for k, r in rel.items()}}
    print("grid", rec, flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"grid_vq{ratio}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(OUT / f"grid_vq{ratio}.npz", t=tg, grid_dK=Kg, grid_W=Wh, herm_dK=dKh, herm_W=Wxh, edge=edge)


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        grid(sys.argv[2], float(sys.argv[3]), *(float(x) for x in sys.argv[4:5]), *(int(x) for x in sys.argv[5:6]))
    else:
        scan(sys.argv[1])
