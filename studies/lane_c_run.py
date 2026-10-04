"""Lane C (no new regularization): one collisionless H05 pump-frame run, nu = 0.

Setup identical to studies/h05_pump_frame.py (v_te = sqrt(1e-3), m_i/m_e = 1836, T_i = T_e, L = 40, Nx = 8, mobile
ions, drive E0 cos(omega t), E0 = (v_q/v_te) v_te, omega = sqrt(1 + 1/1836), frame="pump", remap check every 20,
parent default triggers and caps, rtol 1e-10, atol 1e-14, 200k steps per segment). Only the save cadence differs:
41 saves per segment (every 0.5 omega_pe^-1) so that Q_i = <dn_i dE_x> and the quiver phase are not aliased.

Declared seed realizations (same amplitudes, phases rotated; chosen before any run):
  r0: the gate seeds  [(e, k1, 5e-4), (e, k2, 5e-5j), (i, k1, 5e-4)]
  r1: phases multiplied by exp(i*(2pi/3, pi/2, 4pi/3))
  r2: phases multiplied by exp(i*(4pi/3, 5pi/3, pi/3))
  r-1: no seeds (homogeneous control; W_ext must equal the uniform two-fluid oscillator)

Run: python studies/lane_c_run.py --vq 0.03 --Nn 64 --real 0 [--T 1000]  ->  studies/lane_c/runs/<case>.{json,npz}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds
from darkspectrax import _simulation as _sim
from darkspectrax._model import inner

OUT = Path(__file__).resolve().parent / "lane_c" / "runs"
vte = np.sqrt(1e-3)
a_e, a_i = np.sqrt(2) * vte, np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx, w_tot = 40.0, np.sqrt(1 + 1 / 1836)
BASE = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j), (1, (1, 0, 0), 5e-4)]
PHASES = {0: (0.0, 0.0, 0.0), 1: (2 * np.pi / 3, np.pi / 2, 4 * np.pi / 3), 2: (4 * np.pi / 3, 5 * np.pi / 3, np.pi / 3)}
SEG, NSAVE = 20.0, 41


def seeds(r):
    if r < 0:  # unseeded homogeneous control (H00-like): the exact uniform driven two-fluid oscillator
        return []
    return [(s, k, A * np.exp(1j * ph)) for (s, k, A), ph in zip(BASE, PHASES[r])]


def model(vq, Nn, Nx=8):
    return ds.Model(Nx=Nx, Nn=Nn, nu=0.0, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836),
                    alpha_s=(a_e,) * 3 + (a_i,) * 3, u_s=(0.0,) * 6, mode="prescribed_drive",
                    E_drive=(vq * vte, 0.0, 0.0), omega_drive=w_tot, frame="pump")


def diagnostics(m, out):
    """Per save: K_s, k=0 thermal energy (1/2 m sum_i (M_ii - M_i^2/n)), T_x, flow energy, Q_s = <dn_s dE_x>."""
    K, Th, Tx, Q = [], [], [], []
    for i in range(out["t"].size):
        st = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex),
              "B": out["B"][i].astype(complex)}
        K.append(np.asarray(ds.energies(m, st)["K_species"]))
        n, M, M2 = (np.asarray(x) for x in ds.moments(m, st["Ck"], st["B"]))
        n0, M0, M20 = n[:, 0, 0, 0].real, M[..., 0, 0, 0].real, M2[..., 0, 0, 0].real
        th = np.array([np.trace(M20[s]) - (M0[s] ** 2).sum() / n0[s] for s in range(m.Ns)])
        Th.append(0.5 * np.asarray(m.masses) * th)
        Tx.append(M20[:, 0, 0] - M0[:, 0] ** 2 / n0)
        Ex = np.array(st["Fk"][:1]).copy()
        Ex[..., 0, 0, 0] = 0.0
        q = []
        for s in range(m.Ns):
            dn = n[s][None].copy()
            dn[..., 0, 0, 0] = 0.0
            q.append(float(inner(m.Nx, dn, Ex)))
        Q.append(q)
    return np.array(K), np.array(Th), np.array(Tx), np.array(Q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--Nn", type=int, required=True)
    ap.add_argument("--real", type=int, default=0, choices=(-1, 0, 1, 2))
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--Nx", type=int, default=8)
    a = ap.parse_args(argv)
    stats = []
    run0 = _sim.run

    def counted(*args, **kw):  # aggregate per-segment solver statistics (run_adaptive does not keep them)
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    m = model(a.vq, a.Nn, a.Nx)
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(a.real)))
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, a.T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000)
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    K, Th, Tx, Q = diagnostics(m, out)
    st = np.array(stats)
    case = f"vq{a.vq:g}_Nn{a.Nn}_r{a.real}" + ("" if a.Nx == 8 else f"_Nx{a.Nx}")
    scale = max(np.abs(out["W"][:, 2]).max(), 1e-300)
    rec = {"case": case, "vq_over_vte": a.vq, "Nn": a.Nn, "Nx": a.Nx, "nu": 0.0, "realization": a.real, "phases": PHASES.get(a.real),
           "T_requested": a.T, "status": out["status"], "failure_reason": out["failure_reason"],
           "t_reached": float(out["t"][-1]), "events": len(out["events"]),
           "max_event_moment_defect": max([e["moment_defect"] for e in out["events"] if "moment_defect" in e],
                                          default=0.0),
           "max_ledger_over_W_ext": float(np.abs(out["ledger_defect"][:, good]).max() / scale),
           "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()), "compile_time": float(st[:, 2].sum()),
           "run_time": float(st[:, 3].sum()), "wall_time": wall,
           "repository_commit": _sim._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "command": f"python studies/lane_c_run.py --vq {a.vq:g} --Nn {a.Nn} --real {a.real} --T {a.T:g} --Nx {a.Nx}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{case}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(OUT / f"{case}.npz", t=out["t"], K=K, Th=Th, Tx=Tx, Q=Q, W=out["W"], B=out["B"],
                        U_gamma=out["U_gamma"], seg_stats=st,
                        Ek2=np.abs(out["Fk"][:, 0, 0, :, 0]) ** 2)  # |E_x(k)|^2 per rfft mode
    print(json.dumps(rec, default=float), flush=True)


if __name__ == "__main__":
    main()
