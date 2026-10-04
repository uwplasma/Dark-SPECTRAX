"""H05 gate for the pump frame + segment remaps (SPECTRAX PRs #56-#60), set before running.

Same inputs as studies/hhs_scan.py H05 (v_te = sqrt(1e-3) c, m_i/m_e = 1836, T_i = T_e, L = 40 c/omega_pe, Nx = 8,
same seeds, drive E0 cos(omega t) with E0 = v_q, omega = sqrt(1 + 1/1836)), now with Model(frame="pump") and
run_adaptive (remap check every 20 omega_pe^-1, parent default triggers and caps).

Gate (fixed before the runs; see the adaptive-basis plan):
1. v_q/v_te = 0.03 and 0.1 to omega_pe t = 1000. Refinement directions: Hermite order (Nn 32 vs 64, nu = 0) and
   closure (nu 0 vs 1 at Nn 64). Resolved until the first time (t > 50) either pair differs by more than 10% on
   the electron kinetic-energy change or on W_ext (relative to the running maximum magnitude, see disagreement).
2. Moment-reconstructed kinetic energy and k = 0 temperature of each species stay positive.
3. Rejection: at v_q/v_te = 0.1 the resolved time must be at least 3x the fixed-basis value (210, from
   studies/hhs_scan/run.json); otherwise record the failure.
4. v_q/v_te = 0.03, fixed ions, t <= 300: Hermite pump frame (Nn = 64) against the independent grid
   Vlasov-Ampere code (studies/refs/code/slv.py primitives, with the same drive); 10% on dK_e and W_ext.

Run: python studies/h05_pump_frame.py -> studies/h05_pump/{run.json, run.npz}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds

sys.path.insert(0, str(Path(__file__).resolve().parent / "refs" / "code"))
from slv import VP  # noqa: E402

OUT = Path(__file__).resolve().parent / "h05_pump"
vte = np.sqrt(1e-3)
a_e, a_i = np.sqrt(2) * vte, np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx, w_tot = 40.0, np.sqrt(1 + 1 / 1836)
seeds = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j), (1, (1, 0, 0), 5e-4)]
FIXED_BASIS_RESOLVED = {"0.03": 365.0, "0.1": 210.0}  # studies/hhs_scan/run.json, H05 *_agreement.resolved_until
SEG, NSAVE = 20.0, 11
rec = {"label": "H05 pump frame + remap gate (HHS-v1-inspired, nonrelativistic pilot)",
       "parent_commit": ds.PARENT_COMMIT, "repository_commit": ds._simulation._git_sha(),
       "command": "python studies/h05_pump_frame.py", "segment": SEG, "fixed_basis_resolved_until": FIXED_BASIS_RESOLVED}
arrays = {}


def mobile(E0, Nn, nu):
    return ds.Model(Nx=8, Nn=Nn, nu=nu, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836), alpha_s=(a_e,) * 3 + (a_i,) * 3,
                    u_s=(0.0,) * 6, mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=w_tot, frame="pump")


def thermal(model, out):
    """Per save: kinetic energy per species and the k = 0 x-temperature n T = M_xx - M_x^2/n."""
    K, T = [], []
    for i in range(out["t"].size):
        st = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex),
              "B": out["B"][i].astype(complex)}
        K.append(np.asarray(ds.energies(model, st)["K_species"]))
        n, M, M2 = (np.asarray(x)[..., 0, 0, 0].real for x in ds.moments(model, st["Ck"], st["B"]))
        T.append(M2[:, 0, 0] - M[:, 0] ** 2 / n)
    return np.array(K), np.array(T)


def go(model, y0, T):
    tic = time.perf_counter()
    out = ds.run_adaptive(model, y0, T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000)
    out["wall"] = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    out["ledger_defect"] = out["ledger_defect"][:, good]
    K, Tx = thermal(model, out)
    return out, K, Tx


def disagreement(t, a, b, tol=0.1):
    """First t > 50 with |a - b| > tol * (running max of max(|a|, |b|)); the running max keeps oscillating
    observables (dK_e carries the quiver energy and passes near zero) from failing at their zero crossings."""
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


H05 = {}
for ratio in ("0.03", "0.1"):
    E0 = float(ratio) * vte
    runs = {}
    for Nn, nu in ((32, 0.0), (64, 0.0), (64, 1.0)):
        m = mobile(E0, Nn, nu)
        out, K, Tx = go(m, ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds)), 1000.0)
        key = f"vq{ratio}_Nn{Nn}_nu{nu:g}"
        dK = K - K[0]
        runs[(Nn, nu)] = (out["t"], dK[:, 0], out["W"][:, 2])
        scale = max(np.abs(out["W"][:, 2]).max(), 1e-300)
        H05[key] = {"status": out["status"], "failure_reason": out["failure_reason"], "t_reached": float(out["t"][-1]),
                    "events": len(out["events"]), "max_event_moment_defect": max([e["moment_defect"] for e in out["events"]],
                                                                                 default=0.0),
                    "final_basis_electron_x": [float(out["B"][-1, 0, 0]), float(out["B"][-1, 1, 0])],
                    "max_ledger_over_W_ext": float(np.abs(out["ledger_defect"]).max() / scale),
                    "W_ext_final": float(out["W"][-1, 2]), "dK_electron_final": float(dK[-1, 0]),
                    "dK_ion_final": float(dK[-1, 1]), "min_K_species": K.min(axis=0).tolist(),
                    "min_T_x_species": Tx.min(axis=0).tolist(), "wall_time": out["wall"]}
        arrays[f"{key}_t"], arrays[f"{key}_dK"], arrays[f"{key}_Wext"] = out["t"], dK, out["W"][:, 2]
        arrays[f"{key}_B"] = out["B"]
        print("H05", key, H05[key], flush=True)
    A = {}
    for name, (r1, r2) in {"Nn32_vs_Nn64_nu0": ((32, 0.0), (64, 0.0)), "nu0_vs_nu1_Nn64": ((64, 0.0), (64, 1.0))}.items():
        (t1, k1, w1), (t2, k2, w2) = runs[r1], runs[r2]
        n = min(t1.size, t2.size)
        A[name] = {"dK_e": disagreement(t1[:n], k1[:n], k2[:n]), "W_ext": disagreement(t1[:n], w1[:n], w2[:n]),
                   "common_end": float(t1[n - 1])}
    ends = [v if v is not None else A[nm]["common_end"] for nm in A for k, v in A[nm].items() if k != "common_end"]
    A["resolved_until"] = min(ends)
    A["gain_over_fixed_basis"] = A["resolved_until"] / FIXED_BASIS_RESOLVED[ratio]
    A["positive_kinetic_energy"] = all(min(H05[k]["min_K_species"]) > 0 and min(H05[k]["min_T_x_species"]) > 0
                                       for k in H05 if k.startswith(f"vq{ratio}_"))
    H05[f"vq{ratio}_agreement"] = A
    print("H05", ratio, A, flush=True)
g = H05["vq0.1_agreement"]["gain_over_fixed_basis"]
rec["gate"] = {"resolved_gain_vq0.1": g, "rejected": bool(g < 3.0),
               "verdict": "pass" if g >= 3.0 else "FAIL: pump frame + remap extends the resolved time by less than 3x"}
rec["H05"] = H05

# 4. Independent grid comparison, fixed ions, v_q/v_te = 0.03, t <= 300.
E0, Tc = 0.03 * vte, 300.0
m = ds.Model(Nx=8, Nn=64, Lx=Lx, qs=(-1.0,), Omega_cs=(1.0,), alpha_s=(a_e,) * 3, rho_background=1.0,
             mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=w_tot, frame="pump")
y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0], seeds[:2]))
out, K, _ = go(m, y0, Tc)
s = VP(Lx, 16, 16 * vte, 1024)
x, v = s.x, s.v
k1 = 2 * np.pi / Lx
f = (np.exp(-v[None, :] ** 2 / (2 * vte ** 2)) / np.sqrt(2 * np.pi * vte ** 2)
     * (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x))[:, None])
rho = 1.0 - f.sum(axis=1) * s.dv
Ek = np.fft.fft(rho) / s.Nx
Ek[s.kx != 0] /= 1j * s.kx[s.kx != 0]
Ek[0] = 0.0
cur = lambda f: np.fft.fft(-(f * v[None, :]).sum(axis=1) * s.dv) / s.Nx  # noqa: E731
dt, Wg, tg, Kg, Wh = 0.02, 0.0, [], [], []
K0g = 0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx
tic = time.perf_counter()
for n in range(int(round(Tc / dt))):
    t = n * dt
    if n % 50 == 0:
        tg.append(t), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0g), Wh.append(Wg)
    f = s.adv_x(f, dt / 2)
    Jpre, n_x = cur(f), f.sum(axis=1) * s.dv
    Ed = E0 * np.cos(w_tot * (t + dt / 2))
    J = Jpre
    for _ in range(4):  # mid-step current, as in slv_proca.py
        Emid = Ek - dt / 2 * J
        force = np.fft.ifft(Emid * s.Nx).real + Ed
        J = Jpre + 0.5 * dt * np.fft.fft(n_x * force) / s.Nx
    Wg += dt * J[0].real * Ed  # P_ext = <J> E_drive
    f = s.shift_v(f, -force * dt)
    Ek = Ek - dt * J
    f = s.adv_x(f, dt / 2)
tg.append(Tc), Kg.append(0.5 * (f * v[None, :] ** 2).sum() * s.dv / s.Nx - K0g), Wh.append(Wg)
tg, Kg, Wh = np.array(tg), np.array(Kg), np.array(Wh)
# compare at common save times (every 2 omega_pe^-1); interpolating the quiver-oscillating dK_e is not valid
th = np.round(out["t"], 6)
keep = np.isin(np.round(tg, 6), th)
tg, Kg, Wh = tg[keep], Kg[keep], Wh[keep]
sel = np.isin(th, np.round(tg, 6))
dKh, Wxh = (K[:, 0] - K[0, 0])[sel], out["W"][sel, 2]
rec["grid_comparison_vq0.03"] = {
    "t_max": Tc, "grid": {"Nx": 16, "Nv": 1024, "vmax_over_vte": 16, "dt": dt, "wall_time": time.perf_counter() - tic,
                          "edge_f_over_max": float(np.abs(f[:, [0, -1]]).max() / f.max())},
    "hermite": {"Nn": 64, "status": out["status"], "events": len(out["events"]), "wall_time": out["wall"]},
    "dK_e_final": {"hermite": float(dKh[-1]), "grid": float(Kg[-1])},
    "W_ext_final": {"hermite": float(Wxh[-1]), "grid": float(Wh[-1])},
    "first_disagreement_10pct": {"dK_e": disagreement(tg, dKh, Kg), "W_ext": disagreement(tg, Wxh, Wh)},
    "max_diff_over_running_max_t_gt_50": {
        name: float(np.max((np.abs(h - g) / np.maximum.accumulate(np.maximum(np.abs(h), np.abs(g))))[tg > 50]))
        for name, h, g in (("dK_e", dKh, Kg), ("W_ext", Wxh, Wh))}}
arrays.update(grid_t=tg, grid_dK=Kg, grid_W=Wh, herm_dK=dKh, herm_W=Wxh)
print("grid", rec["grid_comparison_vq0.03"], flush=True)

OUT.mkdir(exist_ok=True)
(OUT / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(OUT / "run.npz", **arrays)
