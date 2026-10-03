"""H05 drive-amplitude scan and H06 swept drive: HHS-v1-inspired, nonrelativistic pilots (not a reproduction).

Inputs as studies/hhs_ladder.py (v_te = sqrt(1e-3) c, m_i/m_e = 1836, T_i = T_e, L = 40 c/omega_pe, Nx = 8, the
same declared seed spectrum). Drive E0 cos(theta(t)) with E0 = v_q omega_pe (parent units, Omega_cs[0] = 1).
H05: theta = omega t, omega = sqrt(1 + 1/1836) (total omega_L, chosen because the ions move), v_q/v_te in
{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, t <= 1000, Hermite orders 32 and 64.
H06: theta = (omega0 + a t) t, instantaneous frequency omega0 + 2 a t. Up: omega0 = 0.8, a = +2e-5 (0.8 -> 1.2 over
t in [0, 1e4], resonance at t = 5000, the HHS v1 App. B/C reading); down: omega0 = 1.2, a = -2e-5. v_q/v_te = 0.1.
A homogeneous fixed-ion copy is checked against scipy solve_ivp of Ebar'' + Ebar = -E0 cos(theta).

Run: python studies/hhs_scan.py -> studies/hhs_scan/{run.json, run.npz}
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import darkspectrax as ds

OUT = Path(__file__).resolve().parent / "hhs_scan"
vte = np.sqrt(1e-3)
a_e, a_i = np.sqrt(2) * vte, np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx, w_tot = 40.0, np.sqrt(1 + 1 / 1836)
seeds = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j), (1, (1, 0, 0), 5e-4)]
rec = {"label": "HHS-v1-inspired, nonrelativistic pilot; not a reproduction", "parent_commit": ds.PARENT_COMMIT,
       "repository_commit": ds._simulation._git_sha(), "command": "python studies/hhs_scan.py",
       "seeds": "density 1e-3 at k1 (electrons and ions), 1e-4 electrons at k2", "omega_H05": w_tot}
arrays = {}


def mobile(E0, w, Nn, sweep=0.0, nu=0.0):
    return ds.Model(Nx=8, Nn=Nn, nu=nu, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836), alpha_s=(a_e,) * 3 + (a_i,) * 3,
                    u_s=(0.0,) * 6, mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=w, sweep_drive=sweep)


def go(m, T, nsave, max_steps=250_000):
    """Step budget keeps each run under a few minutes; exhaustion is reported as a failure, not hidden."""
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds))
    out = ds.run(m, y, T, n_save=nsave, rtol=1e-10, atol=1e-14, max_steps=max_steps)
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    out["t_reached"] = float(out["t"][good][-1])
    led = np.abs(out["ledger_defect"][:, good]).max(axis=0)
    scale = max(np.abs(out["W"][good, 2]).max(), 1e-300)
    bad = np.nonzero(led > 1e-6 * scale)[0]
    # last saved time before the work ledger departs from closure by more than 1e-6 of W_ext
    out["t_ledger_valid"] = float(out["t"][good][bad[0] - 1]) if bad.size else out["t_reached"]
    out["max_ledger_over_W_ext_valid"] = float(led[:bad[0]].max() / scale) if bad.size else float(led.max() / scale)
    stride = max(1, (nsave - 1) // 200)
    Ks = []
    for i in range(0, int(good.sum()), stride):
        st = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex)}
        Ks.append(np.asarray(ds.energies(m, st)["K_species"]))
    Ks = np.array(Ks)
    return out, out["t"][:int(good.sum()):stride], Ks - Ks[0]


H05 = {}
for ratio in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    E0 = ratio * vte
    for Nn, nu in ((32, 0.0), (64, 0.0), (64, 1.0)):
        m = mobile(E0, w_tot, Nn, nu=nu)
        out, tK, dK = go(m, 1000.0, 2001)
        Wx = out["W"][:, 2][np.isfinite(out["W"][:, 2])]
        key = f"vq{ratio:g}_Nn{Nn}_nu{nu:g}"
        H05[key] = {"status": out["status"], "failure_reason": out["failure_reason"],
                    "t_reached": out["t_reached"], "t_ledger_valid": out["t_ledger_valid"],
                    "max_ledger_over_W_ext_valid": out["max_ledger_over_W_ext_valid"], "W_ext_final": float(Wx[-1]),
                    "dK_electron_final": float(dK[-1, 0]), "dK_ion_final": float(dK[-1, 1]),
                    "ion_share_of_dK_final": float(dK[-1, 1] / (dK[-1, 0] + dK[-1, 1])),
                    "W_ext_over_n_Te_final": float(Wx[-1] / (vte ** 2)),
                    "steps": out["num_steps"], "run_time": out["run_time"], "compile_time": out["compile_time"]}
        arrays[f"H05_{key}_t"], arrays[f"H05_{key}_Wext"] = out["t"][:Wx.size], Wx
        arrays[f"H05_{key}_tK"], arrays[f"H05_{key}_dK"] = tK, dK
        print("H05", key, H05[key], flush=True)
    a, b = H05[f"vq{ratio:g}_Nn32_nu0"], H05[f"vq{ratio:g}_Nn64_nu0"]
    if a["status"] != "success" or b["status"] != "success":
        continue
    H05[f"vq{ratio:g}_Nn32_vs_64"] = {
        "W_ext_rel_diff": abs(a["W_ext_final"] - b["W_ext_final"]) / abs(b["W_ext_final"]),
        "dK_ion_rel_diff": abs(a["dK_ion_final"] - b["dK_ion_final"]) / max(abs(b["dK_ion_final"]), 1e-300)}
rec["H05"] = H05

H06 = {}
E0 = 0.1 * vte
for name, w0, a in (("up", 0.8, 2e-5), ("down", 1.2, -2e-5)):
    hom = ds.Model(Nx=1, Nn=3, alpha_s=(a_e,) * 3, rho_background=1.0, mode="prescribed_drive",
                   E_drive=(E0, 0.0, 0.0), omega_drive=w0, sweep_drive=a)
    oh = ds.run(hom, ds.maxwellian(hom, [1.0]), 1e4, n_save=2001, rtol=1e-11, atol=1e-16, max_steps=1_000_000)
    ref = solve_ivp(lambda t, y: [y[1], -y[0] - E0 * np.cos((w0 + a * t) * t)], (0, 1e4), [0.0, 0.0],
                    t_eval=oh["t"], rtol=1e-12, atol=1e-16, method="DOP853")
    for Nn, nu in ((32, 0.0), (64, 1.0)):
        m = mobile(E0, w0, Nn, a, nu)
        out, tK, dK = go(m, 1e4, 4001)
        key = f"{name}_Nn{Nn}_nu{nu:g}"
        H06[key] = {"omega0": w0, "a": a, "instantaneous_frequency": f"{w0} + {2 * a:g} t",
                    "homogeneous_max_err_over_max": float(np.abs(oh["Fk"][:, 0, 0, 0, 0].real - ref.y[0]).max()
                                                          / np.abs(ref.y[0]).max()),
                    "homogeneous_W_ext_final": float(oh["W"][-1, 2]),
                    "status": out["status"], "failure_reason": out["failure_reason"], "t_reached": out["t_reached"],
                    "t_ledger_valid": out["t_ledger_valid"], "W_ext_final": float(out["W"][np.isfinite(out["W"][:, 2]), 2][-1]),
                    "dK_electron_final": float(dK[-1, 0]), "dK_ion_final": float(dK[-1, 1]),
                    "max_ledger_over_W_ext_valid": out["max_ledger_over_W_ext_valid"],
                    "steps": out["num_steps"], "run_time": out["run_time"]}
        arrays[f"H06_{key}_t"], arrays[f"H06_{key}_Ebar"] = out["t"], out["Fk"][:, 0, 0, 0, 0].real
        arrays[f"H06_{key}_Wext"], arrays[f"H06_{key}_tK"], arrays[f"H06_{key}_dK"] = out["W"][:, 2], tK, dK
        arrays[f"H06_{key}_hom_Ebar"] = oh["Fk"][:, 0, 0, 0, 0].real
        print("H06", key, H06[key], flush=True)
rec["H06"] = H06

from hhs_summary import summarize  # noqa: E402

rec = summarize(rec, arrays)
OUT.mkdir(exist_ok=True)
(OUT / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(OUT / "run.npz", **arrays)
# (summary fields computed by studies/hhs_summary.py)
