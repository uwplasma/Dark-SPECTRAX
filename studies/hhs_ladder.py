"""HHS-v1-inspired, nonrelativistic ladder: H00, H01, H02 pilot, H07 pilot. Not a reproduction of HHS.

Inputs follow the HHS v1 manifest (arXiv:2510.13956v1 App. B as read in the literature check):
v_te = sqrt(1e-3) c, T_i = T_e, m_i/m_e = 1836, L = 40 c/omega_pe, uniform k = 0 drive E0 cos(omega t)
with no ramp. Drive amplitude in parent units: the force is q_s Omega_cs[s] E, so with Omega_cs[0] = 1
the electron acceleration amplitude is |E0|; E0 = v_q^D omega_pe, v_q^D/v_te = 0.03 (strong) or 1e-3 (weak).
Not taken from HHS: a relativistic pusher, 1D1V phase space, quiet particle start, their timestep.
Here: 1D3V Hermite (orders (Nn, 1, 1)), deterministic finite-k density seeds (declared below).

Run: python studies/hhs_ladder.py  -> studies/hhs/{run.json, run.npz}
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

import darkspectrax as ds

OUT = Path(__file__).resolve().parent / "hhs"
vte = np.sqrt(1e-3)
a_e = np.sqrt(2) * vte
a_i = np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx = 40.0
drives = {"strong": 0.03 * vte, "weak": 1e-3 * vte}
w_e, w_tot = 1.0, np.sqrt(1 + 1 / 1836)
rec = {"label": "HHS-v1-inspired, nonrelativistic; not a reproduction", "parent_commit": ds.PARENT_COMMIT,
       "repository_commit": ds._simulation._git_sha(), "command": "python studies/hhs_ladder.py",
       "inputs": {"v_te": vte, "alpha_e": a_e, "alpha_i": a_i, "mass_ratio": 1836, "Lx": Lx,
                  "E0": drives, "omega_electron": w_e, "omega_total": w_tot}}
arrays = {}
CONTROL_NOISE_FLOOR = ds.CONTROLS["noise_floor"]  # H00/H01 are controls (studies/noisefloor)


def Ebar_exact(t, E0, w, wL=1.0):
    """Fixed-ion mean field: Ebar'' + wL^2 Ebar = -wL^2 E0 cos(w t), zero initial data."""
    if abs(w - wL) < 1e-12:
        return -E0 * wL * t / 2 * np.sin(wL * t)
    return -E0 * wL ** 2 * (np.cos(w * t) - np.cos(wL * t)) / (wL ** 2 - w ** 2)


def Wext_exact(T, E0, w):
    """W_ext = int Om0 J.E_drive = -int Ebar' E_drive dt (Om0 = 1), by adaptive quadrature."""
    dE = (lambda t: -E0 * (t * np.cos(t) + np.sin(t)) / 2) if abs(w - 1) < 1e-12 else \
        (lambda t: -E0 * (-w * np.sin(w * t) + np.sin(t)) / (1 - w ** 2))
    return -quad(lambda t: dE(t) * E0 * np.cos(w * t), 0, T, limit=4000, epsabs=1e-18, epsrel=1e-12)[0]


def timed(fn, *a, **k):
    tic = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - tic


# ---------------- H00: homogeneous, fixed ions -----------------------------------------------
def h00(noise_floor=None):
    H00 = {}
    for dname, E0 in drives.items():
        for wname, w in (("omega_e", w_e), ("omega_tot", w_tot)):
            m = ds.Model(Nx=1, Nn=3, alpha_s=(a_e,) * 3, rho_background=1.0, mode="prescribed_drive",
                         E_drive=(E0, 0.0, 0.0), omega_drive=w)
            out = ds.run(m, ds.maxwellian(m, [1.0]), 1000.0, n_save=10001, rtol=1e-11, atol=1e-16,
                         noise_floor=noise_floor)
            t, Eb = out["t"], out["Fk"][:, 0, 0, 0, 0].real
            ref = Ebar_exact(t, E0, w)
            Wref = Wext_exact(1000.0, E0, w)
            key = f"{dname}_{wname}"
            H00[key] = {"status": out["status"], "max_abs_Ebar_err_over_max": float(np.abs(Eb - ref).max() / np.abs(ref).max()),
                        "W_ext_final": float(out["W"][-1, 2]), "W_ext_exact": Wref,
                        "W_ext_rel_err": float(abs(out["W"][-1, 2] - Wref) / abs(Wref)),
                        "ledger_defect": float(np.abs(out["ledger_defect"]).max()),
                        "max_quiver_over_vte": float(np.abs(ref).max() / vte),
                        "steps": out["num_steps"], "compile_time": out["compile_time"], "run_time": out["run_time"]}
            arrays[f"H00_{key}_t"], arrays[f"H00_{key}_E"] = t[::10], Eb[::10]
            print("H00", key, H00[key], flush=True)
    return H00


# ---------------- finite-k runs ----------------------------------------------------------------
def finite_k_model(E0, w, mobile, Nn):
    if mobile:
        return ds.Model(Nx=8, Nn=Nn, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836),
                        alpha_s=(a_e,) * 3 + (a_i,) * 3, u_s=(0.0,) * 6, mode="prescribed_drive", E_drive=(E0, 0.0, 0.0),
                        omega_drive=w)
    return ds.Model(Nx=8, Nn=Nn, Lx=Lx, alpha_s=(a_e,) * 3, rho_background=1.0, mode="prescribed_drive",
                    E_drive=(E0, 0.0, 0.0), omega_drive=w)


def seeds(mobile):
    """Declared deterministic seeds: 1e-3 density at k1 (both species if mobile), 1e-4 electron-only at k2."""
    p = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j)]
    return p + [(1, (1, 0, 0), 5e-4)] if mobile else p


def identity_residual(m, out, mobile):
    """R = Ebar'' + sum_s q_s^2 (Om_s/Om0) [n_s0 (Ebar + E_d) + <dn_s dE>], Ebar'' from the RHS itself."""
    om = np.asarray(m.Omega_cs) / m.Omega_cs[0]
    R, Qi = [], []
    for i, t in enumerate(out["t"]):
        y = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex)}
        dy = ds.rhs(t, y, m)
        _, dM, _ = ds.moments(m, dy["Ck"])
        dJ = float(np.real(np.tensordot(np.asarray(m.qs), np.asarray(dM[:, 0, 0, 0, 0]), axes=1)))
        n, _, _ = ds.moments(m, y["Ck"])
        Ex = y["Fk"][0]
        E0bar = float(Ex[0, 0, 0].real)
        Ed = float(m.drive(t)[0])
        rhs_sum = 0.0
        corr = []
        for s in range(m.Ns):
            c = float(ds.inner(m.Nx, np.asarray(n[s])[None], np.asarray(Ex)[None])) - float(n[s, 0, 0, 0].real) * E0bar
            corr.append(c)
            rhs_sum += m.qs[s] ** 2 * om[s] * (float(n[s, 0, 0, 0].real) * (E0bar + Ed) + c)
        R.append(-dJ / m.Omega_cs[0] + rhs_sum)
        Qi.append(corr[1] if mobile else 0.0)
    return np.array(R), np.array(Qi)


def finite_k(name, E0, w, mobile, T, Nn, nsave, noise_floor=None):
    m = finite_k_model(E0, w, mobile, Nn)
    dens = [1.0, 1.0] if mobile else [1.0]
    y = ds.consistent_fields(m, ds.maxwellian(m, dens, seeds(mobile)))
    out = ds.run(m, y, T, n_save=nsave, rtol=1e-10, atol=1e-14, max_steps=2_000_000,
                 noise_floor=noise_floor)
    (R, Qi), tdiag = timed(identity_residual, m, out, mobile)
    wL2 = 1 + (1 / 1836 if mobile else 0.0)
    t = out["t"]
    Eb = out["Fk"][:, 0, 0, 0, 0].real
    r = {"status": out["status"], "failure_reason": out["failure_reason"], "T": T, "Nn": Nn,
         "max_identity_residual_over_wL2E0": float(np.abs(R).max() / (wL2 * E0)),
         "ledger_defect": float(np.abs(out["ledger_defect"]).max()),
         "max_gauss": float(out["gauss"].max()), "steps": out["num_steps"],
         "compile_time": out["compile_time"], "run_time": out["run_time"], "diagnostic_time": tdiag}
    if not mobile:
        ref = Ebar_exact(t, E0, w)
        r["max_abs_Ebar_err_over_max"] = float(np.abs(Eb - ref).max() / np.abs(ref).max())
    else:
        # Q_i phase relative to the pump: least squares Q_i ~ A cos(w t) + B sin(w t) on the last half
        sel = t >= T / 2
        M = np.stack([np.cos(w * t[sel]), np.sin(w * t[sel])], 1)
        A, B = np.linalg.lstsq(M, Qi[sel], rcond=None)[0]
        r.update({"Qi_max_abs": float(np.abs(Qi).max()), "Qi_over_n0E0_max": float(np.abs(Qi).max() / E0),
                  "Qi_pump_quadrature_fit": [float(A), float(B)],
                  "Qi_phase_vs_pump_rad": float(np.arctan2(B, A)),
                  "Qi_fit_residual_rel": float(np.linalg.norm(M @ [A, B] - Qi[sel]) / np.linalg.norm(Qi[sel]))})
    arrays[f"{name}_t"], arrays[f"{name}_E"], arrays[f"{name}_R"] = t, Eb, R
    if mobile:
        arrays[f"{name}_Qi"] = Qi
    print(name, r, flush=True)
    return r


def h01(noise_floor=None):
    return {f"{d}_omega_e": finite_k(f"H01_{d}", drives[d], w_e, False, T, 32, int(T * 4) + 1, noise_floor)
            for d, T in (("weak", 1000.0), ("strong", 100.0))}


def main():
    rec["H00"] = h00(CONTROL_NOISE_FLOOR)
    rec["H01"] = h01(CONTROL_NOISE_FLOOR)
    rec["H02"] = {}
    gate = finite_k("H02_weak_pilot100", drives["weak"], w_tot, True, 100.0, 32, 401)
    rec["H02"]["weak_omega_tot_T100"] = gate
    gate_pass = gate["status"] == "success" and gate["max_identity_residual_over_wL2E0"] < 1e-6
    rec["H02"]["gate_T100_to_T1000"] = {"criterion": "status success and identity residual < 1e-6 (wL^2 E0)",
                                        "passed": bool(gate_pass)}
    if gate_pass:
        for wname, w in (("omega_tot", w_tot), ("omega_e", w_e)):
            rec["H02"][f"weak_{wname}_T1000"] = finite_k(f"H02_weak_{wname}", drives["weak"], w, True, 1000.0, 32, 4001)

    # ---------------- H07: finite reservoir vs prescribed drive, homogeneous fixed ions --------------
    H07 = {}
    for eta in (1e-3, 3e-2):
        E0 = drives["weak"]
        mp = ds.Model(Nx=1, Nn=3, alpha_s=(a_e,) * 3, rho_background=1.0, mode="prescribed_drive",
                      E_drive=(E0, 0.0, 0.0), omega_drive=1.0)
        ms = ds.Model(Nx=1, Nn=3, alpha_s=(a_e,) * 3, rho_background=1.0, mode="self_consistent", eta=eta, Omega_D=1.0)
        y0 = ds.maxwellian(ms, [1.0])
        ys = ds.proca_mode(ms, y0, (0, 0, 0), [-1j * E0 / eta, 0.0, 0.0])  # eta E_D(0) = E0, E_D ~ cos(t)
        op = ds.run(mp, ds.maxwellian(mp, [1.0]), 1000.0, n_save=4001, rtol=1e-11, atol=1e-16)
        osc = ds.run(ms, ys, 1000.0, n_save=4001, rtol=1e-11, atol=1e-16)
        t = osc["t"]
        Mx = np.array([[0, 1, eta, 0], [-1, 0, 0, 0], [-eta, 0, 0, 1], [0, 0, -1, 0]], float)
        v0 = [0.0, 0.0, float(ys["Dk"][0, 0, 0, 0].real), float(ys["Dk"][6, 0, 0, 0].real)]
        ref = np.array([expm(Mx * tt) @ v0 for tt in t[::40]])
        H07[f"eta{eta}"] = {
            "status": [op["status"], osc["status"]],
            "Ebar_vs_matrix_exponential_max_err_over_max": float(
                np.abs(osc["Fk"][::40, 0, 0, 0, 0].real - ref[:, 1]).max() / np.abs(ref[:, 1]).max()),
            "U_D0": float(osc["U_D"][0]), "W_D_final": float(osc["W"][-1, 1]),
            "max_fraction_of_reservoir_transferred": float(osc["W"][:, 1].max() / osc["U_D"][0]),
            "W_ext_prescribed_final": float(op["W"][-1, 2]),
            "max_abs_Ebar_prescribed": float(np.abs(op["Fk"][:, 0, 0, 0, 0].real).max()),
            "max_abs_Ebar_reservoir": float(np.abs(osc["Fk"][:, 0, 0, 0, 0].real).max()),
            "beat_period_estimate_2pi_over_eta": 2 * np.pi / eta,
            "ledger_defect_reservoir": float(np.abs(osc["ledger_defect"]).max()),
            "run_time": [op["run_time"], osc["run_time"]]}
        arrays[f"H07_eta{eta}_t"] = t
        arrays[f"H07_eta{eta}_Ebar_prescribed"] = op["Fk"][:, 0, 0, 0, 0].real
        arrays[f"H07_eta{eta}_Ebar_reservoir"] = osc["Fk"][:, 0, 0, 0, 0].real
        arrays[f"H07_eta{eta}_W_D"], arrays[f"H07_eta{eta}_W_ext"] = osc["W"][:, 1], op["W"][:, 2]
        print("H07", eta, H07[f"eta{eta}"], flush=True)
    rec["H07"] = H07

    OUT.mkdir(exist_ok=True)
    (OUT / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(OUT / "run.npz", **arrays)


if __name__ == "__main__":
    main()
