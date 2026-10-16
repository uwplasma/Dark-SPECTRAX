"""B11 collisional relaxation with the parent's conservative Dougherty operator (SPECTRAX #71).

Operator (Hakim, Francisquez, Juno, Hammett, arXiv:1903.08062, eqs. 3-8; Dougherty 1964):
C_ss = nu_ss d_v.[(v - U_s) f + theta_s d_v f], theta_s = tr P_s / (d m_s n_s), d = number of velocity axes with
N >= 3. Pair operator C_sr: same form with u_sr = u_rs = (U_s + U_r)/2 and a common
T_sr = (m_r T_s + m_s T_r)/(m_s + m_r) + m_s m_r |U_s - U_r|^2 / (2 d (m_s + m_r)), with m_s n_s nu_sr = m_r n_r nu_rs
(Greene's form, beta = 0; derived in #71 from R_sr + R_rs = 0 and Q_sr + Q_rs = 0).

All cases are spatially homogeneous (Nx = 1) and uncharged (qs = 0), which isolates the collision operator: no field,
so every exact rate below is the full dynamics. Parent units; masses m_s = Omega_cs[0] / Omega_cs[s].
Solver: Dopri8, rtol = atol = 1e-12.

Analytic references (exact for this operator, derived from it):
A  two-beam relaxation, 1V: f = [M(v - d) + M(v + d)]/2, beams of width vt. In the shifted-centred basis
   (u = U = 0, a^2 = 2 theta, theta = vt^2 + d^2) each Hermite mode obeys dC_n/dt = -n nu C_n exactly, so
   C_n(t) = C_n(0) exp(-n nu t). nu = 1, d = 0.6, vt = 0.5, Nn = 96, t = 0..5.
B  temperature anisotropy, 3V, (Nn, Nm, Np) = (12, 12, 12): Maxwellian with T = (0.25, 0.16, 0.09), isotropic basis
   a^2 = 2 theta. dT_i/dt = 2 nu (theta - T_i), theta fixed: T_i - theta = (T_i(0) - theta) exp(-2 nu t). nu = 0.5,
   t = 0..4.
C  two-species temperature equilibration, 1V, Nn = 64: Maxwellians at rest, n = 1 each, T_s = 1, T_r = 0.2,
   m_r/m_s = 4 and 100. dT_s/dt = -2 nu_sr (T_s - T_sr) with T_sr = (m_r T_s + m_s T_r)/(m_s + m_r) gives
   T_s - T_r = (T_s - T_r)(0) exp(-2 (nu_sr m_s + nu_rs m_r) t / (m_s + m_r)), nu_rs = nu_sr m_s / m_r.
   nu_sr = 1, self collisions nu_ss = 0.5, nu_rr = 0.2 (they do not change T); t = 0..3 (m = 4), 0..30 (m = 100).
D  two-species drift (displaced Maxwellians), 1V, Nn = 64, m_r/m_s = 4: U_s = +0.3, U_r = -0.075 (zero total
   momentum), T = 0.5 each. dU_s/dt = -nu_sr (U_s - U_r)/2, so U_s - U_r = (U_s - U_r)(0) exp(-(nu_sr + nu_rs) t / 2);
   the lost drift energy becomes heat. nu_sr = 1, t = 0..4.

Gates (declared before running):
G1 (A) for n = 4, 6, 8: max_t |C_n(t) - C_n(0) exp(-n nu t)| <= 1e-10 |C_0|, and the relative error
   |C_n(t) / (C_n(0) exp(-n nu t)) - 1| <= 1e-6 while exp(-n nu t) >= 1e-6; rows 0-2 constant to 1e-12 relative.
   (An uncommitted pre-run used a relative error over the whole window; it is meaningless once C_n decays to
   1e-18 against atol = 1e-12, where it reached 8e-7 for n = 8, so G1 was restated before the recorded run.)
G2 (B) max_t |(T_i - theta) - (T_i(0) - theta) exp(-2 nu t)| <= 1e-8 theta for each axis; theta constant to 1e-12.
G3 (C) max_t |dT(t) - dT(0) exp(-lambda t)| <= 1e-8 |dT(0)|; total energy constant to 1e-11 relative.
G4 (D) max_t |dU(t) - dU(0) exp(-lambda t)| <= 1e-8 |dU(0)|; total momentum constant to 1e-12 (absolute, in units
   of m_s n_s |U_s(0)|) and total energy constant to 1e-11 relative.
G5 every run: status success.
Any gate failure is recorded and reported, not tuned away.

Run: python studies/b11.py  -> studies/b11/run.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds
from spectrax._initialize_maxwellian import _maxwellian_modes

out_dir = Path(__file__).resolve().parent / "b11"
r2 = np.sqrt(2.0)
RTOL = ATOL = 1e-12


def modes(N, drift, vt, a):
    """1D Hermite coefficients (parent normalization) of a unit-density Maxwellian, basis centre 0, width a."""
    g = np.asarray(_maxwellian_modes(np.asarray(drift / a), np.asarray(1 - 2 * (vt / a) ** 2), N))
    return g / a


def state(model, species):
    """species: list of (cx, cy, cz) 1D coefficient vectors (product distribution)."""
    y = model.zeros()
    blocks = [np.einsum("p,m,n->pmn", cz, cy, cx).ravel() for cx, cy, cz in species]
    Ck = np.zeros(y["Ck"].shape, complex)
    Ck[:, 0, 0, 0] = np.concatenate(blocks)
    return {**y, "Ck": Ck}


def coeffs(out, model, s):
    H = model.Nn * model.Nm * model.Np
    return out["Ck"][:, s * H:(s + 1) * H, 0, 0, 0].real.reshape(-1, model.Np, model.Nm, model.Nn)


def moments_1v(C, a, u, m):
    """Density, mean velocity, temperature, energy of a 1V species from rows 0-2 (per time)."""
    n = a * C[:, 0, 0, 0]
    U = u + a * C[:, 0, 0, 1] / (r2 * C[:, 0, 0, 0])
    T = m * a ** 2 * (0.5 + C[:, 0, 0, 2] / (r2 * C[:, 0, 0, 0]) - (C[:, 0, 0, 1] / (r2 * C[:, 0, 0, 0])) ** 2)
    W = 0.5 * m * n * (U ** 2 + T / m)
    return n, U, T, W


def case_A():
    nu, d, vt, N = 1.0, 0.6, 0.5, 96
    theta = vt ** 2 + d ** 2
    a = np.sqrt(2 * theta)
    model = ds.Model(Nn=N, qs=(0.0,), alpha_s=(a, 1.0, 1.0), collision_model="dougherty", nu_dougherty=nu)
    cx = 0.5 * (modes(N, d, vt, a) + modes(N, -d, vt, a))
    y = state(model, [(cx, np.ones(1), np.ones(1))])
    out = ds.run(model, y, 5.0, n_save=51, rtol=RTOL, atol=ATOL)
    C, t = coeffs(out, model, 0)[:, 0, 0], out["t"]
    ref = {n: C[0, n] * np.exp(-n * nu * t) for n in (4, 6, 8)}
    abs_errs = {n: float(np.max(np.abs(C[:, n] - ref[n])) / abs(C[0, 0])) for n in ref}
    errs = {n: float(np.max(np.abs(C[:, n] / ref[n] - 1)[np.exp(-n * nu * t) >= 1e-6])) for n in ref}
    cons = float(max(np.max(np.abs(C[:, j] - C[0, j])) for j in range(3)) / abs(C[0, 0]))
    dist = np.sqrt((C[:, 3:] ** 2).sum(axis=1))
    passed = out["status"] == "success" and max(abs_errs.values()) <= 1e-10 and max(errs.values()) <= 1e-6 and cons <= 1e-12
    return {"case": "A two-beam", "nu": nu, "d": d, "vt": vt, "Nn": N, "status": out["status"],
            "mode_abs_errors": abs_errs, "mode_rel_errors": errs, "rows_0_2_change": cons, "nonmaxwellian_norm": [float(dist[0]), float(dist[-1])],
            "num_steps": out["num_steps"], "run_time": out["run_time"], "passed": bool(passed)}


def case_B():
    nu, T, N = 0.5, np.array([0.25, 0.16, 0.09]), 12
    theta = T.mean()
    a = np.sqrt(2 * theta)
    model = ds.Model(Nn=N, Nm=N, Np=N, qs=(0.0,), alpha_s=(a, a, a), collision_model="dougherty", nu_dougherty=nu)
    y = state(model, [tuple(modes(N, 0.0, np.sqrt(Ti), a) for Ti in T)])
    out = ds.run(model, y, 4.0, n_save=41, rtol=RTOL, atol=ATOL)
    C, t = coeffs(out, model, 0), out["t"]
    C0 = C[:, 0, 0, 0]
    Ti = np.stack([a ** 2 * (0.5 + C[:, 0, 0, 2] / (r2 * C0)), a ** 2 * (0.5 + C[:, 0, 2, 0] / (r2 * C0)),
                   a ** 2 * (0.5 + C[:, 2, 0, 0] / (r2 * C0))], axis=1)
    th = Ti.mean(axis=1)
    err = float(np.max(np.abs((Ti - theta) - (T - theta)[None] * np.exp(-2 * nu * t)[:, None])) / theta)
    th_change = float(np.max(np.abs(th - th[0])) / theta)
    passed = out["status"] == "success" and err <= 1e-8 and th_change <= 1e-12
    return {"case": "B anisotropy 3V", "nu": nu, "T0": T.tolist(), "N": [N, N, N], "status": out["status"],
            "rel_error": err, "theta_change": th_change, "T_final": Ti[-1].tolist(),
            "num_steps": out["num_steps"], "run_time": out["run_time"], "passed": bool(passed)}


def two_species(mr, U, T, nu_sr, nu_self, t_max, N=64):
    ms = np.array([1.0, mr])
    Tmax = max(T) + 0.5 * mr / (1 + mr) * (U[0] - U[1]) ** 2  # upper bound of either final temperature
    a = np.sqrt(2 * Tmax / ms) * 1.05
    nu = np.array([[nu_self[0], nu_sr], [0.0, nu_self[1]]])
    model = ds.Model(Nn=N, qs=(0.0, 0.0), Omega_cs=(1.0, 1.0 / mr), alpha_s=(a[0], 1, 1, a[1], 1, 1), u_s=(0.0,) * 6,
                     collision_model="dougherty", nu_dougherty=tuple(map(tuple, nu)))
    y = state(model, [(modes(N, U[s], np.sqrt(T[s] / ms[s]), a[s]), np.ones(1), np.ones(1)) for s in range(2)])
    out = ds.run(model, y, t_max, n_save=61, rtol=RTOL, atol=ATOL)
    mom = [moments_1v(coeffs(out, model, s), a[s], 0.0, ms[s]) for s in range(2)]
    return out, ms, mom


def case_C(mr, t_max):
    nu_sr, T = 1.0, (1.0, 0.2)
    out, ms, mom = two_species(mr, (0.0, 0.0), T, nu_sr, (0.5, 0.2), t_max)
    t = out["t"]
    nu_rs = nu_sr * ms[0] / ms[1]
    lam = 2 * (nu_sr * ms[0] + nu_rs * ms[1]) / (ms[0] + ms[1])
    dT = mom[0][2] - mom[1][2]
    err = float(np.max(np.abs(dT - dT[0] * np.exp(-lam * t))) / abs(dT[0]))
    W = mom[0][3] + mom[1][3]
    dW = float(np.max(np.abs(W - W[0])) / W[0])
    passed = out["status"] == "success" and err <= 1e-8 and dW <= 1e-11
    return {"case": f"C T-equilibration m_r/m_s={mr:g}", "mass_ratio": mr, "nu_sr": nu_sr, "lambda": lam,
            "status": out["status"], "rel_error": err, "energy_change": dW,
            "T_final": [float(mom[0][2][-1]), float(mom[1][2][-1])],
            "num_steps": out["num_steps"], "run_time": out["run_time"], "passed": bool(passed)}


def case_D():
    mr, nu_sr = 4.0, 1.0
    U = (0.3, -0.3 / mr)
    out, ms, mom = two_species(mr, U, (0.5, 0.5), nu_sr, (0.5, 0.2), 4.0)
    t = out["t"]
    nu_rs = nu_sr * ms[0] / ms[1]
    lam = (nu_sr + nu_rs) / 2
    dU = mom[0][1] - mom[1][1]
    err = float(np.max(np.abs(dU - dU[0] * np.exp(-lam * t))) / abs(dU[0]))
    P = ms[0] * mom[0][0] * mom[0][1] + ms[1] * mom[1][0] * mom[1][1]
    dP = float(np.max(np.abs(P - P[0])) / (ms[0] * U[0]))
    W = mom[0][3] + mom[1][3]
    dW = float(np.max(np.abs(W - W[0])) / W[0])
    passed = out["status"] == "success" and err <= 1e-8 and dP <= 1e-12 and dW <= 1e-11
    return {"case": "D drift relaxation m_r/m_s=4", "mass_ratio": mr, "nu_sr": nu_sr, "lambda": lam,
            "status": out["status"], "rel_error": err, "momentum_change": dP, "energy_change": dW,
            "T_final": [float(mom[0][2][-1]), float(mom[1][2][-1])],
            "num_steps": out["num_steps"], "run_time": out["run_time"], "passed": bool(passed)}


if __name__ == "__main__":
    t0 = time.time()
    rows = [case_A(), case_B(), case_C(4.0, 3.0), case_C(100.0, 30.0), case_D()]
    for r in rows:
        print(json.dumps(r), flush=True)
    import diffrax
    import jax
    import spectrax
    rec = {"case_id": "B11", "command": "python studies/b11.py " + " ".join(sys.argv[1:]),
           "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "units": "parent normalization; homogeneous, uncharged (qs = 0)",
           "settings": {"solver": "Dopri8", "rtol": RTOL, "atol": ATOL},
           "versions": {"jax": jax.__version__, "diffrax": diffrax.__version__, "spectrax": spectrax.__version__},
           "wall_time": time.time() - t0, "passed": all(r["passed"] for r in rows), "results": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
