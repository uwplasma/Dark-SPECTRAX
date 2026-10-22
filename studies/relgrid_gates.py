"""relgrid gates (H04). DECLARED BEFORE ANY PRODUCTION RUN; the thresholds below are not amended after results are seen. A failed gate is
recorded as a failure and any follow-up is declared in a new commit before it is run.

Development note (disclosed): the solver was exercised during development on the homogeneous case and on the nxconv record at T = 100 to
choose the mid-step scheme (`--mid taylor`, explicit Taylor current, 4x lower error than the nxconv fixed-point iteration against the exact
homogeneous ODE at the same dt) and to calibrate the Richardson tolerance of gate B; no production run and none of the gates below had been run.

Common setup unless stated: c = omega_pe = n_e = m_e = 1, m_i/m_e = 1836, u_te^2 = T_e/m_e c^2 = 1e-3 = T_i/m_e c^2, u = p/(m_s c),
omega = sqrt(1 + 1/m_i), drive E0 cos(omega t) with E0 = v_q u_te, deterministic nxconv seeds, osc frame, masked x (2/3), Nu = 4096 electrons.

A1  nonrelativistic limit: `--nonrel --mid iter` (v = u, the nxconv scheme) at v_q = 0.03, Nx 32, Nu 4096, umax 32, dt 0.02, T = 100 reproduces
    the stored nxconv record (studies/nxconv/grid/g_vq0.03_Nx32_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz): max over t <= 100 of
    |difference| / running max <= 1e-9 for dK_e, W_ext, U_k.
A1b `--nonrel --mid taylor` (same, new mid-step current): informational, difference reported, gated only at <= 5e-3 (it differs from the
    record by the O(dt^2) error of the nxconv iteration).
A2  frame generalisation: relativistic lab frame (umax 32 u_te, Nu 8192, Nx 16, T = 200, v_q = 0.03) equals the relativistic osc frame (same
    grid) to <= 1e-6 of the running max in dK_e, W_ext, U_k, Ebar.
B   homogeneous driven oscillator, no seeds (Nx 4, Nu 1024, T = 200, v_q = 0.03, osc frame; lab frame Nu 2048 umax 32):
    B0  the quadrature ODE with sig -> 1e-9 equals the independent cold-fluid ODE with gamma to <= 1e-9 (relative max) in Ebar, u_e, W;
    B1  second-order convergence: error of Ebar vs the exact kinetic ODE at dt = 0.01 over dt = 0.005 in [3.5, 4.5];
    B2  Richardson (dt 0.01, 0.005) vs the exact kinetic ODE: Ebar, U_e relative max error <= 1e-6; W_ext(T) relative error <= 1e-6; the
        pump phase of Ebar (windowed demodulation on [150, 200], studies/relgrid_ref.demod_phase) differs from the ODE by <= 1e-6 rad;
    B3  same as B2 for the lab frame.
C   relativistic Langmuir dispersion: undriven, Juttner electrons and ions (theta = T/m c^2), Nx 16, Nu 4096, umax 16 u_te, lab frame, dt 0.02,
    density seed 1e-4, k lambda_D (lambda_D = u_te) = 0.3 and 0.4, fit A exp(gamma t) cos(omega_r t + phi) to Re E_1 on [20, 200]:
    |omega_r - Re root| / Re root <= 2e-4 and |gamma - Im root| / |Im root| <= 0.02, with the root of the relativistic dielectric function
    (studies/relgrid_ref.eps_langmuir) and, in addition, the grid frequency closer to the relativistic root than to the nonrelativistic
    Maxwellian root at both k.
D   energy ledger (all production and gate runs of the grid, relativistic kinetic energy (gamma - 1) m c^2): max |K_e + K_i + U_k + U_0 -
    initial - W_ext| / max W_ext <= 1e-4; |mass drift| <= 1e-10; edge value of f_e <= 1e-6 of its maximum; plus (homogeneous B) ledger defect
    scales as dt^2 (ratio dt 0.01 / 0.005 in [3, 5]).
E   refinement pairs (v_q = 0.03 relativistic det, T = 1000; base Nx 32, Nu 4096, umax 32, dt 0.02): Nx 64; Nu 8192 (same umax); dt 0.01;
    umax 48 with Nu 6144. Window = last time two runs agree within 10 percent of the running max of the observable, for the observables dK_e,
    dE1D_e (HHS energy), W_ext, U_k; the certified window of every production claim is min over the pairs and observables (nxconv rule,
    studies/nxconv_analysis.py). No threshold: windows are reported; claims are restricted to them.

Run: python studies/relgrid_gates.py a|b|c|e   (A and B and C in a few minutes; E reads the production queue records)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relgrid as R  # noqa: E402
import relgrid_ref as X  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTD = HERE / "relgrid"
OUTD.mkdir(exist_ok=True)


def sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()


def runmax(a, b):
    return float(np.max(np.abs(a - b) / np.maximum.accumulate(np.abs(b) + 1e-300)))


def gate_a(rec):
    d = np.load(HERE / "nxconv/grid/g_vq0.03_Nx32_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz")
    out = {}
    for mid, key in (("iter", "A1"), ("taylor", "A1b")):
        s = R.Sim(vq=0.03, Nx=32, Nu=4096, dt=0.02, umax=32, frame="osc", xshift="mask23", nonrel=True, mid=mid, noise="det")
        s.run(100.0, 0.5)
        r = np.array(s.rec)
        n = len(r)
        dev = {nm: runmax(r[:, c], d[nm][:n]) for nm, c in (("dK_e", 1), ("W_ext", 3), ("U_k", 4))}
        thr = 1e-9 if mid == "iter" else 5e-3
        out[key] = {"dev": dev, "threshold": thr, "pass": bool(max(dev.values()) <= thr), "defect_over_W": float(np.abs(r[:, 6]).max() / r[:, 3].max())}
    # A2 lab vs osc
    res = {}
    for fr in ("lab", "osc"):
        s = R.Sim(vq=0.03, Nx=16, Nu=8192, dt=0.02, umax=32, frame=fr, xshift="mask23", nonrel=False, mid="taylor", noise="det")
        s.run(200.0, 0.5)
        res[fr] = np.array(s.rec)
    dev = {nm: runmax(res["lab"][:, c], res["osc"][:, c]) for nm, c in (("dK_e", 1), ("W_ext", 3), ("U_k", 4), ("Ebar", 8))}
    out["A2"] = {"dev": dev, "threshold": 1e-6, "pass": bool(max(dev.values()) <= 1e-6),
                 "defect_over_W": {fr: float(np.abs(res[fr][:, 6]).max() / res[fr][:, 3].max()) for fr in res}}
    rec["A"] = out


def gate_b(rec):
    T, vq, w = 200.0, 0.03, X.OMEGA
    sol = X.homogeneous(vq, T)
    out = {}
    # B0 cold limit
    a, c = X.homogeneous(vq, T, sig_e=1e-9, sig_i=1e-9), X.cold_fluid(vq, T)
    tt = np.linspace(0, T, 401)
    d = {nm: float(np.abs(a.sol(tt)[i] - c.sol(tt)[i]).max() / np.abs(c.sol(tt)[i]).max()) for nm, i in (("u_e", 0), ("Ebar", 2), ("W", 3))}
    out["B0"] = {"dev": d, "pass": bool(max(d.values()) <= 1e-9)}
    for key, fr, nu in (("B2", "osc", 1024), ("B3", "lab", 2048)):
        res = {}
        for dt in (0.01, 0.005):
            s = R.Sim(vq=vq, Nx=4, Nu=nu, dt=dt, umax=32, frame=fr, noise="none", mid="taylor")
            s.run(T, 1.0)
            res[dt] = np.array(s.rec)
        t = res[0.01][:, 0]
        ref = sol.sol(t)
        # lab frame: electron mean momentum is not U_e (grid is fixed), use Ebar and W only
        e1 = np.abs(res[0.01][:, 8] - ref[2]).max() / np.abs(ref[2]).max()
        e2 = np.abs(res[0.005][:, 8] - ref[2]).max() / np.abs(ref[2]).max()
        ex = lambda c: (4 * res[0.005][:, c] - res[0.01][:, c]) / 3  # noqa: E731
        errE = float(np.abs(ex(8) - ref[2]).max() / np.abs(ref[2]).max())
        errW = float(abs(ex(3)[-1] - ref[3][-1]) / abs(ref[3][-1]))
        errU = float(np.abs(ex(10) - ref[0]).max() / np.abs(ref[0]).max()) if fr == "osc" else None
        ph_g = X.demod_phase(t, ex(8), w, 150.0, 200.0)[0]
        ph_r = X.demod_phase(t, ref[2], w, 150.0, 200.0)[0]
        dph = float(abs(np.angle(np.exp(1j * (ph_g - ph_r)))))
        dl = {dt: float(np.abs(res[dt][:, 6]).max()) for dt in res}
        out[key] = {"frame": fr, "dt2_ratio_Ebar": float(e1 / e2), "err_E_richardson": errE, "err_W_richardson": errW,
                    "err_Ue_richardson": errU, "phase_err_rad": dph, "phase_grid": ph_g, "phase_ode": ph_r,
                    "ledger_ratio": float(dl[0.01] / dl[0.005]),
                    "pass": bool(3.5 <= e1 / e2 <= 4.5 and errE <= 1e-6 and errW <= 1e-6 and (errU is None or errU <= 1e-6) and dph <= 1e-6)}
        out[key]["pass_ledger_ratio"] = bool(3 <= out[key]["ledger_ratio"] <= 5)
    rec["B"] = out


def fit_damped(t, y, w0, g0):
    f = lambda t, A, g, w, p: A * np.exp(g * t) * np.cos(w * t + p)  # noqa: E731
    best = None
    for p0 in (0.0, 1.5, 3.0, -1.5):
        try:
            popt, _ = curve_fit(f, t, y, p0=[y[0] if abs(y[0]) > 0 else 1e-4, g0, w0, p0], maxfev=20000, xtol=1e-14, ftol=1e-14)
        except RuntimeError:
            continue
        r = np.abs(f(t, *popt) - y).max()
        if best is None or r < best[0]:
            best = (r, popt)
    return best[1]


def gate_c(rec):
    out = {}
    guess = None
    for kl in (0.3, 0.4):
        k = kl / X.VTE
        rr = X.langmuir_root(k, rel=True, guess=guess)
        rn = X.langmuir_root(k, rel=False, guess=guess)
        guess = rr
        s = R.Sim(vq=0.0, Nx=16, Nu=4096, dt=0.02, umax=16, frame="lab", xshift="mask23", init="juttner", landau=kl, amp=1e-4)
        s.run(200.0, 0.5)
        r = np.array(s.rec)
        t = r[:, 0]
        m = t >= 20
        A, g, w, p = fit_damped(t[m], r[m, 19], rr.real, rr.imag)
        out[f"k{kl}"] = {"root_rel": [rr.real, rr.imag], "root_nonrel": [rn.real, rn.imag], "grid": [abs(w), g],
                         "err_omega": float(abs(abs(w) - rr.real) / rr.real), "err_gamma": float(abs(g - rr.imag) / abs(rr.imag)),
                         "closer_to_rel": bool(abs(abs(w) - rr.real) < abs(abs(w) - rn.real)),
                         "defect_over_Uk": float(np.abs(r[:, 6]).max() / max(r[:, 4].max(), 1e-300))}
        out[f"k{kl}"]["pass"] = bool(out[f"k{kl}"]["err_omega"] <= 2e-4 and out[f"k{kl}"]["err_gamma"] <= 0.02 and out[f"k{kl}"]["closer_to_rel"])
    rec["C"] = out


if __name__ == "__main__":
    which = sys.argv[1]
    f = OUTD / f"gates_{which}.json"
    rec = {"repository_commit": sha()}
    {"a": gate_a, "b": gate_b, "c": gate_c}[which](rec)
    f.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
