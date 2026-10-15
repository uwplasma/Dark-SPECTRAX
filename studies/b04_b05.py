"""B04 ion-acoustic damping and B05 Buneman growth, ordinary and dark, against kinetic Maxwell-Proca roots.

Parent units (c = 1, omega_pe = 1, electron mass 1). Mobile ions with Omega_cs = 1/m_i. All roots solve
D_L = (Q - Omega_D^2)(1 + chi) + eta^2 Q chi = 0 with Q = w^2 - k^2 and
chi = sum_s n_s q_s^2 Omega_cs[s] / (k v_s)^2 [1 + zeta_s Z(zeta_s)], zeta_s = (w - k u_s)/(sqrt(2) k v_s)  (wofz).

B05 Buneman: electrons drift at u0 = 0.1 c relative to ions at rest, v_te = 0.1 u0, T_i = T_e. A net current
-n u0 would drive the k = 0 field (dE_0/dt = -J_0). It is cancelled by a third, inert return-current
carrier: charge +1, density 1, mass 1e6 m_e, drifting with the electrons (J_0 = 0 exactly at t = 0); a uniform
background rho_background = -1 restores neutrality. Its susceptibility is 1e-6 of the electrons' and is kept
in the roots. Growth is the log-slope of |E_k| over an explicit window ending at the first time
|E_k| > 1e-6 (linear stage). Cold fluid check: gamma_max = (sqrt3/2)(m_e/2m_i)^(1/3) omega_pe.

B04 ion-acoustic: v_te = 0.1 c, m_i = 1836, T_e/T_i = 1, 5, 20, k lambda_De = 0.1, 0.2, 0.3. Electrons and
ions carry the same density seed (quasi-neutral start, so the Langmuir branch is weakly excited). The
complex frequency is fitted by ds.fit_modes on E_k over an explicit window with the ion-acoustic and
Langmuir pairs as guesses. Electron and ion Hermite recurrence (front n ~ (k v_t t)^2) is controlled by the
parent hypercollision closure nu (declared per run); convergence is shown by Nn refinement.

Run: python studies/b04_b05.py b05|b04 [quick]  -> studies/b04_b05/{b05,b04}.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

import darkspectrax as ds

out_dir = Path(__file__).resolve().parent / "b04_b05"
eta, Omega_D = 0.3, 1.0


def chi(w, k, pops):
    """pops: [(n, q, Omega_c, v_t, u)]."""
    tot = 0.0
    for n, q, om, vt, u in pops:
        z = (w - k * u) / (np.sqrt(2) * k * vt)
        tot = tot + n * q * q * om / (k * vt) ** 2 * (1 + z * 1j * np.sqrt(np.pi) * wofz(z))
    return tot


def D_L(w, k, pops, eta_):
    Q = w ** 2 - k ** 2
    c = chi(w, k, pops)
    return (Q - Omega_D ** 2) * (1 + c) + eta_ ** 2 * Q * c


def solve(k, pops, eta_, guess):
    s = root(lambda x: [D_L(x[0] + 1j * x[1], k, pops, eta_).real, D_L(x[0] + 1j * x[1], k, pops, eta_).imag],
             [guess.real, guess.imag], tol=1e-14)
    w = s.x[0] + 1j * s.x[1]
    return w, abs(D_L(w, k, pops, eta_))


def model_of(pops, k, Nn, Nx, dark, nu=0.0, rho_bg=0.0):
    return ds.Model(Nx=Nx, Nn=Nn, nu=nu, Lx=2 * np.pi / k, qs=tuple(p[1] for p in pops),
                    Omega_cs=tuple(p[2] for p in pops),
                    alpha_s=tuple(np.repeat([np.sqrt(2) * p[3] for p in pops], 3)),
                    u_s=tuple(v for p in pops for v in (p[4], 0.0, 0.0)), rho_background=rho_bg,
                    mode="self_consistent" if dark else "ordinary", eta=eta if dark else 0.0, Omega_D=Omega_D)


def common(out):
    return {"status": out["status"], "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
            "max_gauss": [float(v) for v in out["gauss"].max(axis=0)], "steps": int(out["num_steps"]),
            "compile_time": float(out["compile_time"]), "run_time": float(out["run_time"])}


def b05(quick):
    u0, seed, thresh = 0.1, 1e-10, 1e-6
    rows = []
    for mi in (100.0, 1836.0):
        vte = 0.1 * u0
        vti = vte / np.sqrt(mi)
        pops = [(1.0, -1.0, 1.0, vte, u0), (1.0, 1.0, 1.0 / mi, vti, 0.0), (1.0, 1.0, 1e-6, vte, u0)]
        g_fluid = np.sqrt(3) / 2 * (1 / (2 * mi)) ** (1 / 3)
        for ku in ((1.0,) if quick else (0.9, 1.0)):
            k = ku / u0
            best = None
            for wr in np.linspace(0.0, 0.3, 7):
                for wi in (0.03, 0.08, 0.15):
                    w, r = solve(k, pops, 0.0, wr + 1j * wi)
                    if r < 1e-10 and (best is None or w.imag > best.imag + 1e-9):
                        best = w
            w0, r0 = solve(k, pops, 0.0, best)
            wd, rd = solve(k, pops, eta, w0)
            tmax = float(np.ceil((np.log(thresh / seed) + 4) / w0.imag / 10) * 10 + 30)
            for dark in (False, True):
                for Nn, Nx in (((16, 5),) if quick else ((16, 5), (32, 5), (16, 8))):
                    model = model_of(pops, k, Nn, Nx, dark, rho_bg=-1.0)
                    pert = [(0, (1, 0, 0), seed), (1, (1, 0, 0), seed)]
                    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0, 1.0, 1.0], pert))
                    out = ds.run(model, y, tmax, n_save=int(10 * tmax) + 1, rtol=1e-10, atol=1e-20)
                    t, A = out["t"], np.abs(out["Fk"][:, 0, 0, 1, 0])
                    big = np.nonzero(A > thresh)[0]
                    t_end = float(t[big[0]]) if len(big) else tmax
                    L = 3.0 / w0.imag
                    fits = []
                    for a, b in ((t_end - L, t_end), (t_end - 0.75 * L, t_end), (t_end - L, t_end - 0.25 * L)):
                        m = (t >= a) & (t <= b)
                        fits.append(np.polyfit(t[m], np.log(A[m]), 1)[0])
                    ref = wd if dark else w0
                    rows.append({"case": f"B05_mi{mi:g}_ku{ku}", "model": "dark" if dark else "ordinary",
                                 "mass_ratio": mi, "k_u0": ku, "Nn": Nn, "Nx": Nx, "t_max": tmax,
                                 "window": [t_end - L, t_end], "growth_fit": fits[0],
                                 "growth_window_spread": float(np.ptp(fits)), "root": [ref.real, ref.imag],
                                 "root_residual": rd if dark else r0,
                                 "growth_rel_error": (fits[0] - ref.imag) / ref.imag,
                                 "fluid_gamma_max": g_fluid, "root_over_fluid": ref.imag / g_fluid,
                                 "max_abs_E0": float(np.abs(out["Fk"][:, 0, 0, 0, 0]).max()), **common(out)})
                    print(json.dumps(rows[-1]), flush=True)
    return {"u0": u0, "v_te": 0.1 * u0, "T_i/T_e": 1.0, "seed": seed, "threshold": thresh,
            "return_current_carrier": {"q": 1.0, "n": 1.0, "mass": 1e6, "drift": u0, "rho_background": -1.0},
            "rtol": 1e-10, "atol": 1e-20, "nu": 0.0}, rows


def b04(quick):
    vte, mi, seed = 0.1, 1836.0, 1e-6
    rows = []
    nus = 1.0
    for TeTi in ((5.0,) if quick else (1.0, 5.0, 20.0)):
        vti = vte / np.sqrt(mi * TeTi)
        pops = [(1.0, -1.0, 1.0, vte, 0.0), (1.0, 1.0, 1.0 / mi, vti, 0.0)]
        for kl in ((0.2,) if quick else (0.1, 0.2, 0.3)):
            k = kl / vte
            cs = vte * np.sqrt((1 + 3 / TeTi) / mi)
            g0 = k * cs / np.sqrt(1 + kl ** 2)
            w0, r0 = solve(k, pops, 0.0, g0 * (1 - 0.1j))
            wd, rd = solve(k, pops, eta, w0)
            wL, _ = solve(k, pops, 0.0, np.sqrt(1 + 3 * kl ** 2) + 0j)
            period = 2 * np.pi / w0.real
            # window: from 0.5 period to the earlier of 4 periods and the time |E| decays by 1e-4
            t_end = min(4 * period, np.log(1e4) / max(-w0.imag, 1e-12))
            win = (0.5 * period, float(t_end))
            for dark in (False, True):
                ref = wd if dark else w0
                for Nn, Nx in (((32, 5),) if quick else ((32, 5), (64, 5), (128, 5), (32, 8))):
                    model = model_of(pops, k, Nn, Nx, dark, nu=nus)
                    pert = [(0, (1, 0, 0), seed), (1, (1, 0, 0), seed)]
                    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0, 1.0], pert))
                    tmax = win[1]
                    out = ds.run(model, y, tmax, n_save=801, rtol=1e-10, atol=1e-20)
                    t, z = out["t"], out["Fk"][:, 0, 0, 1, 0]
                    m = t >= win[0]
                    if not np.all(np.isfinite(z)):
                        rows.append({"case": f"B04_TeTi{TeTi:g}_k{kl}", "model": "dark" if dark else "ordinary",
                                     "Te_over_Ti": TeTi, "k_lambda_De": kl, "Nn": Nn, "Nx": Nx, "nu": nus,
                                     "fit_window": list(win), "omega_root": [ref.real, ref.imag],
                                     "nonfinite_E_from_t": float(t[np.argmax(~np.isfinite(z))]), **common(out)})
                        print(json.dumps(rows[-1]), flush=True)
                        continue
                    guesses = [ref, -np.conj(ref), wL, -np.conj(wL)]
                    w, res = ds.fit_modes(t[m], z[m], guesses)
                    i = int(np.argmin(np.abs(w - ref)))
                    rows.append({"case": f"B04_TeTi{TeTi:g}_k{kl}", "model": "dark" if dark else "ordinary",
                                 "Te_over_Ti": TeTi, "k_lambda_De": kl, "Nn": Nn, "Nx": Nx, "nu": nus,
                                 "fit_window": list(win), "omega_fit": [w[i].real, w[i].imag],
                                 "omega_root": [ref.real, ref.imag], "root_residual": rd if dark else r0,
                                 "rel_error": abs(w[i] - ref) / abs(ref),
                                 "damping_rel_error": (w[i].imag - ref.imag) / abs(ref.imag),
                                 "fit_residual": res, "gamma_over_omega_root": ref.imag / ref.real,
                                 **common(out)})
                    print(json.dumps(rows[-1]), flush=True)
    return {"v_te": vte, "mass_ratio": mi, "seed": seed, "nu": nus, "rtol": 1e-10, "atol": 1e-20}, rows


if __name__ == "__main__":
    case, quick = sys.argv[1], "quick" in sys.argv[2:]
    t0 = time.time()
    settings, rows = {"b05": b05, "b04": b04}[case](quick)
    import diffrax
    import jax
    import scipy
    import spectrax
    rec = {"case_id": case.upper(), "command": "python studies/b04_b05.py " + " ".join(sys.argv[1:]),
           "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "units": "parent normalization: c = 1, omega_pe = 1, Omega_cs[0] = 1",
           "settings": {**settings, "eta": eta, "Omega_D": Omega_D, "solver": "Dopri8"},
           "versions": {"jax": jax.__version__, "diffrax": diffrax.__version__, "scipy": scipy.__version__,
                        "spectrax": spectrax.__version__},
           "wall_time": time.time() - t0, "results": rows}
    if not quick:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{case}.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
