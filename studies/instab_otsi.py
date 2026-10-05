"""instab: identify the physical finite-k instability of the uniformly quivering H05 plasma.

Linear, one Fourier mode k, nu = 0, the pump-frame AW Hermite linear model of studies/highk_linear.py (branch highk),
which agrees with the exact Volterra solution there (k5: 5.98e5 vs 6.17e5 at t = 700, Nn-independent).

Experiments (declared before running):
  floquet : CONSTANT-amplitude dipole pump E0 cos(w0 t) (k = 0), species velocities U_s = (q_s/m_s) E0 sin(w0 t)/w0.
            Monodromy matrix over one period -> Floquet exponent gamma = max Re ln(mu)/T0 and the frequency
            Re omega = arg(mu)/T0 (mod w0). Scans: k, E0, w0 (detuning), T_i/T_e, m_i/m_e, fixed ions, Nn.
  nishikawa : the exact kinetic dipole-pump parametric dispersion relation (Silin 1965; Nishikawa 1968), written as
            det[diag(1+chi_i(w_l)) + B diag(chi_e(w_l)) B^T] = 0, w_l = w + l w0, B_lj = J_{l-j}(k r), |l| <= L,
            kinetic chi_s from the plasma dispersion function; L = 1 is Nishikawa's three-wave truncation.
            (A first attempt with a hand-written reduced form and complex Newton was wrong in sign and is replaced.)
  adiabatic : H05's resonant pump has a secularly growing excursion r(t) = (1+eps)E0 t/(2w). Prediction of the H05
            |E_k|^2 amplification as exp(2 int gamma_floquet(k, r(t)) dt) vs the measured highk table.
Rejection criteria:
  - if fixed ions give gamma > 1e-3 at any (k, E0) where mobile ions are unstable, the instability is NOT ion-mediated
    (not OTSI/PDI) -> look for an electron-only mechanism;
  - if Nishikawa's DR differs from Floquet by > 25% in gamma over the weak-pump (b < 0.5) part of the scan, the
    instability is NOT reproduced by standard OTSI/PDI theory;
  - if the adiabatic prediction misses the H05 amplification at t = 700 by more than 1 decade in log10 for k5..k12,
    the H05 growth is not explained by the quasi-static OTSI rates.
Run: python studies/instab_otsi.py  -> studies/instab/otsi.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import wofz, jv

VTE = np.sqrt(1e-3)
LX = 40.0
K1 = 2 * np.pi / LX


def species(mi=1836.0, tau=1.0, fixed_ions=False):
    """(q, m, sigma) per species; tau = T_i/T_e. Electron omega_pe = 1."""
    s = [(-1.0, 1.0, VTE)]
    if not fixed_ions:
        s.append((1.0, mi, VTE * np.sqrt(tau / mi)))
    return s


def floquet(k, E0, w0=1.0, mi=1836.0, tau=1.0, fixed_ions=False, Nn=64, Nni=16):
    sp = species(mi, tau, fixed_ions)
    Ns = len(sp)
    nns = [Nn] + [Nni] * (Ns - 1)
    off = np.concatenate([[0], np.cumsum(nns)])
    D = off[-1]
    A0 = np.zeros((D, D), complex)
    diagU = np.zeros(D)
    for s, (q, m, sig) in enumerate(sp):
        a = np.sqrt(2) * sig
        n = np.arange(nns[s])
        o = off[s]
        for j in range(nns[s] - 1):
            A0[o + j, o + j + 1] += -1j * k * a * np.sqrt((j + 1) / 2)
            A0[o + j + 1, o + j] += -1j * k * a * np.sqrt((j + 1) / 2)
        # field: E = sum_s' q_s' g_s'0 / (ik) ; dg_s1 += (q/m) E sqrt2/a
        for s2, (q2, _, _) in enumerate(sp):
            A0[o + 1, off[s2]] += (q / m) * np.sqrt(2) / a * q2 / (1j * k)
        diagU[o:o + nns[s]] = q / m * E0 / w0  # U_s = diagU * sin(w0 t)

    T0 = 2 * np.pi / w0

    def rhs(t, y):
        Y = y.reshape(D, D)
        return (A0 @ Y - 1j * k * (diagU * np.sin(w0 * t))[:, None] * Y).ravel()

    sol = solve_ivp(rhs, (0, T0), np.eye(D, dtype=complex).ravel(), method="DOP853", rtol=1e-11, atol=1e-13)
    mu = np.linalg.eigvals(sol.y[:, -1].reshape(D, D))
    i = np.argmax(np.abs(mu))
    return float(np.log(np.abs(mu[i])) / T0), float(np.angle(mu[i]) / T0)


def chi(w, k, sig, wp2):
    z = w / (np.sqrt(2) * k * sig)
    Z = 1j * np.sqrt(np.pi) * wofz(z)
    return wp2 / (k * sig) ** 2 * (1 + z * Z)


def silin_matrix(w, k, E0, w0=1.0, mi=1836.0, tau=1.0, L=4):
    """Exact kinetic dipole-pump dispersion matrix (Silin 1965; Nishikawa 1968; Aliev-Silin).
    Lab-frame potentials phi_l = phi(w + l w0), electron frame psi = B^T phi with B_lj = J_{l-j}(b),
    b = k r, r = relative e-i excursion (ions taken unpumped):  M = diag(1+chi_i) + B diag(chi_e) B^T.
    L = 1 is Nishikawa's three-wave (w, w +- w0) truncation."""
    r = (1 + 1 / mi) * E0 / w0 ** 2
    ls = np.arange(-L, L + 1)
    sig_i = VTE * np.sqrt(tau / mi)
    B = jv(ls[:, None] - ls[None, :], k * r)
    ce = np.array([chi(w + l * w0, k, VTE, 1.0) for l in ls])
    ci = np.array([chi(w + l * w0, k, sig_i, 1.0 / mi) for l in ls])
    return np.diag(1 + ci) + B @ np.diag(ce) @ B.T


def nishikawa(k, E0, w0=1.0, mi=1836.0, tau=1.0, L=4):
    """Largest purely growing root w = i gamma of det M = 0 (det is real on the imaginary axis to 1e-12)."""
    from scipy.optimize import brentq

    def d(g):
        return np.linalg.det(silin_matrix(1j * g, k, E0, w0, mi, tau, L)).real

    gs = np.geomspace(1e-5, 0.5, 300)
    v = np.array([d(g) for g in gs])
    idx = np.nonzero(np.sign(v[1:]) != np.sign(v[:-1]))[0]
    if idx.size == 0:
        return (0.0, 0.0)
    i = idx[-1]
    return (float(brentq(d, gs[i], gs[i + 1], xtol=1e-12)), 0.0)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "instab"
    out.mkdir(exist_ok=True)
    res = {}
    tic = time.perf_counter()
    E01 = 0.1 * VTE  # H05 drive amplitude at v_q/v_te = 0.1
    # pump amplitudes expressed by relative excursion velocity v_os = E0/w0, in units of v_te
    vos = [0.3, 1.0, 3.0, 10.0]
    ks = [2, 5, 8, 12, 20, 30]
    # 1. k x pump scan, mobile vs fixed ions, Floquet vs Nishikawa
    rows = []
    for v in vos:
        for j in ks:
            k = j * K1
            E0 = v * VTE
            gm, wm = floquet(k, E0)
            gf, _ = floquet(k, E0, fixed_ions=True)
            gn, wn = nishikawa(k, E0)
            g1, _ = nishikawa(k, E0, L=1)
            row = dict(gamma_nishikawa_L1=g1, vos_vte=v, kj=j, k_lD=k * VTE, b=k * E0, gamma_floquet=gm, omega_floquet=wm,
                       gamma_fixed_ions=gf, gamma_nishikawa=gn, omega_nishikawa=wn)
            rows.append(row)
            print(json.dumps(row), flush=True)
    res["scan"] = rows
    # 2. Nn convergence at two points
    res["Nn"] = [dict(kj=j, vos=v, Nn=N, gamma=floquet(j * K1, v * VTE, Nn=N)[0])
                 for (j, v) in ((5, 3.0), (12, 10.0)) for N in (32, 64, 128)]
    print(json.dumps(res["Nn"]), flush=True)
    # 3. detuning, T_i/T_e, mass ratio at k12, v_os = 3 v_te
    k = 12 * K1
    E0 = 3 * VTE
    res["detune"] = [dict(w0=w0, floquet=floquet(k, E0, w0=w0), nish=nishikawa(k, E0, w0=w0))
                     for w0 in (0.9, 0.97, 1.0, 1.003, 1.01, 1.05, 1.2)]
    res["tau"] = [dict(tau=t, floquet=floquet(k, E0, tau=t), nish=nishikawa(k, E0, tau=t)) for t in (0.1, 1.0, 3.0)]
    res["mass"] = [dict(mi=m, floquet=floquet(k, E0, mi=m), nish=nishikawa(k, E0, mi=m)) for m in (100.0, 400.0, 1836.0)]
    print(json.dumps({x: res[x] for x in ("detune", "tau", "mass")}), flush=True)
    # 4. pump-amplitude scaling at k5 and k12 (exponent of gamma vs E0)
    amps = np.geomspace(0.05, 20, 12)
    res["amp"] = {str(j): [dict(vos=float(v), g=floquet(j * K1, v * VTE)[0], gn=nishikawa(j * K1, v * VTE)[0])
                           for v in amps] for j in (5, 12)}
    # 5. adiabatic prediction of the H05 table: relative excursion velocity amplitude (1+eps) E01 t/(2w) grows
    w = np.sqrt(1 + 1 / 1836)
    tg = np.arange(0, 701, 10.0)
    adi = {}
    for j in (2, 5, 8, 12, 20):
        # equivalent constant pump with the same relative excursion: E0_eff = w0 * v_rel(t), w0 = w
        g = np.array([floquet(j * K1, w * E01 * t / 2, w0=w)[0] if t > 0 else 0.0
                      for t in tg])
        lnA = 2 * np.concatenate([[0], np.cumsum(0.5 * (np.maximum(g[1:], 0) + np.maximum(g[:-1], 0)) * 10.0)])
        adi[str(j)] = dict(t=tg.tolist(), gamma=g.tolist(), log10_amp_700=float(lnA[-1] / np.log(10)))
        print(j, adi[str(j)]["log10_amp_700"], flush=True)
    res["adiabatic_h05"] = adi
    res["wall_s"] = time.perf_counter() - tic
    res["command"] = "python studies/instab_otsi.py"
    (out / "otsi.json").write_text(json.dumps(res, indent=1) + "\n")
