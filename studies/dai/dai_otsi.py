"""dai (gate 2): linear OTSI benchmark of studies/instab_otsi.py with Dai's Galerkin method.

Same model as instab_otsi.floquet (one k, dipole pump E0 cos(w0 t), pump frame per species, mobile ions Nni = 16):
  AW     : instab_otsi's AW Petrov-Galerkin model (control; recomputed)
  DAI_AW : + paper eq. 27 last-column term c_n * (-ik a sqrt(Nn/2)) for every species (streaming is the only operator
           with a spill here: the field acts on F0 = psi_0 -> psi_1 only, and the uniform pump force cancels the
           moving-basis term -U' d_v identically, before or after the L2 projection, because both are d_v with equal
           and opposite coefficients and the Gram matrix is translation invariant)
  DAI_SW : the same Galerkin method in orthonormal Hermite functions phi_n(eta), eta = (v - U)/sigma (= a/sqrt2):
           delta f = sum h_n phi_n, F0 = n0 pi^{1/4} phi_0 / (sigma sqrt(2 pi)), density = sigma sum_n I_n h_n,
           I_n = int phi_n d eta (nonzero for every even n).
Silin/Nishikawa reference: instab_otsi.nishikawa (|l| <= 4), recomputed.
Gate 2 (declared before running): at k = 5, 12, 20 k1 and every v_os/v_te in {0.3, 1, 3, 10} where the Silin rate is
> 0 and the AW control is within 1% of it, for Nn = 32, 64, 128, |gamma/gamma_Silin - 1| <= 1%.  All 22 unstable scan
points of studies/instab/otsi.json are also reported at Nn = 64.
Run: python studies/dai/dai_otsi.py -> studies/dai/otsi.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dai_ops as d  # noqa: E402
from dai_spectra import sha  # noqa: E402
from instab_otsi import K1, VTE, nishikawa, species  # noqa: E402


def hermite_fn_integrals(N):
    """I_n = int phi_n(eta) d eta for orthonormal Hermite functions (stable recurrence on a fine grid)."""
    eta = np.linspace(-60, 60, 120001)
    p0 = np.pi ** -0.25 * np.exp(-eta ** 2 / 2)
    p1 = np.sqrt(2) * eta * p0
    out = [np.trapezoid(p0, eta), np.trapezoid(p1, eta)]
    for n in range(2, N):
        p0, p1 = p1, np.sqrt(2 / n) * eta * p1 - np.sqrt((n - 1) / n) * p0
        out.append(np.trapezoid(p1, eta))
    return np.array(out[:N])


def floquet(k, E0, basis, w0=1.0, mi=1836.0, tau=1.0, Nn=64, Nni=16):
    sp = species(mi, tau, False)
    nns = [Nn, Nni]
    off = np.concatenate([[0], np.cumsum(nns)])
    D = off[-1]
    A0 = np.zeros((D, D), complex)
    diagU = np.zeros(D)
    dens = np.zeros((2, D))  # density functional per species
    for s, (q, m, sig) in enumerate(sp):
        a = np.sqrt(2) * sig
        o, N = off[s], nns[s]
        if basis in ("AW", "DAI_AW"):
            for j in range(N - 1):
                A0[o + j, o + j + 1] += -1j * k * a * np.sqrt((j + 1) / 2)
                A0[o + j + 1, o + j] += -1j * k * a * np.sqrt((j + 1) / 2)
            if basis == "DAI_AW":
                A0[o:o + N, o + N - 1] += d.spill_coeffs(N) * (-1j * k * a * np.sqrt(N / 2))
            dens[s, o] = 1.0
        else:  # DAI_SW: v - U = sig * eta, v phi_n = sig (sqrt((n+1)/2) phi_{n+1} + sqrt(n/2) phi_{n-1})
            for j in range(N - 1):
                A0[o + j, o + j + 1] += -1j * k * sig * np.sqrt((j + 1) / 2)
                A0[o + j + 1, o + j] += -1j * k * sig * np.sqrt((j + 1) / 2)
            dens[s, o:o + N] = sig * hermite_fn_integrals(N)
        diagU[o:o + N] = q / m * E0 / w0
    for s, (q, m, sig) in enumerate(sp):  # field E_k = sum_s' q_s' n_s' / (ik) acting on F0 -> row 1
        a = np.sqrt(2) * sig
        if basis in ("AW", "DAI_AW"):
            src = (q / m) * np.sqrt(2) / a  # instab_otsi convention (g_0 = density)
        else:  # -(q/m) E dF0/dv, F0 = pi^{1/4} phi_0 / (sig sqrt(2 pi)), dphi_0/deta = -phi_1/sqrt2
            src = (q / m) * np.pi ** 0.25 / (np.sqrt(2 * np.pi) * sig ** 2 * np.sqrt(2))
        for s2, (q2, _, _) in enumerate(sp):
            A0[off[s] + 1, :] += src * q2 / (1j * k) * dens[s2]
    T0 = 2 * np.pi / w0

    def rhs(t, y):
        Y = y.reshape(D, D)
        return (A0 @ Y - 1j * k * (diagU * np.sin(w0 * t))[:, None] * Y).ravel()

    sol = solve_ivp(rhs, (0, T0), np.eye(D, dtype=complex).ravel(), method="DOP853", rtol=1e-11, atol=1e-13)
    mu = np.linalg.eigvals(sol.y[:, -1].reshape(D, D))
    i = np.argmax(np.abs(mu))
    return float(np.log(np.abs(mu[i])) / T0), float(np.angle(mu[i]) / T0), int(sol.nfev)


if __name__ == "__main__":
    tic = time.perf_counter()
    rec = json.loads((HERE.parent / "instab" / "otsi.json").read_text())
    rows = []
    for r in rec["scan"]:
        if r["gamma_nishikawa"] <= 0:
            continue
        j, v = r["kj"], r["vos_vte"]
        k, E0 = j * K1, v * VTE
        gs = nishikawa(k, E0)[0]
        Nns = (32, 64, 128) if j in (5, 12, 20) else (64,)
        for Nn in Nns:
            row = dict(kj=j, vos_vte=v, Nn=Nn, gamma_silin=gs)
            for b in ("AW", "DAI_SW", "DAI_AW"):
                t0 = time.perf_counter()
                try:
                    g, w, nfev = floquet(k, E0, b, Nn=Nn)
                    row[b] = dict(gamma=g, omega=w, rel=g / gs - 1, nfev=nfev, s=time.perf_counter() - t0)
                except Exception as e:  # record, do not hide
                    row[b] = dict(error=repr(e)[:200], s=time.perf_counter() - t0)
            rows.append(row)
            print(json.dumps(row), flush=True)
    gate = [r for r in rows if r["kj"] in (5, 12, 20) and abs(r["AW"]["rel"]) <= 0.01]
    verdict = {b: bool(all("rel" in r[b] and abs(r[b]["rel"]) <= 0.01 for r in gate)) for b in ("DAI_SW", "DAI_AW")}
    out = dict(sha=sha(), command="python studies/dai/dai_otsi.py", wall_s=time.perf_counter() - tic, rows=rows,
               gate2_points=len(gate), gate2_pass=verdict)
    print(json.dumps(out["gate2_pass"]), len(gate))
    (HERE / "otsi.json").write_text(json.dumps(out, indent=0) + "\n")
