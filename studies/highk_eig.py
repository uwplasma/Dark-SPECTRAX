"""highk (d): spectrum of the truncated Fourier-Hermite operator  df/dt = -v df/dx + E(x) df/dv  (electrons, q/m=-1)
in a FROZEN sinusoidal field E(x) = E1 cos(k1 x), on the kept modes |k| <= K, Hermite order Nn, basis width a,
centre u (pump frame).  AW (SPECTRAX: f = sum g_n H_n(xi) e^{-xi^2}) versus SW (f = sum c_n phi_n(xi), Hermite
functions).  The exact operator is anti-self-adjoint in L^2 (Liouville), so every growth rate max Re(lambda) > 0 of
the truncated system is a discretization artefact.
Rejection criterion (declared): if AW max Re(lambda) at the measured E_k1 amplitude (|E_k1| ~ 4e-3 at t ~ 200-450,
v_q = 0.1) is < 0.01 omega_pe, the AW truncation is NOT the cause of the observed top-mode explosion (growth ~1).
Run: python studies/highk_eig.py -> studies/highk/eig.json
"""
import itertools
import json
from pathlib import Path

import numpy as np

k1 = 2 * np.pi / 40.0


def operator(K, Nn, E1, a, u, basis):
    ks = np.arange(-K, K + 1)
    nk = ks.size
    n = np.arange(Nn)
    L = np.zeros((nk * Nn, nk * Nn), complex)
    idx = lambda ik, m: ik * Nn + m  # noqa: E731
    for ik, kk in enumerate(ks):
        k = kk * k1
        for m in range(Nn):  # streaming  -ik (u + a xi)
            L[idx(ik, m), idx(ik, m)] += -1j * k * u
            if m + 1 < Nn:
                L[idx(ik, m), idx(ik, m + 1)] += -1j * k * a * np.sqrt((m + 1) / 2)
            if m >= 1:
                L[idx(ik, m), idx(ik, m - 1)] += -1j * k * a * np.sqrt(m / 2)
        for dk in (-1, 1):  # force (q/m) E df/dv = -E df/dv,  E_{+-1} = E1/2
            jk = ik - dk
            if not 0 <= jk < nk:
                continue
            for m in range(Nn):
                if basis == "AW":   # d/dv [H_m e^{-xi^2}] = -(1/a) sqrt(2(m+1)) H_{m+1} e^{-xi^2} (normalized)
                    if m >= 1:
                        L[idx(ik, m), idx(jk, m - 1)] += -(E1 / 2) * (-np.sqrt(2 * m) / a)
                else:               # d/dxi phi_m = sqrt(m/2) phi_{m-1} - sqrt((m+1)/2) phi_{m+1}
                    if m >= 1:
                        L[idx(ik, m), idx(jk, m - 1)] += -(E1 / 2) * (-np.sqrt(m / 2) / a)
                    if m + 1 < Nn:
                        L[idx(ik, m), idx(jk, m + 1)] += -(E1 / 2) * (np.sqrt((m + 1) / 2) / a)
    return L


if __name__ == "__main__":
    a = np.sqrt(2e-3)
    res = []
    for basis, K, Nn, E1, u in itertools.product(("AW", "SW"), (2, 5, 7, 10), (32, 64, 128), (1e-4, 1e-3, 4e-3, 1e-2), (0.0, 0.35)):
        lam = np.linalg.eigvals(operator(K, Nn, E1, a, u, basis))
        res.append({"basis": basis, "K": K, "Nn": Nn, "E1": E1, "u": u, "max_re": float(lam.real.max())})
        print(json.dumps(res[-1]), flush=True)
    out = Path(__file__).resolve().parent / "highk"
    (out / "eig.json").write_text(json.dumps(res, indent=0) + "\n")
