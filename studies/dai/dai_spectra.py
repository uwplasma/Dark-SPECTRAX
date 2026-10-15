"""dai (a,b): frozen-field and oscillating-field (Floquet) spectra of the truncated 1D1V Fourier-Hermite operator
-v d_x + (q/m) E d_v (electrons, E = E1 cos(k1 x) [cos(w t)]), highk test points: K = 2, 5, 10; Nn = 32, 64, 128;
frozen E1 = 1e-3, 4e-3, 1e-2 (u = 0); Floquet E1 = 8e-3, 2e-2, w = 1.

Variants (studies/dai/dai_ops.py):
  AW_c0, AW_c1, AW_c4 : SPECTRAX AW Petrov-Galerkin, + #66 field-scaled closure c (Floquet: rate ~ E1 |cos w t|)
  SW                  : symmetric Hermite functions, width a (highk_eig control)
  DAI_AW              : Dai's Galerkin method in the paper's efficient form (eq. 27): AW coordinates, tridiagonal + last
                        column, float64 -- what a SPECTRAX implementation of eq. 27 would run
  DAI_SW              : the SAME Galerkin method (same trial space, same L2(dv) inner product) in the orthonormal basis
                        of Hermite functions of width a/sqrt2 (tridiagonal, no extra column)
Declared before running (gate 1): a variant passes iff max Re lambda <= 1e-10 (frozen) and Floquet exponent <= 1e-10
at EVERY point.  Equivalence check: max over eigenvalues of DAI_AW of the distance to the DAI_SW spectrum.
A pilot (K 2/5, Nn 32/64, E1 1e-2) run before this file was committed already showed DAI_AW max Re 2.3e-11 / 3.1e-3:
the declaration of two DAI implementations was made after that pilot and is reported as such.
Floquet: exponential midpoint (2nd-order Magnus, 100 substeps, as studies/highk_floquet.py) except K = 10, Nn = 128,
where a dense 2688^2 expm x 100 costs hours: there DOP853 on the sparse operator (rtol 1e-12) for all variants but
DAI_AW (whose ~1e10 column entries make the explicit integrator impractical; recorded as not run); cross-checked
against Magnus at K = 10, Nn = 64 for AW_c1 and DAI_SW.
Also: L2(dv) norm growth of DAI_AW under exact-exponential stepping to t = 1000 (frozen E1 = 1e-2), norm evaluated
as ||Lambda^{1/2} L11^T g|| with the Cholesky factor of paper Prop. 7.7.
Run: python studies/dai/dai_spectra.py PART   (PART in frozen, floquet, floquet_big, norm) -> studies/dai/spectra_PART.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.sparse as sps
from scipy.integrate import solve_ivp
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai_ops as d  # noqa: E402

A = np.sqrt(2e-3)
VARIANTS = {"AW_c0": ("AW", 0.0), "AW_c1": ("AW", 1.0), "AW_c4": ("AW", 4.0), "SW": ("SW", 0.0),
            "DAI_AW": ("DAI", 0.0), "DAI_SW": ("SWH", 0.0)}
OUT = Path(__file__).resolve().parent


def sha():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=OUT).stdout.strip()


def split(K, Nn, E1, var):
    """L(t) = L0 + cos(wt) L1 - |cos(wt)| Cl (closure uses max|E| = E1 |cos wt|)."""
    b, c = VARIANTS[var]
    L0 = d.operator(K, Nn, 0.0, A, 0.0, b)
    L1 = d.operator(K, Nn, E1, A, 0.0, b) - L0
    Cl = np.diag(np.tile(d.closure_diag(Nn, c, E1, A), 2 * K + 1)) if c else None
    return L0, L1, Cl


def floquet_magnus(K, Nn, E1, var, w=1.0, nsub=100):
    L0, L1, Cl = split(K, Nn, E1, var)
    T = 2 * np.pi / w
    h = T / nsub
    M = np.eye(L0.shape[0], dtype=complex)
    for j in range(nsub):
        cj = np.cos(w * (j + 0.5) * h)
        G = L0 + cj * L1 - (abs(cj) * Cl if Cl is not None else 0)
        M = expm(h * G) @ M
    return float(np.log(np.abs(np.linalg.eigvals(M)).max()) / T)


def floquet_dop(K, Nn, E1, var, w=1.0):
    L0, L1, Cl = split(K, Nn, E1, var)
    S0, S1 = sps.csr_matrix(L0), sps.csr_matrix(L1)
    cl = np.diag(Cl) if Cl is not None else 0.0
    D = L0.shape[0]
    T = 2 * np.pi / w

    def rhs(t, y):
        Y = y.reshape(D, D)
        ct = np.cos(w * t)
        return (S0 @ Y + ct * (S1 @ Y) - (abs(ct) * cl)[:, None] * Y if Cl is not None else
                S0 @ Y + ct * (S1 @ Y)).ravel()

    sol = solve_ivp(rhs, (0, T), np.eye(D, dtype=complex).ravel(), method="DOP853", rtol=1e-12, atol=1e-14)
    return float(np.log(np.abs(np.linalg.eigvals(sol.y[:, -1].reshape(D, D))).max()) / T)


def chol_R(Nn):
    """R with A11 = R^T R: R = Lambda^{1/2} L11^T (paper Prop. 7.7, L lower triangular so (L Lam L^T)_11 = L11 Lam L11^T)."""
    with mp.workdps(int(60 + 0.4 * Nn)):
        R = np.zeros((Nn, Nn))
        for m in range(Nn):
            lam = mp.power(2, -m - mp.mpf(1) / 2)
            for n in range(m, Nn):
                if (n - m) % 2 == 0:  # (L^T)_{m n} = L_{n m} = (-1)^{(n-m)/2} z_{m n}
                    R[m, n] = float(mp.sqrt(lam) * (-1) ** ((n - m) // 2 % 2) * d._z(m, n))
    assert np.allclose(R.T @ R, d.gram_float(Nn), atol=1e-13 * np.abs(d.gram_float(Nn)).max()) if Nn <= 40 else True
    return R


def run(part):
    res = []
    tic = time.perf_counter()
    if part == "frozen":
        for K in (2, 5, 10):
            for Nn in (32, 64, 128):
                for E1 in (1e-3, 4e-3, 1e-2):
                    lam = {}
                    for var, (b, c) in VARIANTS.items():
                        lam[var] = np.linalg.eigvals(d.operator(K, Nn, E1, A, 0.0, b, c))
                        res.append(dict(K=K, Nn=Nn, E1=E1, variant=var, max_re=float(lam[var].real.max())))
                        print(json.dumps(res[-1]), flush=True)
                    dist = max(np.abs(lam["DAI_SW"] - x).min() for x in lam["DAI_AW"])
                    res.append(dict(K=K, Nn=Nn, E1=E1, variant="equiv", max_dist_DAI_AW_to_DAI_SW=float(dist),
                                    max_abs_lambda=float(np.abs(lam["DAI_SW"]).max())))
                    print(json.dumps(res[-1]), flush=True)
    elif part == "floquet":
        for K, Nn in [(2, 32), (2, 64), (2, 128), (5, 32), (5, 64), (5, 128), (10, 32), (10, 64)]:
            for E1 in (8e-3, 2e-2):
                for var in VARIANTS:
                    t0 = time.perf_counter()
                    res.append(dict(K=K, Nn=Nn, E1=E1, variant=var, method="magnus100",
                                    floquet=floquet_magnus(K, Nn, E1, var), s=time.perf_counter() - t0))
                    print(json.dumps(res[-1]), flush=True)
    elif part == "floquet_big":
        for K, Nn, vs in [(10, 64, ("AW_c1", "DAI_SW")), (10, 128, tuple(v for v in VARIANTS if v != "DAI_AW"))]:
            for E1 in (8e-3, 2e-2):
                for var in vs:
                    t0 = time.perf_counter()
                    res.append(dict(K=K, Nn=Nn, E1=E1, variant=var, method="dop853",
                                    floquet=floquet_dop(K, Nn, E1, var), s=time.perf_counter() - t0))
                    print(json.dumps(res[-1]), flush=True)
    elif part == "norm":
        rng = np.random.default_rng(0)
        for K in (2, 5):
            for Nn in (32, 64, 128):
                R = chol_R(Nn)
                Rb = np.kron(np.eye(2 * K + 1), R)
                for var in ("DAI_AW", "DAI_SW"):
                    b, _ = VARIANTS[var]
                    P = expm(d.operator(K, Nn, 1e-2, A, 0.0, b))
                    g = np.zeros(P.shape[0], complex)
                    g[::Nn] = 1 + rng.normal(size=2 * K + 1)  # psi_0 / phi_0 per k (both = the Maxwellian shape)
                    nrm = (lambda x: np.linalg.norm(Rb @ x)) if var == "DAI_AW" else np.linalg.norm
                    n0 = nrm(g)
                    rec = {}
                    for t in range(1, 1001):
                        g = P @ g
                        if t in (10, 100, 300, 1000):
                            rec[str(t)] = float(nrm(g) / n0)
                    res.append(dict(K=K, Nn=Nn, E1=1e-2, variant=var, l2_norm_ratio=rec))
                    print(json.dumps(res[-1]), flush=True)
    out = dict(part=part, sha=sha(), wall_s=time.perf_counter() - tic, command=f"python studies/dai/dai_spectra.py {part}",
               a=A, k1=d.k1, results=res)
    (OUT / f"spectra_{part}.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
