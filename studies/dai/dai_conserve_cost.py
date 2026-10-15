"""dai (gates 3, 4): which of density / momentum / energy Dai's Galerkin method conserves, and its cost vs AW.

Gate 3 (structural, exact): AW keeps rows 0-2 of the moment hierarchy exact (int w psi_{N+1} dv = 0 for deg w <= 2).
Dai replaces the dropped psi_{N+1} by its L2 projection P psi_{N+1} = sum c_n psi_n, so the rate of the moment
int w f dv is wrong by  spill * kappa_w,  kappa_w = int w P psi_{N+1} dv / int w-normalizer, w in {1, xi, xi^2}
(xi = (v-u)/a; moments about the basis centre).  kappa_w = 0 exactly iff c vanishes on the rows w touches; c_n = 0
unless n = N+1 (mod 2).  Reported: kappa_w for Nn = 31..33, 63..65, 127..129, and a dynamic level: frozen field
E1 = 1e-2, K = 5, initial Maxwellian x (1 + 1e-2 cos k1 x) [DAI_SW form], t = 0..200: defect of the global (k = 0)
density, momentum and kinetic-energy rates vs the exact rate of the same f_N, accumulated over t (density relative to the
density; momentum and kinetic energy relative to their exact change, which the external field drives).
Gate 4: nnz of the per-(k,k') Hermite blocks and sparse CSR matvec time (K = 10, Nn = 128, E1 = 1e-2) for AW, DAI_AW
(eq. 27 form) and DAI_SW (tridiagonal Hermite-function form), median of 200 calls.
Run: python studies/dai/dai_conserve_cost.py -> studies/dai/conserve_cost.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai_ops as d  # noqa: E402
from dai_otsi import hermite_fn_integrals  # noqa: E402
from dai_spectra import A, sha  # noqa: E402


def aw_moments(nmax):
    """int xi^p psi_n d xi, p = 0, 1, 2, n < nmax (Gauss-Hermite in t = sqrt2 xi, exact for these degrees)."""
    x, w = hermgauss(200)
    xi = x / np.sqrt(2)
    out = np.zeros((3, nmax))
    for n in range(min(nmax, 3)):
        hn = eval_hermite(n, xi) / np.sqrt(2.0 ** n * factorial(n) * np.sqrt(np.pi))
        for p in range(3):  # int xi^p H e^{-xi^2} = int xi^p H e^{-t^2/2}... use weight e^{-t^2}: e^{-xi^2}=e^{-t^2/2}
            out[p, n] = np.sum(w * np.exp(x ** 2 / 2) * xi ** p * hn) / np.sqrt(2)
    return out  # rows n >= 3 vanish identically


def structural():
    rows = []
    for Nn in (31, 32, 33, 63, 64, 65, 127, 128, 129):
        c = d.spill_coeffs(Nn)
        mom = aw_moments(Nn)
        kap = [float(mom[p] @ c / mom[p, p]) for p in range(3)]  # normalized by int xi^p psi_p
        rows.append(dict(Nn=Nn, top_index_N=Nn - 1, kappa_density=kap[0], kappa_momentum=kap[1], kappa_energy=kap[2],
                         c0=float(c[0]), c1=float(c[1]), c2=float(c[2])))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def dynamic(Nn, K=5, E1=1e-2, T=200.0, dt=0.5):
    """DAI_SW form (eta = v/sigma, sigma = a/sqrt2).  Exact rate of f_N: apply the untruncated operator into V_{N+1}."""
    nk = 2 * K + 1
    L = d.operator(K, Nn, E1, A, 0.0, "SWH")
    Lx = d.operator(K, Nn + 1, E1, A, 0.0, "SWH")  # one more row: exact image of V_N lies in V_{N+1}
    In = hermite_fn_integrals(Nn + 1)
    eta = np.linspace(-60, 60, 120001)
    # moment functionals of phi_n in eta: int phi_n, int eta phi_n, int eta^2 phi_n
    p0 = np.pi ** -0.25 * np.exp(-eta ** 2 / 2)
    p1 = np.sqrt(2) * eta * p0
    Mo = np.zeros((3, Nn + 1))
    Mo[:, 0] = [np.trapezoid(eta ** p * p0, eta) for p in range(3)]
    Mo[:, 1] = [np.trapezoid(eta ** p * p1, eta) for p in range(3)]
    for n in range(2, Nn + 1):
        p0, p1 = p1, np.sqrt(2 / n) * eta * p1 - np.sqrt((n - 1) / n) * p0
        Mo[:, n] = [np.trapezoid(eta ** p * p1, eta) for p in range(3)]
    assert np.allclose(Mo[0], In)
    ik0 = K  # k = 0 block
    g = np.zeros(nk * Nn, complex)
    g[ik0 * Nn] = 1.0
    g[(ik0 + 1) * Nn] = g[(ik0 - 1) * Nn] = 0.5e-2
    P = expm(dt * L)
    m0 = Mo[:, :Nn] @ g[ik0 * Nn:(ik0 + 1) * Nn]
    drift_d = np.zeros(3, complex)
    drift_x = np.zeros(3, complex)
    for _ in range(int(T / dt)):
        gx = np.zeros(nk * (Nn + 1), complex)
        for ik in range(nk):
            gx[ik * (Nn + 1):ik * (Nn + 1) + Nn] = g[ik * Nn:(ik + 1) * Nn]
        r_dai = Mo[:, :Nn] @ (L @ g)[ik0 * Nn:(ik0 + 1) * Nn]
        r_ex = Mo @ (Lx @ gx)[ik0 * (Nn + 1):(ik0 + 1) * (Nn + 1)]
        drift_d += (r_dai - r_ex) * dt
        drift_x += r_ex * dt
        g = P @ g
    return dict(Nn=Nn, K=K, E1=E1, T=T,
                density_drift_over_density=float(abs(drift_d[0] / m0[0])),
                momentum_drift_over_exact_momentum_change=float(abs(drift_d[1]) / abs(drift_x[1])),
                energy_drift_over_exact_energy_change=float(abs(drift_d[2]) / abs(drift_x[2])),
                energy_drift_over_initial_energy=float(abs(drift_d[2] / m0[2])))


def cost(K=10, Nn=128, E1=1e-2):
    out = {}
    x = np.random.default_rng(1).normal(size=(2 * K + 1) * Nn) + 0j
    for var, b in (("AW", "AW"), ("DAI_AW", "DAI"), ("DAI_SW", "SWH")):
        S = sps.csr_matrix(d.operator(K, Nn, E1, A, 0.0, b))
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            S @ x
            ts.append(time.perf_counter() - t0)
        out[var] = dict(nnz=int(S.nnz), matvec_s=float(np.median(ts)),
                        max_abs_entry=float(abs(S).max()), streaming_block_nnz=int(sps.csr_matrix(
                            d.operator(1, Nn, 0.0, A, 0.0, b)[2 * Nn:, 2 * Nn:]).nnz))
        print(var, json.dumps(out[var]), flush=True)
    for v in ("DAI_AW", "DAI_SW"):
        out[v]["nnz_ratio_vs_AW"] = out[v]["nnz"] / out["AW"]["nnz"]
        out[v]["time_ratio_vs_AW"] = out[v]["matvec_s"] / out["AW"]["matvec_s"]
    return out


if __name__ == "__main__":
    tic = time.perf_counter()
    res = dict(structural=structural())
    res["dynamic"] = [dynamic(Nn) for Nn in (32, 33, 64, 65)]
    print(json.dumps(res["dynamic"]), flush=True)
    res["cost"] = cost()
    res.update(sha=sha(), wall_s=time.perf_counter() - tic, command="python studies/dai/dai_conserve_cost.py")
    (HERE / "conserve_cost.json").write_text(json.dumps(res, indent=0) + "\n")
