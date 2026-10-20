"""Runs for studies/antisym (gates in antisym_gates.py docstring).  python antisym_run.py PART [PART ...]"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import antisym_ops as ops  # noqa: E402

VARS = {"AW_c0": ("AW", 0.0), "AW_c1": ("AW", 1.0), "SQ": ("SQ", 0.0)}
K1 = ops.k1


def sha():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()


def td_rate(L, T=200.0, dt=10.0, seed=0):
    """asymptotic growth rate of exp(L t) v by sparse exponential steps with renormalisation."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(L.shape[0]) + 1j * rng.standard_normal(L.shape[0])
    v /= np.linalg.norm(v)
    S = sps.csr_matrix(L * dt)
    acc = 0.0
    for _ in range(int(T / dt)):
        v = expm_multiply(S, v)
        nv = np.linalg.norm(v)
        acc += np.log(nv)
        v /= nv
    return acc / T


def frozen():
    out = []
    for (K, Nn, E1) in itertools.product((2, 5, 10), (32, 64, 128), (1e-3, 4e-3)):
        for name, (b, c) in VARS.items():
            L = ops.operator(K, Nn, E1, b, c)
            t0 = time.perf_counter()
            ev = np.linalg.eigvals(L)
            t1 = time.perf_counter()
            r = dict(var=name, K=K, Nn=Nn, E1=E1, dim=L.shape[0], dense_max_re=float(ev.real.max()), dense_s=t1 - t0)
            if name != "AW_c1":
                r["td_rate"] = td_rate(L)
                r["td_s"] = time.perf_counter() - t1
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


def floquet_exp(K, Nn, E1, var, w=1.0, nsub=100):
    b, c = VARS[var]
    L0 = ops.operator(K, Nn, 0.0, b, 0.0)
    L1 = ops.operator(K, Nn, E1, b, 0.0) - L0
    Cl = None
    if c:
        Cl = np.diag(np.tile(ops.closure_diag(Nn, c, E1, ops.A), 2 * K + 1))
    T = 2 * np.pi / w
    h = T / nsub
    M = np.eye(L0.shape[0], dtype=complex)
    for j in range(nsub):
        cj = np.cos(w * (j + 0.5) * h)
        G = L0 + cj * L1 - (abs(cj) * Cl if Cl is not None else 0)
        M = expm(h * G) @ M
    return float(np.log(np.abs(np.linalg.eigvals(M)).max()) / T)


def floquet():
    out = []
    for (K, Nn), E1 in itertools.product(((2, 32), (2, 64), (2, 128), (5, 32), (5, 64), (10, 32)), (8e-3, 2e-2)):
        for var in ("AW_c0", "AW_c1", "SQ"):
            t0 = time.perf_counter()
            r = dict(var=var, K=K, Nn=Nn, E1=E1, floquet=floquet_exp(K, Nn, E1, var), s=time.perf_counter() - t0)
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


def otsi():
    rec = json.loads((HERE.parent / "dai" / "otsi.json").read_text())["rows"]
    vte = np.sqrt(1e-3)
    out = []
    for r in rec:
        if r["kj"] not in (5, 12, 20):
            continue
        k, E0 = r["kj"] * K1, r["vos_vte"] * vte
        t0 = time.perf_counter()
        g, w = ops.sq_otsi_floquet(k, E0, Nn=r["Nn"])
        row = dict(kj=r["kj"], vos_vte=r["vos_vte"], Nn=r["Nn"], gamma_silin=r["gamma_silin"], gamma_aw=r["AW"]["gamma"],
                   aw_rel=r["AW"]["rel"], gamma_sq=g, omega_sq=w, sq_rel=g / r["gamma_silin"] - 1,
                   sq_vs_aw=g / r["AW"]["gamma"] - 1, s=time.perf_counter() - t0)
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def cons():
    rng = np.random.default_rng(1)
    qm, K, M = -1.0, 5, 24
    s = ops.A
    out = []
    kk = K1 * np.arange(K + 1)
    x = np.arange(M) * 40.0 / M
    for Nn, p in itertools.product((32, 64, 128), (1, 2, 4)):
        X, X2 = ops.sq_moment_mats(Nn)
        Dm = np.diag(np.sqrt(np.arange(1, Nn) / 2.0), 1) - np.diag(np.sqrt(np.arange(1, Nn) / 2.0), -1)
        # random Fourier-Hermite state, x-modes 0..K, decaying like 1/(1+k)
        amp = (1.0 + np.arange(Nn)) ** (-float(p))
        C = np.zeros((K + 1, Nn), complex)
        C = (rng.standard_normal((K + 1, Nn)) + 1j * rng.standard_normal((K + 1, Nn))) * amp[None, :] / (1 + np.arange(K + 1))[:, None]
        C[0] = C[0].real + 0j
        C[0, 0] += 3.0
        Ek = np.zeros(K + 1, complex)
        Ek[1:] = 4e-3 * (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.arange(1, K + 1)
        irf = lambda G: np.fft.irfft(G, n=M, axis=0) * M  # noqa: E731
        rf = lambda g: np.fft.rfft(g, axis=0) / M  # noqa: E731
        mask = (np.arange(K + 1) <= K).astype(float)
        Ex = irf(Ek)
        g = irf(C)  # (M, Nn)
        dC = -1j * kk[:, None] * s * (C @ X.T) - (qm / s) * rf(Ex[:, None] * irf(C @ Dm.T))[:K + 1]
        dg = irf(dC)
        dx = 40.0 / M
        nn = s * np.sum(g * g, 1)
        mm = s ** 2 * np.einsum("xn,nm,xm->x", g, X, g)
        dN = np.sum(2 * s * np.einsum("xn,xn->x", g, dg)) * dx
        dP = np.sum(2 * s ** 2 * np.einsum("xn,nm,xm->x", g, X, dg)) * dx
        dK = np.sum(0.5 * 2 * s ** 3 * np.einsum("xn,nm,xm->x", g, X2, dg)) * dx
        exP = qm * np.sum(Ex * nn) * dx
        exK = qm * np.sum(Ex * mm) * dx
        scN = np.sum(nn) * dx * np.sqrt(2 * Nn) * np.abs(qm) * np.abs(Ex).max() / s
        scP = abs(qm) * np.sum(np.abs(Ex) * nn) * dx
        scK = abs(qm) * np.sum(np.abs(Ex) * s ** 2 * np.einsum("xn,nm,xm->x", np.abs(g), np.abs(X), np.abs(g))) * dx
        top = float(np.sum(np.abs(C[:, -2:]) ** 2) / np.sum(np.abs(C) ** 2))
        r = dict(Nn=Nn, p=p, top2_frac=top, dN_rel=abs(dN) / scN, dP_rel=abs(dP - exP) / scP, dK_rel=abs(dK - exK) / scK)
        out.append(r)
        print(json.dumps(r), flush=True)
    return out


def cost():
    """Banded (tridiagonal) kernels for both variants -- the fair implementation (amendment: the first run used dense N x N
    matmuls and a dense moment einsum, which inflated the SQ/AW ratio to 3.2-5.8; recorded in docs)."""
    from time import perf_counter
    out = []
    Nx, s, qm = 32, ops.A, -1.0
    nk = Nx // 2 + 1
    kk = K1 * np.arange(nk)
    mask = (np.arange(nk) <= (Nx - 1) // 3).astype(float)
    rf = lambda g: np.fft.rfft(g, axis=0) / Nx  # noqa: E731
    irf = lambda G: np.fft.irfft(G * Nx, n=Nx, axis=0)  # noqa: E731
    rng = np.random.default_rng(0)

    def shifts(Nn):
        o = np.sqrt(np.arange(1, Nn) / 2.0)
        return o

    def Xmul(C, o):  # (X C)_n = o_n C_{n+1} + o_{n-1} C_{n-1}
        R = np.zeros_like(C)
        R[:, :-1] += o * C[:, 1:]
        R[:, 1:] += o * C[:, :-1]
        return R

    def Dmul(C, o):  # (D C)_n = o_n C_{n+1} - o_{n-1} C_{n-1}
        R = np.zeros_like(C)
        R[:, :-1] += o * C[:, 1:]
        R[:, 1:] -= o * C[:, :-1]
        return R

    for Nn in (64, 128):
        o = shifts(Nn)
        f2 = np.sqrt(2 * np.arange(1, Nn)) / s
        C = (rng.standard_normal((nk, Nn)) + 1j * rng.standard_normal((nk, Nn))) * mask[:, None] * 1e-3
        E = (rng.standard_normal(nk) + 1j * rng.standard_normal(nk)) * mask * 1e-3

        def rhs_aw(C, E):
            Et = irf(E)
            F = np.zeros_like(C)
            F[:, 1:] = f2 * C[:, :-1]
            dC = -1j * kk[:, None] * s * Xmul(C, o) - qm * rf(Et[:, None] * irf(F)) * mask[:, None]
            return dC, -(C[:, 1] * s * np.sqrt(0.5)) * mask

        def rhs_sq(C, E):
            Et = irf(E)
            gx = irf(C)
            dC = -1j * kk[:, None] * s * Xmul(C, o) - (qm / s) * rf(Et[:, None] * irf(Dmul(C, o))) * mask[:, None]
            j = s ** 2 * 2 * np.sum(gx[:, :-1] * gx[:, 1:] * o, axis=1)
            return dC, -rf(j) * mask

        res = {}
        for nm, f in (("AW", rhs_aw), ("SQ", rhs_sq)):
            f(C, E)
            t = []
            for _ in range(5):
                t0 = perf_counter()
                for _ in range(200):
                    f(C, E)
                t.append((perf_counter() - t0) / 200)
            res[nm] = min(t)
        nz = {}
        for v in ("AW", "SQ"):
            L = sps.csr_matrix(ops.operator(10, Nn, 4e-3, v))
            y = rng.standard_normal(L.shape[0]) + 0j
            t0 = perf_counter()
            for _ in range(500):
                L @ y
            nz[v] = dict(nnz=int(L.nnz), matvec_s=(perf_counter() - t0) / 500)
        r = dict(Nn=Nn, Nx=Nx, rhs_s=res, rhs_ratio=res["SQ"] / res["AW"], frozen_K10=nz,
                 nnz_ratio=nz["SQ"]["nnz"] / nz["AW"]["nnz"])
        out.append(r)
        print(json.dumps(r), flush=True)
    return out


def cond():
    out = []
    rng = np.random.default_rng(3)
    for K, Nn, E1 in ((5, 64, 4e-3), (5, 64, 1e-2), (10, 128, 4e-3)):
        for name, (b, c) in VARS.items():
            if name == "AW_c1":
                continue
            L = ops.operator(K, Nn, E1, b, c)
            nL = np.linalg.norm(L, 2)
            norm_def = np.linalg.norm(L @ L.conj().T - L.conj().T @ L, 2) / nL ** 2
            w, V = np.linalg.eig(L)
            sv = np.linalg.svd(V, compute_uv=False)
            P = rng.standard_normal(L.shape) + 1j * rng.standard_normal(L.shape)
            P *= 1e-13 * nL / np.linalg.norm(P, 2)
            w2 = np.linalg.eigvals(L + P)
            r = dict(var=name, K=K, Nn=Nn, E1=E1, normality_defect=float(norm_def), cond_eigvec=float(sv[0] / sv[-1]),
                     dense_max_re=float(w.real.max()), dense_max_re_perturbed_1e13=float(w2.real.max()),
                     td_rate=td_rate(L), L2norm=float(nL))
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


if __name__ == "__main__":
    for part in sys.argv[1:]:
        tic = time.perf_counter()
        res = globals()[part]()
        (HERE / f"{part}.json").write_text(json.dumps(dict(sha=sha(), command=f"python studies/antisym/antisym_run.py {part}",
                                                             wall_s=time.perf_counter() - tic, results=res), indent=0) + "\n")
