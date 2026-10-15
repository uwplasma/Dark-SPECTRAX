"""Dai (arXiv 2608.09827) stable asymmetric-Hermite Galerkin operator, linear 1D1V prototype (no SPECTRAX change).

Trial AND test functions are the AW functions psi_n(xi) = H_n(xi) exp(-xi^2) / sqrt(2^n n! sqrt(pi)), xi = (v-u)/a,
with the UNWEIGHTED L2(dv) inner product.  Gram A_nm = (psi_n, psi_m)_{L2} (paper Lemma 7.1).  Any operator O with
O V_N subset V_{N+1} (streaming v d_x, force E d_v, moving-basis -u' d_v) is discretized as P O, P = L2 projection on
V_N = span{psi_0..psi_N}.  O psi_N has one coefficient outside V_N (on psi_{N+1}); P psi_{N+1} = sum_n c_n psi_n with
c = A11^{-1} A[0:N+1, N+1] = Z11 L21^T (paper Lemma 7.8, eq. 27).  Hence Dai = AW Petrov-Galerkin (tridiagonal cut-off)
plus one dense LAST column per (k, k') block: entry spill * c_n.  c_n = 0 unless n = N+1 (mod 2).

Equivalence (derived here, checked numerically in dai_spectra.py): span{H_n(xi) e^{-xi^2}, n<=N} = span{Hermite
functions phi_n(eta) = H_n(eta) e^{-eta^2/2}, n<=N} with eta = sqrt2 xi.  A Galerkin method is fixed by its trial space
and inner product, not by the basis, so Dai's scheme is the symmetric-Hermite (SW) Galerkin scheme with width a/sqrt2,
written in AW coordinates (triangular change of basis, condition ~ 2^N).
"""
import sys
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from highk_eig import operator as hk_operator, k1  # noqa: E402,F401


def _gram_mp(n, m):
    if (n + m) % 2:
        return mp.mpf(0)
    s = (n + m) // 2
    return (-1) ** ((n - m) // 2 % 2) * mp.power(2, -(n + m) - mp.mpf(1) / 2) * mp.factorial(n + m) / (
        mp.factorial(s) * mp.sqrt(mp.factorial(n) * mp.factorial(m)))


def _z(m, n):
    if n < m or (n - m) % 2:
        return mp.mpf(0)
    r = (n - m) // 2
    return mp.sqrt(mp.factorial(n) / mp.factorial(m)) / (mp.power(4, r) * mp.factorial(r))


@lru_cache(None)
def spill_coeffs(Nn, check=True):
    """c_n (n < Nn): L2(dv) projection of psi_Nn onto span{psi_0..psi_{Nn-1}} (paper eq. 26-27).
    Computed two ways in high precision: direct Gram solve and the Z L^T closed form; returns float64."""
    with mp.workdps(int(60 + 0.4 * Nn)):
        N1 = Nn  # index N+1 of the first excluded function
        c_z = [sum(_z(n, j) * (-1) ** ((N1 - j) // 2 % 2) * _z(j, N1) for j in range(n, Nn) if (N1 - j) % 2 == 0)
               for n in range(Nn)]
        if check:
            A11 = mp.matrix(Nn, Nn)
            rhs = mp.matrix(Nn, 1)
            for i in range(Nn):
                rhs[i] = _gram_mp(i, N1)
                for j in range(Nn):
                    A11[i, j] = _gram_mp(i, j)
            c_d = mp.lu_solve(A11, rhs)
            err = max(abs(c_d[i] - c_z[i]) for i in range(Nn)) / max(abs(x) for x in c_z)
            assert err < mp.mpf(10) ** (-20), float(err)
        return np.array([float(x) for x in c_z])


def gram_float(Nn):
    return np.array([[float(_gram_mp(i, j)) for j in range(Nn)] for i in range(Nn)])


def closure_diag(Nn, c, Emax, a):
    """SPECTRAX #66 field-scaled order-2 hypercollision rate on s(n), |q/m| = 1 (study units of highk_eig)."""
    n = np.arange(Nn, dtype=float)
    s = n * (n - 1) * (n - 2) / ((Nn - 1) * (Nn - 2) * (Nn - 3))
    return c * np.sqrt(2.0 * Nn) * Emax / a * s


def operator(K, Nn, E1, a, u, basis, c_closure=0.0):
    """Frozen field E = E1 cos(k1 x), electrons q/m = -1, kept |k| <= K.  basis in AW, SW, DAI, SWH (SW width a/sqrt2)."""
    if basis == "SWH":
        return hk_operator(K, Nn, E1, a / np.sqrt(2), u, "SW")
    L = hk_operator(K, Nn, E1, a, u, "AW" if basis in ("AW", "DAI") else basis)
    nk = 2 * K + 1
    if basis == "DAI":
        cs = spill_coeffs(Nn)
        top = Nn - 1
        for ik, kk in enumerate(np.arange(-K, K + 1)):
            k = kk * k1
            # streaming spill: -ik a sqrt((N+1)/2) psi_{N+1} from psi_N, same block
            L[ik * Nn:(ik + 1) * Nn, ik * Nn + top] += cs * (-1j * k * a * np.sqrt(Nn / 2))
            for dk in (-1, 1):  # force spill: -E d_v psi_N = (E/a) sqrt(2(N+1)) psi_{N+1}; E_{+-1} = E1/2
                jk = ik - dk
                if 0 <= jk < nk:
                    L[ik * Nn:(ik + 1) * Nn, jk * Nn + top] += cs * ((E1 / 2) * np.sqrt(2 * Nn) / a)
    if c_closure:
        L -= np.diag(np.tile(closure_diag(Nn, c_closure, E1, a), nk))
    return L
