"""Operators for the anti-symmetric (Issan et al., arXiv:2312.05439) candidate: SQ = square-root symmetric-weight Hermite.

f = g^2, g = sum_n C_n(x,t) psi_n(eta), eta = (v - u)/s, psi_n orthonormal Hermite functions, s = sqrt2 * sigma = a
(g0 = sqrt(Maxwellian) is the single mode psi_0).  g obeys the SAME Vlasov equation as f (Issan Prop. 1), so the
frozen-field linear operator on C is the symmetric-weight Galerkin operator (anti-Hermitian: Galerkin projection of a
skew-adjoint operator with trial = test space and weight 1).  Nonlinearity is only in the moments:
    n = s sum C_n^2,  j-ish = s^2 C^T X C,  energy-ish = s^3 C^T X_inf^2 C   (X_inf: untruncated Jacobi matrix).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dai"))
from highk_eig import operator as hk_operator, k1  # noqa: E402,F401
from dai_ops import closure_diag  # noqa: E402

A = np.sqrt(2e-3)  # AW / SW width a = sqrt2 v_t (v_t^2 = 1e-3)


def operator(K, Nn, E1, variant, c=0.0):
    """Frozen E = E1 cos(k1 x), electrons q/m = -1.  variants: AW (Petrov-Galerkin, + closure c), SQ (sqrt-SW linear op)."""
    if variant == "SQ":
        return hk_operator(K, Nn, E1, A, 0.0, "SW")
    L = hk_operator(K, Nn, E1, A, 0.0, "AW")
    if c:
        L -= np.diag(np.tile(closure_diag(Nn, c, E1, A), 2 * K + 1))
    return L


def X_inf(N, extra=2):
    M = N + extra
    o = np.sqrt(np.arange(1, M) / 2.0)
    return np.diag(o, 1) + np.diag(o, -1)


def sq_moment_mats(N):
    """(w, X, X2) with X the N x N truncated Jacobi matrix (exact for g in span(psi_0..psi_{N-1})) and X2 = (X_inf^2)[:N,:N]."""
    Xi = X_inf(N)
    return Xi[:N, :N].copy(), (Xi @ Xi)[:N, :N].copy()


def sq_otsi_floquet(k, E0, w0=1.0, mi=1836.0, tau=1.0, Nn=64, Nni=16, vte=np.sqrt(1e-3)):
    """Floquet exponent of the sqrt-SW system linearised about g0 = sqrt(Maxwellian) for the quivering two-species plasma
    (pump frame, U_s = (q/m)_s E0 sin(w0 t)/w0, Poisson field).  Per species: dC/dt = -ik(U + s X) C
    + (q/m) (dE/s) c0 (1/sqrt2) e_1,  dn = 2 sqrt(s) C_0 (n0 = 1, c0 = 1/sqrt(s)),  ik dE = sum q dn."""
    from scipy.integrate import solve_ivp
    sp = [(-1.0, 1.0, vte), (1.0, mi, vte * np.sqrt(tau / mi))]
    nns = [Nn, Nni]
    off = np.concatenate([[0], np.cumsum(nns)])
    D = off[-1]
    A0 = np.zeros((D, D), complex)
    diagU = np.zeros(D)
    s_of = [np.sqrt(2) * sg for (_, _, sg) in sp]
    for s, (q, m, sig) in enumerate(sp):
        sw, o = s_of[s], off[s]
        for j in range(nns[s] - 1):
            A0[o + j, o + j + 1] += -1j * k * sw * np.sqrt((j + 1) / 2)
            A0[o + j + 1, o + j] += -1j * k * sw * np.sqrt((j + 1) / 2)
        for s2, (q2, _, _) in enumerate(sp):
            dn = 2 * np.sqrt(s_of[s2])           # dn_s2 = 2 sqrt(s2) C0_s2
            A0[o + 1, off[s2]] += (q / m) * (1.0 / sw) * (1.0 / np.sqrt(sw)) / np.sqrt(2) * q2 * dn / (1j * k)
        diagU[o:o + nns[s]] = q / m * E0 / w0
    T0 = 2 * np.pi / w0

    def rhs(t, y):
        Y = y.reshape(D, D)
        return (A0 @ Y - 1j * k * (diagU * np.sin(w0 * t))[:, None] * Y).ravel()

    sol = solve_ivp(rhs, (0, T0), np.eye(D, dtype=complex).ravel(), method="DOP853", rtol=1e-11, atol=1e-13)
    mu = np.linalg.eigvals(sol.y[:, -1].reshape(D, D))
    i = np.argmax(np.abs(mu))
    return float(np.log(np.abs(mu[i])) / T0), float(np.angle(mu[i]) / T0)
