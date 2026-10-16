"""Symmetric-weight (SW) Hermite operators for the 1D1V prototype (option B of docs/design/l2-stable-basis.md).

Basis: orthonormal Hermite functions phi_n(xi) = H_n(xi) exp(-xi^2/2) / sqrt(2^n n! sqrt(pi)), xi = (v - u(t)) / sigma,
i.e. SW coefficients of f / sqrt(w) with w = exp(-xi^2): f(x, v, t) = sum_n c_n(x, t) phi_n(xi).  With sigma = v_t the
Maxwellian is the single mode phi_0.  The truncated Galerkin system (weight 1) is the plain truncation of
    xi phi_m   = sqrt((m+1)/2) phi_{m+1} + sqrt(m/2) phi_{m-1}          -> X (symmetric tridiagonal)
    d_xi phi_m = sqrt(m/2) phi_{m-1} - sqrt((m+1)/2) phi_{m+1}          -> D (antisymmetric tridiagonal)
Moving basis (pump frame), derived here: at fixed v, d_t f = sum (dc_n/dt) phi_n - (u'/sigma) sum c_n phi_n'(xi), so
    dc/dt = -ik (u + sigma X) c - (1/sigma) [(q/m) F(x) - u'] D c,      F = E(x) + E_drive(t).
Choosing u' = (q/m) (E_0 + E_drive) (the uniform force: k = 0 field plus drive, as the grid's --frame osc) cancels the
uniform part exactly, so only the non-uniform field E~(x) multiplies D.  -ik(u + sigma X) is anti-Hermitian and
P [E~(x) D] P is anti-self-adjoint for any orthogonal projection P (2/3 mask in x, truncation in n): the truncated
system conserves sum |c|^2 = int f^2 dv dx / sigma exactly (no closure).  A width change would add
-(sigma'/sigma) (xi d_xi) with xi d_xi = X D (not antisymmetric: L2 then changes by the physical factor); unused here.
Moments (w_n = int phi_n dxi; w_{n+1} = sqrt(n/(n+1)) w_{n-1}, w_0 = sqrt(2) pi^(1/4)):
    n = sigma c.w,  int (v - u) f dv = sigma^2 c.m1,  int (v - u)^2 f dv = sigma^3 c.m2,  m1 = X w, m2 = X X w
(using the untruncated X, i.e. the exact moments of the truncated f).  These are not all conserved: the dropped
phi_N components of X c and D c leak through w_N, m1_N, m2_N.
"""
import numpy as np


def X_mat(N):
    o = np.sqrt(np.arange(1, N) / 2.0)
    return np.diag(o, 1) + np.diag(o, -1)


def D_mat(N):
    o = np.sqrt(np.arange(1, N) / 2.0)
    return np.diag(o, 1) - np.diag(o, -1)


def phi(N, xi):
    """phi_0..phi_{N-1} at xi (stable three-term recurrence), shape (N, len(xi))."""
    xi = np.asarray(xi, float)
    P = np.zeros((N, xi.size))
    P[0] = np.pi ** -0.25 * np.exp(-xi ** 2 / 2)
    if N > 1:
        P[1] = np.sqrt(2.0) * xi * P[0]
    for n in range(1, N - 1):
        P[n + 1] = np.sqrt(2.0 / (n + 1)) * xi * P[n] - np.sqrt(n / (n + 1.0)) * P[n - 1]
    return P


def moments(N):
    """w, m1, m2 of length N (exact integrals of phi_n, xi phi_n, xi^2 phi_n)."""
    M = N + 3
    w = np.zeros(M)
    w[0] = np.sqrt(2.0) * np.pi ** 0.25
    for n in range(1, M - 1):
        w[n + 1] = np.sqrt(n / (n + 1.0)) * w[n - 1]
    X = X_mat(M)
    m1 = X @ w
    m2 = X @ m1
    return w[:N], m1[:N], m2[:N]


def quad_nodes(n=600, L=40.0):
    """Trapezoid on [-L, L] (spectrally accurate for these decaying integrands)."""
    xi = np.linspace(-L, L, 20 * n + 1)
    wt = np.full(xi.size, xi[1] - xi[0])
    wt[[0, -1]] *= 0.5
    return xi, wt
