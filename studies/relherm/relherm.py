"""Relativistic momentum-space Hermite prototype (1D1V, electrostatic Vlasov-Ampere), standalone JAX.

Variable u = p/(m c) per species, c = 1 in the field units; ``inv_c2`` = (v_limit)^-2 lets the same code run the
nonrelativistic limit (inv_c2 = 0: v = u, gamma - 1 -> u^2/2 * inv_c2 ... see ``speed``/``kin``).
Basis (same as SPECTRAX): f = sum_n C_n psi_n(xi)/sqrt(pi), xi = (u - u0)/a, psi_n = H_n e^{-xi^2}/sqrt(2^n n!),
density n = a C_0; test functions phi_n = H_n/sqrt(2^n n!).  Equations (E only, no B):
    dC_n/dt = -i k [V C]_n + (q/m) sqrt(2 n)/a * (E C_{n-1})(x)       V = P diag(v(u0 + a lam_j)) P^T
with T = P diag(lam) P^T the Hermite recurrence matrix (nodes lam_j = Gauss-Hermite nodes), exactly the
eigendecomposition used by spectrax._lawson_streaming.  v = u (inv_c2 = 0) gives V = u0 + a T (the SPECTRAX ladder).
Current J = q a [V C]_0; kinetic energy density m a [G C]_0, G = P diag(g(u0 + a lam)) P^T, g = (gamma - 1)/inv_c2.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def hermite_eig(N):
    off = np.sqrt(np.arange(1, N) / 2.0)
    T = np.diag(off, 1) + np.diag(off, -1)
    lam, P = np.linalg.eigh(T)
    return jnp.asarray(lam), jnp.asarray(P), jnp.asarray(T)


def speed(u, inv_c2):
    return u / jnp.sqrt(1.0 + u * u * inv_c2)


def kin_per_mass(u, inv_c2):
    """(gamma - 1) c^2 with c^-2 = inv_c2; -> u^2/2 as inv_c2 -> 0 (evaluated stably)."""
    s = u * u * inv_c2
    return u * u / (1.0 + jnp.sqrt(1.0 + s))


class Setup:
    """Static data. species: list of dicts q, m, a (width in u)."""

    def __init__(self, N, Nx, Lx, q, m, a, inv_c2=0.0, dense_nonrel=False):
        self.N, self.Nx, self.Lx, self.inv_c2 = N, Nx, Lx, inv_c2
        self.q, self.m, self.a = (jnp.asarray(x, float) for x in (q, m, a))
        self.Ns = len(q)
        self.lam, self.P, self.T = hermite_eig(N)
        n = jnp.arange(N, dtype=float)
        self.sq2n = jnp.sqrt(2.0 * n)
        self.nk = Nx // 2 + 1
        idx = jnp.arange(self.nk)
        self.k = 2 * jnp.pi * idx / Lx
        self.mask = (3 * idx < Nx).astype(float)
        self.dense_nonrel = dense_nonrel  # use T-ladder streaming (reference cost), valid for inv_c2 = 0

    def V(self, u0, s):
        """Streaming/current matrix of species s."""
        if self.dense_nonrel:
            return u0[s] * jnp.eye(self.N) + self.a[s] * self.T
        vj = speed(u0[s] + self.a[s] * self.lam, self.inv_c2)
        return (self.P * vj) @ self.P.T

    def G(self, u0, s):
        gj = kin_per_mass(u0[s] + self.a[s] * self.lam, self.inv_c2)
        return (self.P * gj) @ self.P.T


def _to_x(S, Ak):
    return jnp.fft.irfft(Ak * S.mask, n=S.Nx, axis=-1, norm="forward")


def _to_k(S, Ax):
    return jnp.fft.rfft(Ax, axis=-1, norm="forward") * S.mask


def kinetic_rhs(S, C, Ek, u0, Eref=None):
    """dC/dt from streaming + force.  C (Ns,N,nk) complex, Ek (nk,) complex, u0 (Ns,).
    Eref: uniform field already carried by the moving basis (pump frame): force uses E - Eref."""
    Ex = _to_x(S, Ek)
    if Eref is not None:
        Ex = Ex - Eref
    out = []
    for s in range(S.Ns):
        Vs = S.V(u0, s)
        stream = -1j * S.k * jnp.einsum("nm,mk->nk", Vs, C[s])
        Cs = C[s]
        Cx = _to_x(S, Cs)  # (N, Nx)
        low = jnp.concatenate([jnp.zeros_like(Cx[:1]), Cx[:-1]], axis=0)  # C_{n-1}
        force = (S.q[s] / S.m[s]) * (S.sq2n / S.a[s])[:, None] * Ex[None, :] * low
        out.append(stream + _to_k(S, force))
    return jnp.stack(out)


def current_k(S, C, u0):
    return sum(S.q[s] * S.a[s] * (S.V(u0, s) @ C[s])[0] for s in range(S.Ns))


def kinetic_energy(S, C, u0):
    """sum_s m a <[G C]_0>  (box average = k = 0 component)."""
    return sum(S.m[s] * S.a[s] * jnp.real((S.G(u0, s) @ C[s])[0, 0]) for s in range(S.Ns))


def field_energy(Ek):
    return 0.5 * (jnp.abs(Ek[0]) ** 2 + 2 * jnp.sum(jnp.abs(Ek[1:]) ** 2))


def make_rhs(S, pump, drive):
    """State y = (C, Ek, u0, W). drive(t) -> uniform external field E_d(t)."""

    def rhs(t, y):
        C, Ek, u0, W = y
        Ed = drive(t)
        Ebar = jnp.real(Ek[0]) + Ed
        Etot = Ek.at[0].add(Ed)
        dC = kinetic_rhs(S, C, Etot, u0, Eref=Ebar if pump else None)
        J = current_k(S, C, u0)
        dE = -J
        if pump:
            du0 = S.q / S.m * Ebar
        else:
            du0 = jnp.zeros_like(u0)
        dW = jnp.real(J[0]) * Ed
        return (dC, dE, du0, dW)

    return rhs


def rk4(rhs, y, t, dt):
    add = lambda y, k, c: jax.tree_util.tree_map(lambda a, b: a + c * b, y, k)
    k1 = rhs(t, y)
    k2 = rhs(t + dt / 2, add(y, k1, dt / 2))
    k3 = rhs(t + dt / 2, add(y, k2, dt / 2))
    k4 = rhs(t + dt, add(y, k3, dt))
    return jax.tree_util.tree_map(lambda a, b, c, d, e: a + dt / 6 * (b + 2 * c + 2 * d + e), y, k1, k2, k3, k4)


def run(S, y0, drive, pump, T, dt, nsave):
    rhs = make_rhs(S, pump, drive)
    nsteps = int(round(T / dt))
    per = nsteps // nsave

    def inner(y, i):
        return rk4(rhs, y, (i) * dt, dt), None

    @jax.jit
    def chunk(y, j):
        y, _ = jax.lax.scan(inner, y, j * per + jnp.arange(per))
        return y, (y[1][0].real, y[3], kinetic_energy(S, y[0], y[2]), field_energy(y[1]), y[2], y[1])

    ys = [(0.0, 0.0, 0.0, 0.0)]
    outs = []
    y = y0
    for j in range(nsave):
        y, o = chunk(y, j)
        outs.append(o)
    return y, outs


def maxwell_C(a, u0_unused=None, N=0):
    """Gaussian f (density 1) centred on the basis: C_0 = 1/a, rest zero."""
    c = np.zeros(N)
    c[0] = 1.0 / a
    return c


def project(f_of_u, a, u0, N, nq=2048, span=16.0):
    """C_n = (1/a) int f(u) phi_n((u-u0)/a) du, phi_n = H_n/sqrt(2^n n!), by dense trapezoid."""
    xi = np.linspace(-span, span, nq)
    h = xi[1] - xi[0]
    f = f_of_u(u0 + a * xi)
    phi = np.zeros((N, nq))
    phi[0] = 1.0
    if N > 1:
        phi[1] = np.sqrt(2.0) * xi / np.sqrt(2.0) * 1.0  # H_1/sqrt(2) = xi*sqrt(2)... (H_1 = 2 xi)/sqrt(2) = sqrt(2) xi
        phi[1] = np.sqrt(2.0) * xi
    for n in range(1, N - 1):  # orthonormal recurrence: xi phi_n = sqrt((n+1)/2) phi_{n+1} + sqrt(n/2) phi_{n-1}
        phi[n + 1] = (xi * phi[n] - np.sqrt(n / 2.0) * phi[n - 1]) / np.sqrt((n + 1) / 2.0)
    return (phi * f).sum(axis=1) * h  # int f phi du / a = int f phi dxi
