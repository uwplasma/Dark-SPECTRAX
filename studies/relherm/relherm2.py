"""1D2V extension of relherm: (x; ux, uy), fields Ex, Ey, Bz (+ uniform external Bz0 and uniform drive Ey_d(t)).

State C[s, nx, ny, k].  Equations (c = 1 in field units, inv_c2 as in relherm):
  dC/dt = -i k Vx C + (q/m) [ (sqrt(2 nx)/ax) low_x( Ex C + Bz Vy C ) + (sqrt(2 ny)/ay) low_y( (Ey+Ed) C - Bz Vx C ) ]
  Vx, Vy: nodal multiplication by vx, vy = u/gamma on the tensor Gauss-Hermite grid, gamma = sqrt(1 + inv_c2 (ux^2 + uy^2)).
  (v x B).grad_u phi_n = +- (u_i ...) uses grad_u . (v x B) = 0, so int phi (F.grad f) = - int f F.grad phi.)
  Fields: dEx = -Jx, dEy = -Jy - d_x Bz, dBz = -d_x Ey,  J = q ax ay [V C]_00.
"""
import jax
import jax.numpy as jnp
import numpy as np
from relherm import hermite_eig, project

jax.config.update("jax_enable_x64", True)


class Setup2:
    def __init__(self, N, Nx, Lx, q, m, ax, ay, inv_c2=0.0):
        self.N, self.Nx, self.Lx, self.inv_c2 = N, Nx, Lx, inv_c2
        self.q, self.m, self.ax, self.ay = (jnp.asarray(x, float) for x in (q, m, ax, ay))
        self.Ns = len(q)
        self.lam, self.P, self.T = hermite_eig(N)
        self.sq2n = jnp.sqrt(2.0 * jnp.arange(N))
        self.nk = Nx // 2 + 1
        idx = jnp.arange(self.nk)
        self.k = 2 * jnp.pi * idx / Lx
        self.mask = (3 * idx < Nx).astype(float)

    def grid(self, s, u0):
        ux = u0[s, 0] + self.ax[s] * self.lam[:, None]
        uy = u0[s, 1] + self.ay[s] * self.lam[None, :]
        return ux, uy, jnp.sqrt(1.0 + self.inv_c2 * (ux ** 2 + uy ** 2))

    def apply(self, g, A):
        """Multiply A[nx, ny, k] by the nodal function g[jx, jy] (Galerkin matrix on the tensor Gauss-Hermite grid)."""
        P = self.P
        F = jnp.einsum("ai,bj,abk->ijk", P, P, A)
        return jnp.einsum("ai,bj,ijk->abk", P, P, g[:, :, None] * F)

    def vx(self, s, u0, A):
        ux, uy, gm = self.grid(s, u0)
        return self.apply(ux / gm, A)

    def vy(self, s, u0, A):
        ux, uy, gm = self.grid(s, u0)
        return self.apply(uy / gm, A)

    def kin(self, s, u0, A):
        ux, uy, gm = self.grid(s, u0)
        s2 = self.inv_c2 * (ux ** 2 + uy ** 2)
        g = (ux ** 2 + uy ** 2) / (1.0 + jnp.sqrt(1.0 + s2))  # (gamma-1)/inv_c2
        return self.apply(g, A)


def to_x(S, A):
    return jnp.fft.irfft(A * S.mask, n=S.Nx, axis=-1, norm="forward")


def to_k(S, A):
    return jnp.fft.rfft(A, axis=-1, norm="forward") * S.mask


def low(A, axis):
    pad = [(0, 0)] * A.ndim
    pad[axis] = (1, 0)
    return jnp.pad(A, pad)[tuple(slice(0, n) for n in A.shape)]


def kinetic_rhs(S, C, Ex, Ey, Bz, u0):
    """Ex, Ey, Bz: Fourier coefficients (nk,) of the fields entering the force (Ey includes the drive, Bz the external field at k=0)."""
    Exx, Eyx, Bzx = to_x(S, Ex), to_x(S, Ey), to_x(S, Bz)
    out = []
    for s in range(S.Ns):
        Cs = C[s]
        VxC, VyC = S.vx(s, u0, Cs), S.vy(s, u0, Cs)
        stream = -1j * S.k * VxC
        Cx, VxCx, VyCx = to_x(S, Cs), to_x(S, VxC), to_x(S, VyC)
        Xx = Exx * Cx + Bzx * VyCx
        Xy = Eyx * Cx - Bzx * VxCx
        fx = (S.sq2n / S.ax[s])[:, None, None] * low(Xx, 0)
        fy = (S.sq2n / S.ay[s])[None, :, None] * low(Xy, 1)
        out.append(stream + (S.q[s] / S.m[s]) * to_k(S, fx + fy))
    return jnp.stack(out)


def current_k(S, C, u0):
    Jx = sum(S.q[s] * S.ax[s] * S.ay[s] * S.vx(s, u0, C[s])[0, 0] for s in range(S.Ns))
    Jy = sum(S.q[s] * S.ax[s] * S.ay[s] * S.vy(s, u0, C[s])[0, 0] for s in range(S.Ns))
    return Jx, Jy


def kinetic_energy(S, C, u0):
    return sum(S.m[s] * S.ax[s] * S.ay[s] * jnp.real(S.kin(s, u0, C[s])[0, 0, 0]) for s in range(S.Ns))


def field_energy(Ex, Ey, Bz):
    f = lambda A: 0.5 * (jnp.abs(A[0]) ** 2 + 2 * jnp.sum(jnp.abs(A[1:]) ** 2))
    return f(Ex) + f(Ey) + f(Bz)


def make_rhs(S, u0, drive, Bext, evolve_fields=True):
    def rhs(t, y):
        C, Ex, Ey, Bz, W = y
        Ed = drive(t)
        Eyt = Ey.at[0].add(Ed)
        Bzt = Bz.at[0].add(Bext)
        dC = kinetic_rhs(S, C, Ex, Eyt, Bzt, u0)
        Jx, Jy = current_k(S, C, u0)
        if evolve_fields:
            dEx, dEy, dBz = -Jx, -Jy - 1j * S.k * Bz, -1j * S.k * Ey
        else:
            dEx, dEy, dBz = jnp.zeros_like(Ex), jnp.zeros_like(Ey), jnp.zeros_like(Bz)
        dW = jnp.real(Jy[0]) * Ed
        return (dC, dEx, dEy, dBz, dW)
    return rhs


def run(S, y0, u0, drive, Bext, T, dt, nsave, evolve_fields=True, probe=None):
    from relherm import rk4
    rhs = make_rhs(S, u0, drive, Bext, evolve_fields)
    per = int(round(T / dt)) // nsave

    @jax.jit
    def chunk(y, j):
        y, _ = jax.lax.scan(lambda y, i: (rk4(rhs, y, i * dt, dt), None), y, j * per + jnp.arange(per))
        Jx, Jy = current_k(S, y[0], u0)
        return y, (jnp.real(Jx[0]), jnp.real(Jy[0]), kinetic_energy(S, y[0], u0), field_energy(y[1], y[2], y[3]), y[4])

    outs, y = [], y0
    for j in range(nsave):
        y, o = chunk(y, j)
        outs.append(o)
    return y, outs


def hyper_spectrum(N):
    """SPECTRAX precompute_collisions on both axes: n(n-1)(n-2)/((N-1)(N-2)(N-3)) summed over nx and ny."""
    i = jnp.arange(N, dtype=float)
    h = i * (i - 1) * (i - 2) / ((N - 1) * (N - 2) * (N - 3))
    return h[:, None] + h[None, :]


def make_rhs_closure(S, u0, drive, Bext, nu):
    base = make_rhs(S, u0, drive, Bext, True)
    col = hyper_spectrum(S.N)

    def rhs(t, y):
        C, Ex, Ey, Bz, W, D = y
        dC, dEx, dEy, dBz, dW = base(t, (C, Ex, Ey, Bz, W))
        loss = -nu * col[None, :, :, None] * C
        dD = sum(S.m[s] * S.ax[s] * S.ay[s] * jnp.real(S.kin(s, u0, loss[s])[0, 0, 0]) for s in range(S.Ns))
        return (dC + loss, dEx, dEy, dBz, dW, dD)
    return rhs


def run_closure(S, y0, u0, drive, Bext, T, dt, nsave, nu):
    from relherm import rk4
    rhs = make_rhs_closure(S, u0, drive, Bext, nu)
    per = int(round(T / dt)) // nsave

    @jax.jit
    def chunk(y, j):
        y, _ = jax.lax.scan(lambda y, i: (rk4(rhs, y, i * dt, dt), None), y, j * per + jnp.arange(per))
        Jx, Jy = current_k(S, y[0], u0)
        return y, (jnp.real(Jx[0]), jnp.real(Jy[0]), kinetic_energy(S, y[0], u0), field_energy(y[1], y[2], y[3]), y[4], y[5])

    outs, y = [], y0
    for j in range(nsave):
        y, o = chunk(y, j)
        outs.append(o)
    return y, outs
