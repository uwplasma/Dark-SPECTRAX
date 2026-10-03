"""Independent 1D1V Vlasov-Ampere-Proca grid reference (NumPy/SciPy; no SPECTRAX or Dark-SPECTRAX code).

Electrostatic units of slv.py: omega_pe = 1, electrons q/m = -1, fixed uniform ions, epsilon_0 = 1, light
speed c (= 1/beta). Longitudinal fields only, every Fourier mode including k = 0:

    dE/dt = -J,  dE_D/dt = Omega_D^2 A_D - eta J,  dA_D/dt = -E_D - i k phi_D,  dphi_D/dt = -c^2 i k A_D,
    dv/dt = -(E + eta E_D),   J = -int v f dv.

Strang step: x half-advection; mid-step current J = J_pre + n F dt/2 (the exact velocity shift changes the
current by n F dt), solved by fixed-point iteration together with the mid-step fields; dark fields advanced
exactly over dt (matrix exponential per k with J held at that value); E^{n+1} = E^n - dt J; velocity kick by
-(E + eta E_D) dt; x half-advection.
The ordinary Gauss law then holds to the accuracy of the split current; it is monitored, not imposed.
Energies: K = int v^2 f/2, U = <E^2>/2, U_D = <E_D^2 + Omega_D^2 (A_D^2 + phi_D^2/c^2)>/2 (box averages).
"""

import numpy as np
from scipy.linalg import expm

from slv import VP


class VAP(VP):
    def __init__(self, L, Nx, vmax, Nv, eta, Omega_D, c):
        super().__init__(L, Nx, vmax, Nv)
        self.mix, self.Om, self.c = eta, Omega_D, c  # self.eta is the velocity-Fourier grid of VP
        self._cache = {}

    def _prop(self, dt):
        """Per-k propagators for Y = (E_D, A_D, phi_D) with a constant source s = -eta J: exp of a 4x4 block."""
        if dt in self._cache:
            return self._cache[dt]
        P = []
        for k in self.kx:
            M = np.zeros((4, 4), complex)
            M[0, 1] = self.Om ** 2
            M[0, 3] = 1.0              # source column (the 4th state is the constant -eta J)
            M[1, 0] = -1.0
            M[1, 2] = -1j * k
            M[2, 1] = -self.c ** 2 * 1j * k
            P.append(expm(M * dt))
        self._cache[dt] = np.array(P)
        return self._cache[dt]

    def current(self, f):
        return np.fft.fft(-(f * self.v[None, :]).sum(axis=1) * self.dv) / self.Nx   # J_k (forward-normalized)

    def density(self, f):
        return np.fft.fft(f.sum(axis=1) * self.dv) / self.Nx

    def yukawa(self, f):
        """Gauss-consistent ordinary field and static Yukawa dark near field; returns (E_k, Y_k)."""
        rho = -self.density(f)
        rho[0] = 0.0                    # neutralizing ions
        E = np.zeros(self.Nx, complex)
        nz = self.kx != 0
        E[nz] = rho[nz] / (1j * self.kx[nz])
        phi = self.mix * rho / (self.kx ** 2 + self.Om ** 2 / self.c ** 2)
        Y = np.stack([-1j * self.kx * phi, np.zeros(self.Nx, complex), phi], axis=1)
        return E, Y

    def energies(self, f, E, Y):
        w = lambda a: np.sum(np.abs(a) ** 2)  # Parseval for forward-normalized full FFT
        K = 0.5 * (f * self.v[None, :] ** 2).sum() * self.dv / self.Nx
        U = 0.5 * w(E)
        UD = 0.5 * (w(Y[:, 0]) + self.Om ** 2 * (w(Y[:, 1]) + w(Y[:, 2]) / self.c ** 2))
        return K, U, UD

    def run_proca(self, f, dt, tmax, modes=(1,), save_every=1, kicks=()):
        E, Y = self.yukawa(f)
        P, Ph = self._prop(dt), self._prop(dt / 2)
        nsteps = int(round(tmax / dt))
        kick_steps = {int(round(t / dt)): fn for t, fn in kicks}
        ts, Eh, Dh, diag = [], [], [], []
        WD = 0.0

        def record(n):
            ts.append(n * dt)
            Eh.append([E[m] for m in modes])
            Dh.append([Y[m, 0] for m in modes])
            K, U, UD = self.energies(f, E, Y)
            rho = -self.density(f)
            rho[0] = 0
            gauss = np.max(np.abs(1j * self.kx * E - rho))
            dgauss = np.max(np.abs(1j * self.kx * Y[:, 0] + self.Om ** 2 * Y[:, 2] / self.c ** 2 - self.mix * rho))
            diag.append([K, U, UD, WD, gauss, dgauss])

        record(0)
        for n in range(nsteps):
            if n in kick_steps:
                f = self.shift_v(f, kick_steps[n](self.x))
            f = self.adv_x(f, dt / 2)
            Jpre = self.current(f)
            n_x = f.sum(axis=1) * self.dv
            J = Jpre
            for _ in range(4):  # mid-step current: the kick shifts J by n F dt, use its average
                src = np.concatenate([Y, (-self.mix * J)[:, None]], axis=1)
                Ymid = np.einsum("kij,kj->ki", Ph, src)[:, :3]
                Emid = E - dt / 2 * J
                force = np.fft.ifft((Emid + self.mix * Ymid[:, 0]) * self.Nx).real
                J = Jpre + 0.5 * dt * np.fft.fft(n_x * force) / self.Nx
            WD += dt * self.mix * np.sum((J * np.conj(Ymid[:, 0])).real)   # P_D = eta <J E_D>
            f = self.shift_v(f, -force * dt)
            Y = np.einsum("kij,kj->ki", P, src)[:, :3]
            E = E - dt * J
            f = self.adv_x(f, dt / 2)
            if (n + 1) % save_every == 0:
                record(n + 1)
        return f, np.array(ts), np.array(Eh), np.array(Dh), np.array(diag)
