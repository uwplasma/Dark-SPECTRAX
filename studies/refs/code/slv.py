"""Independent 1D1V Vlasov-Poisson semi-Lagrangian reference (NumPy only; no SPECTRAX code).

Electrostatic units: omega_pe = 1 (total electron density 1), lengths in lambda_D units of the
reference thermal speed chosen by the caller, electrons q/m = -1, fixed uniform neutralizing ions.
Strang splitting: x half-advection -> Poisson field -> v full-advection -> x half-advection.
Both advections are exact Fourier phase shifts (spectral interpolation): x is periodic;
v is treated as periodic on [-vmax, vmax) (f ~ 0 at the ends; wraparound mass is monitored).
Mean (k=0) field: zero (Poisson). For the symmetric initial states used here the mean current is
zero by the (x,v)->(-x,-v) symmetry, so this matches the Ampere k=0 mode of SPECTRAX.
Optional external impulsive kicks f(x,v) -> f(x, v - dv(x)) at given times (echo B06).
"""
import numpy as np

class VP:
    def __init__(self, L, Nx, vmax, Nv):
        self.L, self.Nx, self.Nv = L, Nx, Nv
        self.x = np.arange(Nx)*L/Nx
        self.dv = 2*vmax/Nv
        self.v = -vmax + np.arange(Nv)*self.dv
        self.kx = 2*np.pi*np.fft.fftfreq(Nx, d=L/Nx)
        self.eta = 2*np.pi*np.fft.fftfreq(Nv, d=self.dv)   # velocity-Fourier variable

    def efield(self, f):
        rho = 1.0 - f.sum(axis=1)*self.dv      # ions(+1) - electrons
        rk = np.fft.fft(rho)
        Ek = np.zeros_like(rk)
        nz = self.kx != 0
        Ek[nz] = rk[nz]/(1j*self.kx[nz])
        return np.fft.ifft(Ek).real, Ek/self.Nx

    def adv_x(self, f, dt):
        fk = np.fft.fft(f, axis=0)
        return np.fft.ifft(fk*np.exp(-1j*self.kx[:, None]*self.v[None, :]*dt), axis=0).real

    def shift_v(self, f, s):
        """f(x, v) -> f(x, v - s(x))."""
        fe = np.fft.fft(f, axis=1)
        return np.fft.ifft(fe*np.exp(-1j*self.eta[None, :]*s[:, None]), axis=1).real

    def run(self, f, dt, tmax, kicks=(), modes=(1,), save_every=1):
        nsteps = int(round(tmax/dt))
        ts, Ek_hist, diag = [], [], []
        kick_steps = {int(round(t/dt)): fn for t, fn in kicks}
        def record(n, f):
            E, Ek = self.efield(f)
            ts.append(n*dt); Ek_hist.append([Ek[m] for m in modes])
            mass = f.sum()*self.dv*self.L/self.Nx
            kin = 0.5*(f*self.v[None, :]**2).sum()*self.dv*self.L/self.Nx
            fe = 0.5*(E**2).sum()*self.L/self.Nx
            edge = np.abs(f[:, [0, -1]]).max()
            diag.append([mass, kin, fe, edge, f.min()])
        record(0, f)
        for n in range(nsteps):
            if n in kick_steps:
                f = self.shift_v(f, kick_steps[n](self.x))
            f = self.adv_x(f, dt/2)
            E, _ = self.efield(f)
            f = self.shift_v(f, -E*dt)            # dv/dt = -E for electrons
            f = self.adv_x(f, dt/2)
            if (n+1) % save_every == 0:
                record(n+1, f)
        return f, np.array(ts), np.array(Ek_hist), np.array(diag)

def maxwellian(v, vt=1.0, u=0.0):
    return np.exp(-(v-u)**2/(2*vt**2))/np.sqrt(2*np.pi*vt**2)
