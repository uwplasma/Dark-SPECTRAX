"""relgrid: relativistic 1D1V (x, p_x) Vlasov-Ampere masked spectral grid, electrons and mobile ions (H04 reference).

No SPECTRAX code. Extends studies/nxconv_grid.py (nonrelativistic, same seeds, same Strang x/2 - kick - x/2 order, same diagnostics) to
the momentum variable u = p/(m_s c): c = omega_pe = n_e = m_e = 1, gamma = sqrt(1 + u^2), v = u/gamma, du/dt = (q_s/m_s) E (electrostatic,
no B in 1D). f_s(x, u) is a density per du. Charge density from n_s = sum f du, current J = sum_s q_s sum f v du, Ampere dE/dt = -J.

  --nonrel          v = u, gamma = 1 (reproduces studies/nxconv_grid.py; with --mid iter it is the same scheme to rounding)
  --frame osc       quiver frame generalised relativistically: u = U_s(t) + u'. The x-mean of the force moves U_s exactly
                    (dU_s/dt = (q_s/m_s) Fbar), the grid in u' carries only the non-uniform kick; x-advection uses v(U_s + u'),
                    current and kinetic energy use the full u. Algebraically the lab scheme (v(u) is evaluated, not linearised).
  --frame lab       fixed u grid (needs umax > the quiver momentum).
  --mid taylor      mid-step current J(t + dt/2) from an explicit Taylor expansion in h = dt/2 about the pre-kick state:
                    J' = sum q (q/m) F n1, n1 = int f gamma^-3 du; J'' = sum q (q/m)^2 F^2 n2 + q (q/m) n1 F', n2 = -3 int f u gamma^-5 du,
                    F' = -J(x) + dE_drive/dt (relativistic: no assumption that v is linear in u).
  --mid iter        the nxconv fixed-point iteration with the nonrelativistic n1 = n (exact only for v = u).
  --init maxwell    Maxwellian in u with variance T/m (the nxconv load; T = 1e-3 m c^2 gives u_te = v_te = 0.0316)
         juttner    exp(-gamma m c^2 / T) normalised in u (used for the dispersion gate)
  --noise det       the nxconv deterministic seeds (1e-3 cos k1 x - 1e-4 sin 2k1 x on f_e, 1e-3 cos k1 x on f_i)
          shot      Poisson-type density noise: each Fourier mode k_m (m = 1..(Nx-1)//3, inside the 2/3 mask) of the density of each species gets a
                    complex Gaussian amplitude of variance 1/Np, Np = --npc * 1000 (HHS: Nx = 1000 cells), independent for e and i, seed --seed;
                    the velocity distribution at each x is the local Maxwellian (no phase-space-correlated noise). The initial k != 0 field
                    energy is U_k(0) ~ 6.5e-4 (205/npc) at these 10 modes (box L = 40): npc = 205 is a white-Poisson load of 2e5 particles and
                    puts a Langmuir wave of energy ~ E_th (5e-4) in the box (see results: f < 0 within 1 omega_pe^-1, not usable); npc = 2.05e5
                    gives U_k(0) = 6.5e-7, the value of HHS SM Eq. S37 read as a total energy density.
          none      unseeded (homogeneous)

Run: python studies/relgrid.py --vq 0.03 --T 1000 --Nx 32 --Nu 4096 --dt 0.02 --frame osc --umax 32 --xshift mask23 --chunk 200
     -> studies/relgrid/runs/<tag>.{npz,json} (+ <tag>.ckpt.npz while incomplete); rerunning the same command resumes.
Diagnostics (every --save-every, default 0.5): dK_e, dK_i (relativistic, (gamma-1) m c^2), W_ext, U_k (k != 0 field energy), U_0, energy
defect K_e + K_i + U_k + U_0 - initial (kinetic and field) - W_ext, E1D_e = (1/2) m (<M_xx> - <M_x>^2/<n>) (HHS definition: 1-D velocity variance with the
box-mean flow removed; M_x = int f v du), Ebar (k = 0 field), J0 (k = 0 current), U_e/U_i (frame momentum), <gamma-1>, mass, f positivity.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft

OUT = Path(__file__).resolve().parent / "relgrid" / "runs"
W = 2


def fft(a, axis=0):
    return sfft.fft(a, axis=axis, workers=W)


def ifft(a, axis=0):
    return sfft.ifft(a, axis=axis, workers=W)


class Species:
    def __init__(self, Lx, Nx, umax, Nu, qm, mass, nonrel, xshift):
        self.Nx, self.Nu, self.qm, self.mass, self.nonrel, self.xshift = Nx, Nu, qm, mass, nonrel, xshift
        self.q = np.sign(qm)
        self.mask = np.abs(np.fft.fftfreq(Nx, d=1.0 / Nx)) <= (Nx - 1) // 3
        self.x = np.arange(Nx) * Lx / Nx
        self.du = 2 * umax / Nu
        self.u = -umax + np.arange(Nu) * self.du
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        self.eta = 2 * np.pi * np.fft.fftfreq(Nu, d=self.du)
        self.U = 0.0

    def full(self):
        return self.u + self.U

    def vel(self):
        w = self.full()
        return w if self.nonrel else w / np.sqrt(1.0 + w * w)

    def adv_x(self, f, dt):
        fk = fft(f, 0) * np.exp(-1j * self.kx[:, None] * self.vel()[None, :] * dt)
        if self.xshift == "mask23":
            fk *= self.mask[:, None]
        return ifft(fk, 0).real

    def kick(self, f, s):
        """f(x, u) -> f(x, u - s(x))  (spectral, periodic in u)."""
        return ifft(fft(f, 1) * np.exp(-1j * self.eta[None, :] * s[:, None]), 1).real

    def dens(self, f):
        return f.sum(1) * self.du

    def flux(self, f):
        return (f * self.vel()[None, :]).sum(1) * self.du

    def n1(self, f):  # int f dv/du du
        if self.nonrel:
            return self.dens(f)
        w = self.full()
        return (f * ((1.0 + w * w) ** -1.5)[None, :]).sum(1) * self.du

    def n2(self, f):  # int f d^2 v/du^2 du
        if self.nonrel:
            return np.zeros(f.shape[0])
        w = self.full()
        return (f * (-3.0 * w * (1.0 + w * w) ** -2.5)[None, :]).sum(1) * self.du

    def kin(self, f):
        w = self.full()
        ek = 0.5 * w * w if self.nonrel else w * w / (1.0 + np.sqrt(1.0 + w * w))  # gamma - 1
        return self.mass * (f * ek[None, :]).sum() * self.du / self.Nx

    def gm1(self, f):
        w = self.full()
        ek = 0.5 * w * w if self.nonrel else w * w / (1.0 + np.sqrt(1.0 + w * w))
        return (f * ek[None, :]).sum() * self.du / self.Nx

    def e1d(self, f):
        """(1/2) m (<M_xx> - <M_x>^2 / <n>), box means (k = 0 flow removed)."""
        v = self.vel()
        mxx = (f * (v * v)[None, :]).sum() * self.du / self.Nx
        mx = (f * v[None, :]).sum() * self.du / self.Nx
        n = f.sum() * self.du / self.Nx
        return 0.5 * self.mass * (mxx - mx * mx / n)


def juttner(u, theta):
    from scipy.special import k1e
    g = np.sqrt(1.0 + u * u)
    return np.exp(-(g - 1.0) / theta) / (2.0 * k1e(1.0 / theta))


class Sim:
    def __init__(self, vq=0.0, Nx=32, Nu=4096, Nui=512, dt=0.02, umax=32.0, frame="osc", xshift="mask23", nonrel=False, mid="taylor",
                 init="maxwell", noise="det", amp=1e-4, landau=None, npc=205.0, seed=0, vte2=1e-3, omega=None, Lx=40.0, mi=1836.0,
                 theta_i=None):
        self.a = dict(locals())
        del self.a["self"]
        self.vte = np.sqrt(vte2)
        self.mi = mi
        self.dt, self.frame, self.mid, self.vq = dt, frame, mid, vq
        vti = self.vte / np.sqrt(mi)
        if landau is not None:
            Lx = 2 * np.pi * self.vte / landau
        self.Lx, self.Nx = Lx, Nx
        self.se = Species(Lx, Nx, umax * self.vte, Nu, -1.0, 1.0, nonrel, xshift)
        self.si = Species(Lx, Nx, 40 * vti, Nui, 1.0 / mi, mi, nonrel, xshift)
        self.w = np.sqrt(1 + 1 / mi) if omega is None else omega
        self.E0 = vq * self.vte
        k1, x = 2 * np.pi / Lx, self.se.x

        def load(s, vt):
            if init == "juttner":
                return juttner(s.u, vt ** 2)[None, :] * np.ones((Nx, 1))
            return (np.exp(-s.u[None, :] ** 2 / (2 * vt ** 2)) / np.sqrt(2 * np.pi * vt ** 2)) * np.ones((Nx, 1))

        fe, fi = load(self.se, self.vte), load(self.si, vti)
        if landau is not None:
            fe = fe * (1 + amp * np.cos(k1 * x))[:, None]
        elif noise == "det":
            fe = fe * (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x))[:, None]
            fi = fi * (1 + 1e-3 * np.cos(k1 * x))[:, None]
        elif noise == "shot":
            rng = np.random.default_rng(seed)
            Np = npc * 1000.0
            nm = (Nx - 1) // 3
            for which in ("e", "i"):
                a = (rng.standard_normal(nm) + 1j * rng.standard_normal(nm)) * np.sqrt(0.5 / Np)
                dn = sum(2 * (a[m - 1] * np.exp(1j * m * k1 * x)).real for m in range(1, nm + 1))
                if which == "e":
                    fe = fe * (1 + dn)[:, None]
                else:
                    fi = fi * (1 + dn)[:, None]
        self.fe, self.fi = fe, fi
        self.Ek = fft(self.si.dens(fi) - self.se.dens(fe), 0) / Nx
        self.Ek[self.se.kx != 0] /= 1j * self.se.kx[self.se.kx != 0]
        self.Ek[0] = 0.0
        self.K0 = (self.se.kin(fe), self.si.kin(fi))
        self.E1D0 = self.se.e1d(fe)
        self.U00 = 0.5 * np.sum(np.abs(self.Ek) ** 2)
        self.M0 = (self.se.dens(fe).sum(), self.si.dens(fi).sum())
        self.Wx, self.n = 0.0, 0
        self.fmin_v = self.fmin_x = np.inf
        self.rec = []
        self.spec = []

    def cur(self):
        return fft(self.si.flux(self.fi) - self.se.flux(self.fe), 0) / self.Nx

    def record(self):
        se, si, Ek = self.se, self.si, self.Ek
        t = self.n * self.dt
        fe = self.fe
        fmax = fe.max()
        Ux, U0 = 0.5 * np.sum(np.abs(Ek[1:]) ** 2), 0.5 * np.abs(Ek[0]) ** 2
        dKe, dKi = se.kin(fe) - self.K0[0], si.kin(self.fi) - self.K0[1]
        neg = -fe[fe < 0].sum() / np.abs(fe).sum()
        J0 = self.cur()[0].real
        fv = self.fmin_v if np.isfinite(self.fmin_v) else fe.min()
        fx = self.fmin_x if np.isfinite(self.fmin_x) else fe.min()
        self.rec.append([t, dKe, dKi, self.Wx, Ux, U0, dKe + dKi + Ux + U0 - self.U00 - self.Wx, se.e1d(fe) - self.E1D0, Ek[0].real, J0, se.U, si.U,
                         se.gm1(fe), se.dens(fe).sum() / self.M0[0] - 1, si.dens(self.fi).sum() / self.M0[1] - 1, fv / fmax, fx / fmax,
                         neg, np.abs(fe[:, [0, -1]]).max() / fmax, Ek[1].real, Ek[1].imag])
        self.spec.append(np.abs(Ek[: self.Nx // 2 + 1]) ** 2)
        self.fmin_v = self.fmin_x = np.inf

    def step(self):
        se, si, dt, N = self.se, self.si, self.dt, self.Nx
        t = self.n * dt
        h = dt / 2
        self.fe, self.fi = se.adv_x(self.fe, h), si.adv_x(self.fi, h)
        Jpre = self.cur()
        Edot = -self.E0 * self.w * np.sin(self.w * (t + h))
        Ed = self.E0 * np.cos(self.w * (t + h))
        if self.mid == "taylor":
            Ed0 = self.E0 * np.cos(self.w * t)
            n1 = [(s.q * s.qm, s.n1(f)) for s, f in ((se, self.fe), (si, self.fi))]
            n2 = [(s.q * s.qm ** 2, s.n2(f)) for s, f in ((se, self.fe), (si, self.fi))]
            F0 = ifft(self.Ek * N, 0).real + Ed0
            Jx = ifft(Jpre * N, 0).real
            Fd = -Jx - self.E0 * self.w * np.sin(self.w * t)
            J1 = sum(c * n for c, n in n1)
            J2 = sum(c * n for c, n in n2)
            Jm = Jpre + h * fft(F0 * J1, 0) / N + 0.5 * h * h * fft(F0 * F0 * J2 + J1 * Fd, 0) / N
            F = ifft((self.Ek - h * Jm) * N, 0).real + Ed
        else:
            ne, ni = se.dens(self.fe), si.dens(self.fi)
            J = Jpre
            for _ in range(4):
                F = ifft((self.Ek - h * J) * N, 0).real + Ed
                J = Jpre + h * fft(F * (ne + ni / self.mi), 0) / N
            Jm = J
        # mid-step current for the field update and the work: second-order accurate J(t + dt/2) was built above; the field update uses
        # the full-step current estimate Jm (as nxconv: E <- E - dt J_mid, W += dt J0_mid E_drive)
        self.Wx += dt * Jm[0].real * Ed
        if self.frame == "osc":
            Fbar = F.mean()
            se.U += se.qm * Fbar * dt
            si.U += si.qm * Fbar * dt
            dF = F - Fbar
        else:
            dF = F
        self.fe, self.fi = se.kick(self.fe, se.qm * dF * dt), si.kick(self.fi, si.qm * dF * dt)
        self.fmin_v = min(self.fmin_v, self.fe.min())
        self.Ek = self.Ek - dt * Jm
        self.fe, self.fi = se.adv_x(self.fe, h), si.adv_x(self.fi, h)
        self.fmin_x = min(self.fmin_x, self.fe.min())
        self.n += 1

    def run(self, T, every=0.5, chunk_end=None):
        nsteps = int(round(T / self.dt))
        ev = max(1, int(round(every / self.dt)))
        nend = nsteps if chunk_end is None else min(nsteps, chunk_end)
        while True:
            if self.n % ev == 0 and (not self.rec or self.rec[-1][0] < self.n * self.dt - 1e-9):
                self.record()
            if self.n >= nend:
                break
            self.step()
            if not np.isfinite(self.fe[0, 0]):
                break
        return self.n >= nsteps or not np.isfinite(self.fe[0, 0])

    def save_state(self, path, wall):
        np.savez(path, fe=self.fe, fi=self.fi, Ek=self.Ek, Ue=self.se.U, Ui=self.si.U, Wx=self.Wx, n=self.n, rec=np.array(self.rec),
                 spec=np.array(self.spec), wall=wall, K0=np.array(self.K0), E1D0=self.E1D0, U00=self.U00, M0=np.array(self.M0),
                 fmin_v=self.fmin_v, fmin_x=self.fmin_x)

    def load_state(self, path):
        c = np.load(path, allow_pickle=True)
        self.fe, self.fi, self.Ek = c["fe"], c["fi"], c["Ek"]
        self.se.U, self.si.U, self.Wx, self.n = float(c["Ue"]), float(c["Ui"]), float(c["Wx"]), int(c["n"])
        self.rec, self.spec = list(c["rec"]), list(c["spec"])
        self.K0, self.E1D0, self.M0 = tuple(c["K0"]), float(c["E1D0"]), tuple(c["M0"])
        self.U00 = float(c["U00"])
        self.fmin_v, self.fmin_x = float(c["fmin_v"]), float(c["fmin_x"])
        return float(c["wall"])


COLS = ["t", "dK_e", "dK_i", "W_ext", "U_k", "U_0", "energy_defect", "dE1D_e", "Ebar", "J0", "U_e", "U_i", "gm1_e", "mass_e", "mass_i",
        "fmin_v", "fmin_x", "negfrac", "edge", "Ek1_re", "Ek1_im"]


def tag_of(a):
    return (f"r_vq{a.vq:g}_Nx{a.Nx}_Nu{a.Nu}_dt{a.dt:g}_{a.frame}_u{a.umax:g}_{'nr' if a.nonrel else 'rel'}_{a.mid}_{a.noise}"
            + (f"{a.seed}n{a.npc:g}" if a.noise == "shot" else "") + ("" if a.xshift == "spectral" else f"_x{a.xshift}")
            + (f"_landau{a.landau:g}" if a.landau is not None else "") + ("" if a.init == "maxwell" else f"_{a.init}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--Nu", type=int, default=4096)
    ap.add_argument("--Nui", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--umax", type=float, default=32.0, help="electron half width of the u' grid in units of v_te")
    ap.add_argument("--frame", choices=("lab", "osc"), default="osc")
    ap.add_argument("--xshift", choices=("spectral", "mask23"), default="mask23")
    ap.add_argument("--nonrel", action="store_true")
    ap.add_argument("--mid", choices=("taylor", "iter"), default="taylor")
    ap.add_argument("--init", choices=("maxwell", "juttner"), default="maxwell")
    ap.add_argument("--noise", choices=("det", "shot", "none"), default="det")
    ap.add_argument("--npc", type=float, default=205.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--landau", type=float, default=None)
    ap.add_argument("--amp", type=float, default=1e-4)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--chunk", type=float, default=1e9)
    ap.add_argument("--save-every", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args(argv)
    global W, OUT
    W = a.workers
    if a.out:
        OUT = Path(a.out)
    s = Sim(vq=a.vq, Nx=a.Nx, Nu=a.Nu, Nui=a.Nui, dt=a.dt, umax=a.umax, frame=a.frame, xshift=a.xshift, nonrel=a.nonrel, mid=a.mid,
            init=a.init, noise=a.noise, amp=a.amp, landau=a.landau, npc=a.npc, seed=a.seed, omega=a.omega)
    tag = tag_of(a)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / f"{tag}.ckpt.npz"
    wall0 = s.load_state(ck) if ck.exists() else 0.0
    tic = time.perf_counter()
    done = s.run(a.T, a.save_every, chunk_end=s.n + int(round(a.chunk / a.dt)))
    wall = wall0 + time.perf_counter() - tic
    if not done:
        s.save_state(ck, wall)
        print(f"{tag} checkpoint t={s.n * a.dt:g} wall={wall:.0f}", flush=True)
        return 3
    r = np.array(s.rec)
    np.savez_compressed(OUT / f"{tag}.npz", **{c: r[:, i] for i, c in enumerate(COLS)}, Ek2=np.array(s.spec))
    neg = np.nonzero(np.minimum(r[:, 15], r[:, 16]) < -1e-3)[0]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    (OUT / f"{tag}.json").write_text(json.dumps({
        "case": tag, "T": float(r[-1, 0]), "finite": bool(np.isfinite(s.fe[0, 0])), "wall_time": wall, "steps": int(s.n),
        "t_fneg_1e-3": float(r[neg[0], 0]) if neg.size else None, "min_f_rel": float(np.nanmin(r[:, 15:17])),
        "max_negfrac": float(np.nanmax(r[:, 17])), "max_edge": float(np.nanmax(r[:, 18])),
        "max_energy_defect_over_W": float(np.nanmax(np.abs(r[:, 6])) / max(np.abs(r[:, 3]).max(), 1e-300)),
        "max_mass_drift": float(np.nanmax(np.abs(r[:, 13:15]))), "max_U_e": float(np.abs(r[:, 10]).max()),
        "repository_commit": sha, "command": "python studies/relgrid.py " + " ".join(sys.argv[1:] if argv is None else argv)}, indent=1) + "\n")
    if ck.exists():
        ck.unlink()
    print(tag, "done", f"{wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
