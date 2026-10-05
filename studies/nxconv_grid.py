"""nxconv: improved mobile-ion grid reference for H05 (two-species 1D1V Vlasov-Ampere, no SPECTRAX code).

Same physics and splitting as studies/lane_c_grid_mobile.py (r0 seeds, v_te^2 = 1e-3, m_i/m_e = 1836, T_i = T_e,
L = 40, drive E0 cos(w t) on both species, Strang x/2 - v - x/2 with the 4-pass mid-step field/current), plus:

  --frame osc      oscillating frame: v = U_s(t) + v'. The uniform part of the force (drive + k=0 field) moves
                   U_s exactly; the grid in v' only carries the non-uniform kick. The x shift uses v' + U_s, the
                   current adds n_s U_s and the kinetic energy uses (v' + U_s)^2. Algebraically identical to the lab
                   scheme, but the v-grid no longer has to contain the quiver (~50 v_te at v_q/v_te = 0.1, t = 1000).
  --vshift spectral   exact Fourier phase shift in v (lane_c_grid_mobile.py)
           filtered   same, then an exponential filter exp(-36 (|eta|/eta_max)^36) in the v-Fourier variable
           pfc        positive flux-conservative third-order shift (Filbet, Sonnendrucker, Bertrand 2001), with the
                      positivity limiter only (no upper bound); exact integer part, mass-conservative.
  --chunk / resume: the run is integrated in chunks of --chunk omega_pe^-1 with a checkpoint (state, U_s, W_ext, t)
                   so that each process stays under the 15 min cap; rerunning the same command resumes.

Diagnostics every 0.5: dK_e, dK_i, W_ext, nonzero-k field energy U_k, k=0 field energy U_0, min f_e / max f_e after
the v step and after the x step, negative-mass fraction of f_e, edge value |f_e(v_edge)|/max, energy defect
(K_e + K_i + U_k + U_0 - initial - W_ext), mass and momentum (sum_s m_s P_s) drift, |E_k|^2 spectrum.

Run: python studies/nxconv_grid.py --vq 0.1 --T 1000 --Nx 32 --Nv 8192 --dt 0.02 --frame osc --vmax 32 --vshift pfc
     -> studies/nxconv/grid/<tag>.{npz,json} (+ <tag>.ckpt.npz while incomplete)
Validation (no drive): --vq 0 --landau KLD [--amp A]: electrons only on L = 2 pi/(KLD v_te), mobile ions frozen out.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft

OUT = Path(__file__).resolve().parent / "nxconv" / "grid"
W = 2  # fft workers (laptop load limit)


def fft(a, axis):
    return sfft.fft(a, axis=axis, workers=W)


def ifft(a, axis):
    return sfft.ifft(a, axis=axis, workers=W)


class Species:
    def __init__(self, Lx, Nx, vmax, Nv, qm, mass, vshift):
        self.Nx, self.Nv, self.qm, self.mass, self.vshift = Nx, Nv, qm, mass, vshift
        self.x = np.arange(Nx) * Lx / Nx
        self.dv = 2 * vmax / Nv
        self.v = -vmax + np.arange(Nv) * self.dv
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        self.eta = 2 * np.pi * np.fft.fftfreq(Nv, d=self.dv)
        self.filt = np.exp(-36.0 * (np.abs(self.eta) / np.abs(self.eta).max()) ** 36)
        self.U = 0.0

    def adv_x(self, f, dt):
        fk = fft(f, 0)
        return ifft(fk * np.exp(-1j * self.kx[:, None] * (self.v[None, :] + self.U) * dt), 0).real

    def shift_v(self, f, s):
        """f(x, v) -> f(x, v - s(x))."""
        if self.vshift in ("spectral", "filtered"):
            fe = fft(f, 1) * np.exp(-1j * self.eta[None, :] * s[:, None])
            if self.vshift == "filtered":
                fe *= self.filt[None, :]
            return ifft(fe, 1).real
        return pfc_shift(f, s / self.dv)

    def dens(self, f):
        return f.sum(1) * self.dv

    def flux(self, f):
        return (f * self.v[None, :]).sum(1) * self.dv + self.U * self.dens(f)

    def kin(self, f):
        return 0.5 * self.mass * (f * (self.v[None, :] + self.U) ** 2).sum() * self.dv / self.Nx

    def mom(self, f):
        return self.mass * self.flux(f).sum() / self.Nx


def pfc_shift(f, c):
    """Row-wise translation f(x, v_j) -> f(x, v_j - c_x dv), PFC third order with the positivity limiter."""
    m = np.floor(c).astype(int)
    a = (c - m)[:, None]  # in [0, 1)
    Nv = f.shape[1]
    j = np.arange(Nv)[None, :]
    g = np.take_along_axis(f, (j - m[:, None]) % Nv, axis=1)  # exact integer shift (periodic)
    gp, gm = np.roll(g, -1, 1), np.roll(g, 1, 1)
    dp, dm = gp - g, g - gm
    with np.errstate(divide="ignore", invalid="ignore"):
        ep = np.where(dp > 0, np.minimum(1.0, 2 * g / np.where(dp > 0, dp, 1.0)), 1.0)
        em = np.where(dm < 0, np.minimum(1.0, -2 * g / np.where(dm < 0, dm, -1.0)), 1.0)
    ep, em = np.clip(ep, 0, 1), np.clip(em, 0, 1)
    # flux through v_{j+1/2} for a right shift by a < 1 cell: mass of cell j that crosses into j+1
    phi = a * (g + ep / 6 * (1 - a) * (2 - a) * dp + em / 6 * (1 - a) * (1 + a) * dm)
    return g - phi + np.roll(phi, 1, 1)


def setup(a):
    vte = np.sqrt(1e-3)
    mi = 1836.0
    vti = vte / np.sqrt(mi)
    if a.landau is not None:
        Lx = 2 * np.pi * vte / a.landau  # k lambda_D = a.landau, lambda_D = v_te
    else:
        Lx = 40.0
    se = Species(Lx, a.Nx, a.vmax * vte, a.Nv, -1.0, 1.0, a.vshift)
    si = Species(Lx, a.Nx, 40 * vti, a.Nvi, 1.0 / mi, mi, a.vshift)
    k1, x = 2 * np.pi / Lx, se.x
    mx = lambda v, s: np.exp(-v[None, :] ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)  # noqa: E731
    if a.landau is not None:
        fe = mx(se.v, vte) * (1 + a.amp * np.cos(k1 * x))[:, None]
        fi = mx(si.v, vti) * np.ones_like(x)[:, None]
    elif a.unseeded:
        fe = mx(se.v, vte) * np.ones_like(x)[:, None]
        fi = mx(si.v, vti) * np.ones_like(x)[:, None]
    else:
        fe = mx(se.v, vte) * (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x))[:, None]
        fi = mx(si.v, vti) * (1 + 1e-3 * np.cos(k1 * x))[:, None]
    return se, si, fe, fi, vte, mi, Lx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--Nx", type=int, default=16)
    ap.add_argument("--Nv", type=int, default=4096)
    ap.add_argument("--Nvi", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--vmax", type=float, default=64.0, help="electron v-grid half width in v_te")
    ap.add_argument("--frame", choices=("lab", "osc"), default="lab")
    ap.add_argument("--vshift", choices=("spectral", "filtered", "pfc"), default="spectral")
    ap.add_argument("--chunk", type=float, default=1e9, help="omega_pe^-1 per process before checkpointing")
    ap.add_argument("--landau", type=float, default=None, help="validation: k lambda_D (electrons, no drive)")
    ap.add_argument("--amp", type=float, default=1e-4)
    ap.add_argument("--unseeded", action="store_true")
    ap.add_argument("--save-every", type=float, default=0.5)
    ap.add_argument("--save-f", type=str, default="", help="comma-separated times at which to store f_e(x, v)")
    a = ap.parse_args(argv)
    se, si, fe, fi, vte, mi, Lx = setup(a)
    w = np.sqrt(1 + 1 / mi)
    E0 = a.vq * vte
    tag = (f"g_vq{a.vq:g}_Nx{a.Nx}_Nv{a.Nv}_dt{a.dt:g}_{a.frame}_v{a.vmax:g}_{a.vshift}"
           + (f"_landau{a.landau:g}" if a.landau is not None else "") + ("_unseeded" if a.unseeded else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / f"{tag}.ckpt.npz"
    nsteps, every = int(round(a.T / a.dt)), max(1, int(round(a.save_every / a.dt)))
    save_f = {int(round(float(s) / a.dt)) for s in a.save_f.split(",") if s}
    Ek = fft(si.dens(fi) - se.dens(fe), 0) / a.Nx
    Ek[se.kx != 0] /= 1j * se.kx[se.kx != 0]
    Ek[0] = 0.0
    K0 = (se.kin(fe), si.kin(fi))
    M0 = (se.dens(fe).sum(), si.dens(fi).sum())
    n0, Wx, rec, spec, fsnap, wall0 = 0, 0.0, [], [], {}, 0.0
    fmin_v = fmin_x = 1.0
    if ck.exists():
        c = np.load(ck, allow_pickle=True)
        fe, fi, Ek, se.U, si.U, Wx, n0 = c["fe"], c["fi"], c["Ek"], float(c["Ue"]), float(c["Ui"]), float(c["Wx"]), int(c["n"])
        rec, spec, wall0 = list(c["rec"]), list(c["spec"]), float(c["wall"])
        fsnap = dict(c["fsnap"].item()) if "fsnap" in c else {}
        K0, M0 = tuple(c["K0"]), tuple(c["M0"])
        fmin_v, fmin_x = float(c["fmin_v"]), float(c["fmin_x"])
    tic = time.perf_counter()
    nend = min(nsteps, n0 + int(round(a.chunk / a.dt)))
    cur = lambda: fft(si.flux(fi) - se.flux(fe), 0) / a.Nx  # noqa: E731
    n = n0
    while True:
        t = n * a.dt
        if n % every == 0 and (not rec or rec[-1][0] < t - 1e-9):
            fmax = fe.max()
            Ux, U0 = 0.5 * np.sum(np.abs(Ek[1:]) ** 2), 0.5 * np.abs(Ek[0]) ** 2
            dKe, dKi = se.kin(fe) - K0[0], si.kin(fi) - K0[1]
            neg = -fe[fe < 0].sum() / np.abs(fe).sum()
            rec.append([t, dKe, dKi, Wx, Ux, U0, fmin_v / fmax, fmin_x / fmax, neg,
                        np.abs(fe[:, [0, -1]]).max() / fmax, dKe + dKi + Ux + U0 - Wx,
                        se.dens(fe).sum() / M0[0] - 1, si.dens(fi).sum() / M0[1] - 1, se.mom(fe) + si.mom(fi),
                        se.U, si.U])
            spec.append(np.abs(Ek[: a.Nx // 2 + 1]) ** 2)
            fmin_v = fmin_x = np.inf
        if n in save_f:
            fsnap[f"{t:g}"] = fe.astype(np.float32)
        if n >= nend:
            break
        fe, fi = se.adv_x(fe, a.dt / 2), si.adv_x(fi, a.dt / 2)
        Jpre, ne, ni = cur(), se.dens(fe), si.dens(fi)
        Ed = E0 * np.cos(w * (t + a.dt / 2))
        J = Jpre
        for _ in range(4):  # mid-step field and current (as in lane_c_grid_mobile.py)
            Emid = Ek - a.dt / 2 * J
            F = ifft(Emid * a.Nx, 0).real + Ed
            J = Jpre + 0.5 * a.dt * fft(F * (ne + ni / mi), 0) / a.Nx
        Wx += a.dt * J[0].real * Ed
        if a.frame == "osc":
            Fbar = F.mean()
            se.U += se.qm * Fbar * a.dt
            si.U += si.qm * Fbar * a.dt
            dF = F - Fbar
        else:
            dF = F
        fe, fi = se.shift_v(fe, se.qm * dF * a.dt), si.shift_v(fi, si.qm * dF * a.dt)
        fmin_v = min(fmin_v, fe.min())
        Ek = Ek - a.dt * J
        fe, fi = se.adv_x(fe, a.dt / 2), si.adv_x(fi, a.dt / 2)
        fmin_x = min(fmin_x, fe.min())
        n += 1
        if not np.isfinite(fe[0, 0]):
            break
    wall = wall0 + time.perf_counter() - tic
    done = n >= nsteps or not np.isfinite(fe[0, 0])
    if not done:
        np.savez(ck, fe=fe, fi=fi, Ek=Ek, Ue=se.U, Ui=si.U, Wx=Wx, n=n, rec=np.array(rec), spec=np.array(spec),
                 wall=wall, K0=np.array(K0), M0=np.array(M0), fmin_v=fmin_v, fmin_x=fmin_x,
                 fsnap=np.array(fsnap, dtype=object))
        print(f"{tag} checkpoint t={n * a.dt:g} wall={wall:.0f}", flush=True)
        return 3
    r = np.array(rec)
    np.savez_compressed(OUT / f"{tag}.npz", t=r[:, 0], dK_e=r[:, 1], dK_i=r[:, 2], W_ext=r[:, 3], U_k=r[:, 4],
                        U_0=r[:, 5], fmin_v=r[:, 6], fmin_x=r[:, 7], negfrac=r[:, 8], edge=r[:, 9],
                        energy_defect=r[:, 10], mass_e=r[:, 11], mass_i=r[:, 12], momentum=r[:, 13], U_e=r[:, 14],
                        U_i=r[:, 15], Ek2=np.array(spec), **{f"f_t{k}": v for k, v in fsnap.items()})
    neg = np.nonzero(np.minimum(r[:, 6], r[:, 7]) < -1e-3)[0]
    sha = __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                       cwd=Path(__file__).resolve().parent).stdout.strip()
    (OUT / f"{tag}.json").write_text(json.dumps({
        "case": tag, "T": float(r[-1, 0]), "finite": bool(np.isfinite(fe[0, 0])), "wall_time": wall,
        "steps": int(n), "t_fneg_1e-3": float(r[neg[0], 0]) if neg.size else None,
        "min_f_rel": float(np.nanmin(r[:, 6:8])), "max_negfrac": float(np.nanmax(r[:, 8])),
        "max_edge": float(np.nanmax(r[:, 9])), "max_energy_defect_over_W": float(np.nanmax(np.abs(r[:, 10]))
                                                                                / max(np.abs(r[:, 3]).max(), 1e-300)),
        "max_mass_drift": float(np.nanmax(np.abs(r[:, 11:13]))), "max_momentum": float(np.nanmax(np.abs(r[:, 13]))),
        "repository_commit": sha, "command": "python studies/nxconv_grid.py " + " ".join(sys.argv[1:])}, indent=1) + "\n")
    if ck.exists():
        ck.unlink()
    print(tag, "done", f"{wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
