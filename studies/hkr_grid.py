"""hkr: HHS k-range study, masked spectral grid reference (two-species 1D1V Vlasov-Ampere, nonrelativistic).

Question. The Hermite runs keep k lambda_De <= 0.045 (Nx 32) / 0.1 (Nx 64); HHS instabilities reach k lambda_De ~ 1.
Does adding the high-k modes change the HHS energy measure (k != 0 electron 1D kinetic energy, mean flow removed)?

Physics and splitting are those of studies/nxconv_grid.py (v_te^2 = 1e-3, m_i/m_e = 1836, T_i = T_e, L = 40, prescribed
drive E0 cos(w t) on both species, oscillating frame, Strang x/2 - v - x/2 with the 4-pass mid-step field/current,
2/3 mask on f after every x shift). Differences: real FFTs (same numbers to round-off, ~2x cheaper), the HHS energy
measure is recorded, and an optional broadband (shot-noise-like) seed.

Seeds.  --seed det : the deterministic nxconv seed (modes 1, 2 only).
        --seed noise : det + white density noise on every retained k for both species, per-mode
        <|delta n_k / n|^2> = 1/Np with Np = 1e7 per species (independent real/imag Gaussians, fixed RNG stream indexed
        by mode number so that the lower-Nx runs see the subset of the higher-Nx noise).

HHS energy measure (recorded every --save-every): dE_e = (m_e/2) [ <M2> - <M1>^2/<n> ] - initial, with M2 = int f (v+U)^2 dv,
M1 = int f (v+U) dv, box averages <.>. Units n_e m_e c^2 (initial value v_te^2 / 2 = 5e-4 = E_th).
The 200/omega_pe trailing average is taken in studies/hkr_analysis.py.

Run: python studies/hkr_grid.py --vq 0.03 --T 1000 --Nx 64 --Nv 2048 --vmax 16 --seed det [--chunk 30]
     -> studies/hkr/grid/<tag>.{npz,json} (+ checkpoint while incomplete; rerun resumes; exit code 3 = continue)
Validation: --vq 0 --landau KLD (electrons-only linear Landau damping on L = 2 pi/(KLD v_te)).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nxconv_grid as ng  # noqa: E402

OUT = Path(__file__).resolve().parent / "hkr" / "grid"
W = 2
NP = 1e7
KREF = 400  # noise stream covers modes 1..KREF


class Sp(ng.Species):
    def adv_x(self, f, dt):
        fk = sfft.rfft(f, axis=0, workers=W)
        k = self.kx[: self.Nx // 2 + 1]
        fk *= np.exp(-1j * k[:, None] * (self.v[None, :] + self.U) * dt)
        fk *= self.mask[: self.Nx // 2 + 1, None]
        return sfft.irfft(fk, n=self.Nx, axis=0, workers=W)

    def shift_v(self, f, s):
        fe = sfft.rfft(f, axis=1, workers=W)
        eta = self.eta[: self.Nv // 2 + 1]
        fe *= np.exp(-1j * eta[None, :] * s[:, None])
        return sfft.irfft(fe, n=self.Nv, axis=1, workers=W)


def noise_profile(Nx, Lx, seed):
    rng = np.random.default_rng(seed)
    c = (rng.standard_normal(KREF) + 1j * rng.standard_normal(KREF)) / np.sqrt(2 * NP)
    K = (Nx - 1) // 3
    ck = np.zeros(Nx // 2 + 1, complex)
    ck[1 : K + 1] = c[:K]
    return sfft.irfft(ck * Nx, n=Nx)  # n(x) - 1 = sum_k c_k e^{ikx} + cc


def main(argv=None):
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--Nx", type=int, default=64)
    ap.add_argument("--Nv", type=int, default=2048)
    ap.add_argument("--Nvi", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--vmax", type=float, default=16.0)
    ap.add_argument("--seed", choices=("det", "noise"), default="det")
    ap.add_argument("--rng", type=int, default=1)
    ap.add_argument("--chunk", type=float, default=1e9)
    ap.add_argument("--landau", type=float, default=None)
    ap.add_argument("--amp", type=float, default=1e-4)
    ap.add_argument("--save-every", type=float, default=0.5)
    ap.add_argument("--save-f", type=str, default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    W = a.workers
    ng.W = a.workers
    vte = np.sqrt(1e-3)
    mi = 1836.0
    vti = vte / np.sqrt(mi)
    Lx = 2 * np.pi * vte / a.landau if a.landau is not None else 40.0
    se = Sp(Lx, a.Nx, a.vmax * vte, a.Nv, -1.0, 1.0, "spectral", "mask23")
    si = Sp(Lx, a.Nx, 40 * vti, a.Nvi, 1.0 / mi, mi, "spectral", "mask23")
    k1, x = 2 * np.pi / Lx, se.x
    mx = lambda v, s: np.exp(-v[None, :] ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)  # noqa: E731
    if a.landau is not None:
        fe = mx(se.v, vte) * (1 + a.amp * np.cos(k1 * x))[:, None]
        fi = mx(si.v, vti) * np.ones_like(x)[:, None]
    else:
        pe = 1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x)
        pi = 1 + 1e-3 * np.cos(k1 * x)
        if a.seed == "noise":
            pe = pe + noise_profile(a.Nx, Lx, 10 * a.rng)
            pi = pi + noise_profile(a.Nx, Lx, 10 * a.rng + 1)
        fe, fi = mx(se.v, vte) * pe[:, None], mx(si.v, vti) * pi[:, None]
    w = np.sqrt(1 + 1 / mi)
    E0 = a.vq * vte
    tag = f"h_vq{a.vq:g}_Nx{a.Nx}_Nv{a.Nv}_v{a.vmax:g}_dt{a.dt:g}_{a.seed}" + (f"{a.rng}" if a.seed == "noise" else "") + (
        f"_landau{a.landau:g}" if a.landau is not None else "") + ("_fine" if a.save_every < 0.1 else "")
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / f"{tag}.ckpt.npz"
    nsteps, every = int(round(a.T / a.dt)), max(1, int(round(a.save_every / a.dt)))
    save_f = {int(round(float(s) / a.dt)) for s in a.save_f.split(",") if s}
    fft, ifft = ng.fft, ng.ifft
    Ek = fft(si.dens(fi) - se.dens(fe), 0) / a.Nx
    Ek[se.kx != 0] /= 1j * se.kx[se.kx != 0]
    Ek[0] = 0.0
    K0 = (se.kin(fe), si.kin(fi))
    M0 = (se.dens(fe).sum(), si.dens(fi).sum())
    n0, Wx, rec, spec, fsnap, wall0 = 0, 0.0, [], [], {}, 0.0
    fmin_v = fmin_x = 1.0

    def hhs(f, s):  # (m/2)[<M2> - <M1>^2/<n>] box averages
        M2 = (f * (s.v[None, :] + s.U) ** 2).sum() * s.dv / s.Nx
        M1 = (f * (s.v[None, :] + s.U)).sum() * s.dv / s.Nx
        n = f.sum() * s.dv / s.Nx
        return 0.5 * s.mass * (M2 - M1 ** 2 / n)

    H0 = (hhs(fe, se), hhs(fi, si))
    if ck.exists():
        c = np.load(ck, allow_pickle=True)
        fe, fi, Ek, se.U, si.U, Wx, n0 = c["fe"], c["fi"], c["Ek"], float(c["Ue"]), float(c["Ui"]), float(c["Wx"]), int(c["n"])
        rec, spec, wall0 = list(c["rec"]), list(c["spec"]), float(c["wall"])
        fsnap = dict(c["fsnap"].item()) if "fsnap" in c else {}
        K0, M0, H0 = tuple(c["K0"]), tuple(c["M0"]), tuple(c["H0"])
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
                        se.U, si.U, hhs(fe, se) - H0[0], hhs(fi, si) - H0[1]])
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
        for _ in range(4):
            Emid = Ek - a.dt / 2 * J
            F = ifft(Emid * a.Nx, 0).real + Ed
            J = Jpre + 0.5 * a.dt * fft(F * (ne + ni / mi), 0) / a.Nx
        Wx += a.dt * J[0].real * Ed
        Fbar = F.mean()
        se.U += se.qm * Fbar * a.dt
        si.U += si.qm * Fbar * a.dt
        dF = F - Fbar
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
                 wall=wall, K0=np.array(K0), M0=np.array(M0), H0=np.array(H0), fmin_v=fmin_v, fmin_x=fmin_x,
                 fsnap=np.array(fsnap, dtype=object))
        print(f"{tag} checkpoint t={n * a.dt:g} wall={wall:.0f}", flush=True)
        return 3
    r = np.array(rec)
    np.savez_compressed(OUT / f"{tag}.npz", t=r[:, 0], dK_e=r[:, 1], dK_i=r[:, 2], W_ext=r[:, 3], U_k=r[:, 4],
                        U_0=r[:, 5], fmin_v=r[:, 6], fmin_x=r[:, 7], negfrac=r[:, 8], edge=r[:, 9],
                        energy_defect=r[:, 10], mass_e=r[:, 11], mass_i=r[:, 12], momentum=r[:, 13], U_e=r[:, 14],
                        U_i=r[:, 15], dE_e=r[:, 16], dE_i=r[:, 17], Ek2=np.array(spec),
                        **{f"f_t{k}": v for k, v in fsnap.items()})
    neg = np.nonzero(np.minimum(r[:, 6], r[:, 7]) < -1e-3)[0]
    ed = np.nonzero(r[:, 9] > 1e-6)[0]
    de = np.abs(r[:, 10]) / np.maximum(np.maximum.accumulate(np.abs(r[:, 3])), 1e-300)
    dd = np.nonzero(de > 1e-6)[0]
    sha = __import__("subprocess").run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                       cwd=Path(__file__).resolve().parent).stdout.strip()
    (OUT / f"{tag}.json").write_text(json.dumps({
        "case": tag, "T": float(r[-1, 0]), "finite": bool(np.isfinite(fe[0, 0])), "wall_time": wall, "steps": int(n),
        "t_fneg_1e-3": float(r[neg[0], 0]) if neg.size else None,
        "t_edge_1e-6": float(r[ed[0], 0]) if ed.size else None,
        "t_defect_1e-6": float(r[dd[0], 0]) if dd.size else None,
        "min_f_rel": float(np.nanmin(r[:, 6:8])), "max_energy_defect_over_W": float(np.nanmax(de)),
        "repository_commit": sha, "command": "python studies/hkr_grid.py " + " ".join(sys.argv[1:])}, indent=1) + "\n")
    if ck.exists():
        ck.unlink()
    print(tag, "done", f"{wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
