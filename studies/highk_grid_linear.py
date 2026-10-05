"""highk variant (linear test of the reference): identical to lane_c_grid_mobile.py except that the ONLY
perturbation is an electron density seed 1e-8 cos(J k1 x) (J = argv[6]); output tag gridlin_. Compared with
studies/highk_linear.py (exact Volterra and linear Hermite), it tests whether the grid reproduces the physical
finite-k growth of the quivering plasma. Run: python studies/highk_grid_linear.py VQ T NX NVE DT J

Original docstring: Lane C independent check with MOBILE ions: two-species 1D1V grid Vlasov-Ampere (spectral x and v shifts, the
studies/refs/code/slv.py primitives, no SPECTRAX code), lab frame, same physics as studies/lane_c_run.py r0:
v_te^2 = 1e-3, m_i/m_e = 1836, T_i = T_e, L = 40, drive E0 cos(w t) on both species, seeds
n_e = 1 + 1e-3 cos k1x - 1e-4 sin 2k1x, n_i = 1 + 1e-3 cos k1x. Electron v-grid +-64 v_te (the quiver reaches
~30 v_te at v_q/v_te = 0.1, t = 600); ion v-grid +-40 v_ti. Purpose: is the abrupt loss of agreement of the
collisionless Hermite runs (t_res ~ t_pos) also present, at the same time, in an independent discretization?

Run: python studies/lane_c_grid_mobile.py VQ T NX NVE DT -> studies/lane_c/gridm_vq<VQ>_Nx<NX>_Nv<NVE>_dt<DT>.npz/.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "refs" / "code"))  # slv.py
from slv import VP  # noqa: E402

vq, T, Nx, Nve, dt = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
J = int(sys.argv[6])
vte, mi = np.sqrt(1e-3), 1836.0
vti = vte / np.sqrt(mi)
Lx, w = 40.0, np.sqrt(1 + 1 / mi)
E0 = vq * vte
se, si = VP(Lx, Nx, 64 * vte, Nve), VP(Lx, Nx, 40 * vti, 512)
k1, x = 2 * np.pi / Lx, se.x
max_ = lambda v, s: np.exp(-v[None, :] ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)  # noqa: E731
fe = max_(se.v, vte) * (1 + 1e-8 * np.cos(J * k1 * x))[:, None]
fi = max_(si.v, vti) * np.ones_like(x)[:, None]
dens = lambda s, f: f.sum(axis=1) * s.dv  # noqa: E731
flux = lambda s, f: (f * s.v[None, :]).sum(axis=1) * s.dv  # noqa: E731
kin = lambda s, f, m: 0.5 * m * (f * s.v[None, :] ** 2).sum() * s.dv / Nx  # noqa: E731
Ek = np.fft.fft(dens(si, fi) - dens(se, fe)) / Nx
Ek[se.kx != 0] /= 1j * se.kx[se.kx != 0]
Ek[0] = 0.0
cur = lambda: np.fft.fft(flux(si, fi) - flux(se, fe)) / Nx  # noqa: E731
Wx, rec, spec = 0.0, [], []
K0 = (kin(se, fe, 1.0), kin(si, fi, mi))
every = int(round(0.5 / dt))
tic = time.perf_counter()
for n in range(int(round(T / dt)) + 1):
    t = n * dt
    if n % every == 0:
        Ux = 0.5 * np.sum(np.abs(Ek[1:]) ** 2)  # nonzero-k field energy (box average)
        spec.append(np.abs(Ek[: Nx // 2 + 1]) ** 2)
        rec.append([t, kin(se, fe, 1.0) - K0[0], kin(si, fi, mi) - K0[1], Wx, Ux, 0.5 * np.abs(Ek[0]) ** 2,
                    fe.min() / fe.max(), np.abs(fe[:, [0, -1]]).max() / fe.max()])
    if n == int(round(T / dt)):
        break
    fe, fi = se.adv_x(fe, dt / 2), si.adv_x(fi, dt / 2)
    Jpre, ne, ni = cur(), dens(se, fe), dens(si, fi)
    Ed = E0 * np.cos(w * (t + dt / 2))
    J = Jpre
    for _ in range(4):  # mid-step field and current (as in slv_proca.py / the gate's grid check)
        Emid = Ek - dt / 2 * J
        F = np.fft.ifft(Emid * Nx).real + Ed
        J = Jpre + 0.5 * dt * np.fft.fft(F * (ne + ni / mi)) / Nx
    Wx += dt * J[0].real * Ed
    fe, fi = se.shift_v(fe, -F * dt), si.shift_v(fi, F * dt / mi)
    Ek = Ek - dt * J
    fe, fi = se.adv_x(fe, dt / 2), si.adv_x(fi, dt / 2)
r = np.array(rec)
tag = f"gridlin_J{J}_vq{vq:g}_Nx{Nx}_Nv{Nve}_dt{dt:g}"
out = Path(__file__).resolve().parent / "highk"
np.savez_compressed(out / f"{tag}.npz", t=r[:, 0], dK_e=r[:, 1], dK_i=r[:, 2], W_ext=r[:, 3], U_k=r[:, 4], U_0=r[:, 5],
                    fmin_rel=r[:, 6], edge=r[:, 7], Ek2=np.array(spec))
(out / f"{tag}.json").write_text(json.dumps({"case": tag, "wall_time": time.perf_counter() - tic, "T": T,
                                             "min_f_rel": float(r[:, 6].min()), "max_edge": float(r[:, 7].max()),
                                             "command": "python studies/highk_grid_linear.py " + " ".join(sys.argv[1:])}, indent=1) + "\n")
print(tag, time.perf_counter() - tic)
