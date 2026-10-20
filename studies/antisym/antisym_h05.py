"""antisym stage 2: SQUARE-ROOT symmetric-weight Hermite (Issan et al. arXiv:2312.05439) nonlinear 1D1V prototype for H05, no
SPECTRAX code.  Same physics, driver, pump frame, masks, RK4 and diagnostics as studies/antisym/antisym_h05.py, but the unknowns are the
coefficients C_n of g = sqrt(f) (f = g^2; g obeys the same Vlasov equation, so dC/dt has the SW linear form) and the
moments are quadratic: n = s sum C_n^2, int (v-u) f = s^2 C^T X C, int (v-u)^2 f = s^3 C^T X2 C (X2 from the untruncated
Jacobi matrix), s = sqrt2 sigma_s.  Density is positive and sum C^2 = particle number for any E(t), u(t).  Gates: see the
docstring of studies/antisym/antisym_h05_analysis.py (declared before the runs).
Original header follows.

SW Hermite (option B) nonlinear 1D1V prototype for H05: two-species Vlasov-Ampere, no closure, no SPECTRAX code.

Physics and inputs as studies/nxconv_grid.py (r0 seeds f_e ~ 1 + 1e-3 cos k1x - 1e-4 sin 2k1x, f_i ~ 1 + 1e-3 cos k1x,
v_te^2 = 1e-3, m_i/m_e = 1836, T_i = T_e, L = 40, drive E0 cos(w t) on both species, E0 = v_q v_te,
w = sqrt(1 + 1/1836)).  Velocity: orthonormal Hermite functions, width sigma_s = v_ts (Maxwellian = one mode), centre
u_s(t) moving with the uniform force u_s' = (q/m)_s (E_0 + E_drive) (pump frame; studies/sw/sw_ops.py has the
derivation, studies/sw/sw_verify.py the quadrature checks).  x: Fourier, strict 2/3 mask |k| <= (Nx-1)//3 on every
product (pseudo-spectral force E~(x) * (D c)(x), then masked).  Time: classical RK4, fixed dt.  nu = 0.
Diagnostics every 0.5 (same names as the grid records): dK_e, dK_i, W_ext, U_k, U_0, energy_defect, mass_e/i drift,
momentum, U_e/U_i, L2 drift (sum |c|^2, exactly conserved by the semi-discrete system), top-quarter Hermite energy
fraction of the electrons, |E_k|^2 spectrum.
Run: python studies/antisym/antisym_h05.py --vq 0.1 --Nx 32 --Nn 64 --T 1000 -> studies/sw/runs/<tag>.{npz,json}
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from sw_ops import D_mat, X_mat  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--Nni", type=int, default=0, help="ion Hermite modes (default = Nn)")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--save-every", type=float, default=0.5)
    a = ap.parse_args(argv)
    Nx, dt = a.Nx, a.dt
    Nni = a.Nni or a.Nn
    vte, mi, Lx = np.sqrt(1e-3), 1836.0, 40.0
    w0 = np.sqrt(1 + 1 / mi)
    E0 = a.vq * vte
    k1 = 2 * np.pi / Lx
    nk = Nx // 2 + 1
    kk = k1 * np.arange(nk)
    mask = (np.arange(nk) <= (Nx - 1) // 3).astype(float)
    x = np.arange(Nx) * Lx / Nx
    sp = []  # q, m, qm, sigma, N, X, D, w, m1, m2
    for q, m, sig, N in ((-1.0, 1.0, vte, a.Nn), (1.0, mi, vte / np.sqrt(mi), Nni)):
        Xi = X_mat(N + 2)
        sw_ = np.sqrt(2.0) * sig  # SW width s = sqrt2 sigma: g0 = sqrt(Maxwellian) = psi_0
        sp.append(dict(q=q, m=m, qm=q / m, s=sw_, N=N, XT=X_mat(N).T.copy(), DT=D_mat(N).T.copy(), X=Xi[:N, :N].copy(),
                       X2=(Xi @ Xi)[:N, :N].copy(), o=np.sqrt(np.arange(1, N) / 2.0)))
    rf = lambda g: np.fft.rfft(g, axis=0) / Nx  # noqa: E731
    irf = lambda G: np.fft.irfft(G * Nx, n=Nx, axis=0)  # noqa: E731

    def dens(g, s):
        return s["s"] * np.sum(g * g, axis=1)

    def flux(g, s):  # int (v-u) f dv
        return s["s"] ** 2 * 2.0 * np.sum(g[:, :-1] * g[:, 1:] * s["o"], axis=1)

    def init():
        C = []
        for s, prof in zip(sp, (1 + 1e-3 * np.cos(k1 * x) - 1e-4 * np.sin(2 * k1 * x), 1 + 1e-3 * np.cos(k1 * x))):
            c = np.zeros((Nx, s["N"]))
            c[:, 0] = np.sqrt(prof / s["s"])  # g0 = sqrt(f0): C_0^2 = prof / s
            C.append(rf(c) * mask[:, None])
        rho = sum(s["q"] * rf(dens(irf(Cs), s)) * mask for s, Cs in zip(sp, C))[:nk]
        E = np.zeros(nk, complex)
        E[1:] = rho[1:] / (1j * kk[1:]) * mask[1:]
        return [C[0], C[1], E, np.array([0.0, 0.0, 0.0, 0.0])]  # last: u_e, u_i, E_0, W_ext

    def rhs(t, y):
        Ce, Ci, E, sc = y
        ue, ui, Ez, _ = sc
        Ed = E0 * np.cos(w0 * t)
        Et = irf(E)  # non-uniform field (E[0] = 0 by construction)
        J = np.zeros(nk, complex)
        out = []
        for s, C, u in zip(sp, (Ce, Ci), (ue, ui)):
            dC = -1j * kk[:, None] * (u * C + s["s"] * (C @ s["XT"]))
            dC -= (s["qm"] / s["s"]) * rf(Et[:, None] * irf(C @ s["DT"])) * mask[:, None]
            gx = irf(C)
            J += s["q"] * rf(u * dens(gx, s) + flux(gx, s)) * mask
            out.append(dC)
        dE = -J * mask
        dE[0] = 0.0
        F = Ez + Ed
        return [out[0], out[1], dE, np.array([sp[0]["qm"] * F, sp[1]["qm"] * F, -J[0].real, J[0].real * Ed])]

    def axpy(y, h, k):
        return [yi + h * ki for yi, ki in zip(y, k)]

    def diag(y):
        Ce, Ci, E, sc = y
        K, M, P, L2 = [], [], 0.0, 0.0
        for s, C, u in zip(sp, (Ce, Ci), sc[:2]):
            g = irf(C)
            n_, m1_ = dens(g, s), flux(g, s)
            m2_ = s["s"] ** 3 * np.einsum("xn,nm,xm->x", g, s["X2"], g)
            K.append(0.5 * s["m"] * np.mean(u ** 2 * n_ + 2 * u * m1_ + m2_))
            M.append(np.mean(n_))
            P += s["m"] * (u * M[-1] + np.mean(m1_))
            L2 += np.mean(n_) * (1.0 if s["m"] == 1 else 0.0)  # electron particle number (= sum C^2)
        p = np.abs(Ce[0]) ** 2 + 2 * np.sum(np.abs(Ce[1:]) ** 2, 0)
        top = p[3 * sp[0]["N"] // 4:].sum() / p.sum()
        Ux = np.sum(np.abs(E[1:]) ** 2)
        return K, M, P, L2, Ux, 0.5 * sc[2] ** 2, top

    y = init()
    K0, M0, P0, L20, *_ = diag(y)
    nsteps, every = int(round(a.T / dt)), max(1, int(round(a.save_every / dt)))
    rec, spec = [], []
    tic = time.perf_counter()
    n = 0
    while True:
        t = n * dt
        if n % every == 0:
            K, M, P, L2, Ux, U0, top = diag(y)
            dKe, dKi = K[0] - K0[0], K[1] - K0[1]
            Wx = y[3][3]
            rec.append([t, dKe, dKi, Wx, Ux, U0, dKe + dKi + Ux + U0 - Wx, M[0] / M0[0] - 1, M[1] / M0[1] - 1,
                        P - P0, y[3][0], y[3][1], L2 / L20 - 1, top])
            spec.append(np.abs(y[2]) ** 2)
            if not np.isfinite(dKe):
                break
        if n >= nsteps:
            break
        k_1 = rhs(t, y)
        k_2 = rhs(t + dt / 2, axpy(y, dt / 2, k_1))
        k_3 = rhs(t + dt / 2, axpy(y, dt / 2, k_2))
        k_4 = rhs(t + dt, axpy(y, dt, k_3))
        y = [yi + dt / 6 * (p1 + 2 * p2 + 2 * p3 + p4) for yi, p1, p2, p3, p4 in zip(y, k_1, k_2, k_3, k_4)]
        n += 1
    wall = time.perf_counter() - tic
    r = np.array(rec)
    tag = f"sq_vq{a.vq:g}_Nx{Nx}_Nn{a.Nn}" + (f"_Nni{Nni}" if a.Nni else "") + f"_dt{dt:g}_T{a.T:g}"
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["t", "dK_e", "dK_i", "W_ext", "U_k", "U_0", "energy_defect", "mass_e", "mass_i", "momentum", "U_e", "U_i",
             "L2_drift", "top_frac"]
    np.savez_compressed(OUT / f"{tag}.npz", Ek2=np.array(spec), **{nm: r[:, i] for i, nm in enumerate(names)})
    Wm = np.maximum.accumulate(np.abs(r[:, 3]))
    rel = np.abs(r[:, 6]) / np.maximum(Wm, 1e-300)
    first = lambda c: float(r[np.nonzero(c)[0][0], 0]) if np.any(c) else None  # noqa: E731
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent).stdout.strip()
    pre = r[:, 0] <= 600
    out = {"case": tag, "T": float(r[-1, 0]), "finite": bool(np.all(np.isfinite(r[:, 1]))), "wall_time": wall,
           "steps": n, "rhs_evals": 4 * n,
           "max_energy_defect_over_runmaxW_t<=600": float(rel[pre & (r[:, 0] > 50)].max(initial=0.0)),
           "first_energy_defect_over_W_gt_1e-3": first((r[:, 0] > 50) & (rel > 1e-3)),
           "max_mass_drift_e": float(np.abs(r[:, 7]).max()), "max_mass_drift_i": float(np.abs(r[:, 8]).max()),
           "max_momentum_drift": float(np.abs(r[:, 9]).max()),
           "max_momentum_drift_t<=600": float(np.abs(r[pre, 9]).max()),
           "max_L2_drift_e": float(np.abs(r[:, 12]).max()), "max_top_frac": float(r[:, 13].max()),
           "repository_commit": sha, "command": "python studies/antisym/antisym_h05.py " + " ".join(sys.argv[1:] if argv is None else argv)}
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
