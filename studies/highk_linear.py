"""highk (c)+(d): finite-k LINEAR stability of the quivering two-species Maxwellian of H05, per Fourier mode k = j k1.

Two independent solvers of the same linear problem (uniform quivering background, one perturbed mode k, nu = 0):
  exact : Volterra equation of the linearized Vlasov-Poisson system written in each species' oscillation frame,
          where free streaming is exact (kernel ik tau exp(-k^2 v_t^2 tau^2/2)); fields carry the phase exp(ik X_s(t)).
          No velocity discretization at all -> the physical answer (oscillating two-stream / Silin parametric).
  herm  : the asymmetric-Hermite (AW) Fourier system exactly as SPECTRAX's pump frame discretizes it (basis centred
          on u_s(t), width a_s = sqrt2 v_ts, closure g_N = 0), linearized about g = delta_n0, integrated by DOP853.
Background: the exact uniform driven two-fluid oscillator r'' + w^2 r = -(1+eps) E0 cos(w t), r = x_e - x_i,
x_e = r/(1+eps), x_i = -eps r/(1+eps)  (H05: v_te^2 = 1e-3, m_i = 1836, T_i = T_e, L = 40, w = sqrt(1+eps)).
Perturbation: electron density 1 at mode k (any normalisation: linear). Output |E_k|^2(t) per method.

Rejection criteria (declared before running):
  - if herm(Nn) agrees with exact to 10% in |E_k|^2 up to t_end for k1..k5 at v_q = 0.1, the Hermite linear operator
    is NOT the cause of the Nx=16 k5 growth (look at nonlinear/aliasing instead);
  - if exact shows |E_k5|^2 amplification >= 1e6 by t = 450 at v_q = 0.1, the k5 growth is PHYSICAL (OTSI/Silin).
Run: python studies/highk_linear.py VQ TEND  -> studies/highk/linear_vq<VQ>.json/.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

vte, eps = np.sqrt(1e-3), 1 / 1836
vts = np.array([vte, vte * np.sqrt(eps)])  # sigma of each Maxwellian
qs, ms = np.array([-1.0, 1.0]), np.array([1.0, 1836.0])
Lx, w = 40.0, np.sqrt(1 + eps)
k1 = 2 * np.pi / Lx


def background(E0, t):
    A = (1 + eps) * E0
    r = -(A / (2 * w)) * t * np.sin(w * t)
    rd = -(A / (2 * w)) * (np.sin(w * t) + w * t * np.cos(w * t))
    X = np.stack([r / (1 + eps), -eps * r / (1 + eps)])
    U = np.stack([rd / (1 + eps), -eps * rd / (1 + eps)])
    return X, U


def exact(k, E0, T, dt):
    t = np.arange(0.0, T + dt / 2, dt)
    X, _ = background(E0, t)
    ph = np.exp(1j * k * X)                    # (2, Nt)
    tau = t
    Ker = [(qs[s] / ms[s]) * 1j * k * tau * np.exp(-0.5 * (k * vts[s] * tau) ** 2) for s in range(2)]
    G0 = [np.exp(-0.5 * (k * vts[s] * t) ** 2) for s in range(2)]
    dn0 = np.array([1.0, 0.0])
    E = np.zeros(t.size, complex)
    for n in range(t.size):
        acc = 0.0
        for s in range(2):
            # trapezoid of Ker(t_n - t') E(t') ph(t'), t' in [0, t_n]; Ker(0) = 0 so E(t_n) does not enter
            if n > 0:
                f = Ker[s][n::-1][: n + 1] * E[: n + 1] * ph[s, : n + 1]
                integ = dt * (f.sum() - 0.5 * (f[0] + f[-1]))
            else:
                integ = 0.0
            dn = (dn0[s] * G0[s][n] - integ) / ph[s, n]
            acc += qs[s] * dn
        E[n] = acc / (1j * k)
    return t, E


def hermite(k, E0, T, Nn, tsave):
    a = np.sqrt(2) * vts
    n = np.arange(Nn)
    sp, sm = np.sqrt((n + 1) / 2), np.sqrt(n / 2)

    def f(t, y):
        g = y.reshape(2, Nn)
        _, U = background(E0, np.array(t))
        Ek = (qs[0] * g[0, 0] + qs[1] * g[1, 0]) / (1j * k)
        dg = np.empty_like(g)
        for s in range(2):
            up = np.zeros(Nn, complex); up[:-1] = g[s, 1:]
            dn = np.zeros(Nn, complex); dn[1:] = g[s, :-1]
            dg[s] = -1j * k * (a[s] * (sp * up + sm * dn) + U[s] * g[s])
            dg[s, 1] += (qs[s] / ms[s]) * Ek * np.sqrt(2) / a[s]
        return dg.ravel()

    y0 = np.zeros(2 * Nn, complex); y0[0] = 1.0
    sol = solve_ivp(f, (0, T), y0, method="DOP853", t_eval=tsave, rtol=1e-10, atol=1e-14)
    g = sol.y.reshape(2, Nn, -1)
    return (qs[0] * g[0, 0] + qs[1] * g[1, 0]) / (1j * k), g


if __name__ == "__main__":
    vq, T = float(sys.argv[1]), float(sys.argv[2])
    E0 = vq * vte
    out, res = Path(__file__).resolve().parent / "highk", {}
    arrays = {}
    for j in range(1, 9):
        k = j * k1
        tic = time.perf_counter()
        t, Ex = exact(k, E0, T, 0.05)
        t2, Ex2 = exact(k, E0, T, 0.025)
        ts = t[::20]
        rec = {"exact_dt_rel_diff_max": float(np.max(np.abs(np.abs(Ex2[::2][::20]) ** 2 - np.abs(Ex[::20]) ** 2)
                                                     / np.maximum.accumulate(np.abs(Ex[::20]) ** 2)))}
        arrays[f"t"] = ts
        arrays[f"exact_k{j}"] = np.abs(Ex[::20]) ** 2
        for Nn in (32, 64, 128):
            Eh, g = hermite(k, E0, T, Nn, ts)
            arrays[f"herm{Nn}_k{j}"] = np.abs(Eh) ** 2
            arrays[f"tail{Nn}_k{j}"] = np.sum(np.abs(g[0, 2 * Nn // 3:]) ** 2, axis=0) / np.sum(np.abs(g[0]) ** 2, axis=0)
            rel = np.abs(np.abs(Eh) ** 2 - np.abs(Ex[::20]) ** 2) / np.maximum.accumulate(np.abs(Ex[::20]) ** 2)
            bad = np.nonzero(rel > 0.1)[0]
            rec[f"herm{Nn}_t_10pct"] = float(ts[bad[0]]) if bad.size else None
        e = np.abs(Ex[::20]) ** 2
        rec.update({"exact_E2_max_over_E2_0": float(e.max() / e[0]), "exact_E2_end_over_0": float(e[-1] / e[0]),
                    **{f"herm{Nn}_E2_end_over_0": float(arrays[f'herm{Nn}_k{j}'][-1] / e[0]) for Nn in (32, 64, 128)},
                    "wall": time.perf_counter() - tic})
        res[f"k{j}"] = rec
        print(j, json.dumps(rec), flush=True)
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"linear_vq{vq:g}.npz", **arrays)
    (out / f"linear_vq{vq:g}.json").write_text(json.dumps({"vq": vq, "T": T, "res": res,
                                                         "command": "python studies/highk_linear.py " + " ".join(sys.argv[1:])}, indent=1) + "\n")
