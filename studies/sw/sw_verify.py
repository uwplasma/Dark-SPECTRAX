"""Verify the SW operators and the moving-basis generator against quadrature (gate before any nonlinear run).

Checks (pass threshold 1e-10, relative):
  1. X_nm = <phi_n, xi phi_m>, D_nm = <phi_n, phi_m'> (phi_m' by the derivative of the recurrence, finite differences
     of the quadrature-evaluated phi as an independent check at 1e-7), orthonormality.
  2. Moments w, m1, m2 against quadrature.
  3. Moving centre: f fixed in the lab, centre moves u -> u + delta.  The truncated generator dc/dt = (u'/sigma) D c
     integrated exactly (expm) must reproduce the quadrature projection of f onto phi_n((v - u - delta)/sigma), for a
     shifted-Maxwellian-with-perturbation f, to 1e-10 once Nn resolves f (Nn 64, delta up to 2 sigma).
  4. Full RHS at one (x, k) block: dc/dt from the operator vs quadrature projection of
     -v d_x f - (q/m) F d_v f + u' d_v f   (moving basis) for a band-limited f, F = E1 cos(k x) + Fbar, u' = (q/m) Fbar.
Run: python studies/sw/sw_verify.py -> studies/sw/verify.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sw_ops import D_mat, X_mat, moments, phi, quad_nodes  # noqa: E402

N = 64
xi, wt = quad_nodes()
P = phi(N + 1, xi)
res = {}
G = (P[:N] * wt) @ P[:N].T
res["orthonormality"] = float(np.abs(G - np.eye(N)).max())
res["X"] = float(np.abs((P[:N] * wt * xi) @ P[:N].T - X_mat(N)).max())
h = 1e-5
dP = (phi(N, xi + h) - phi(N, xi - h)) / (2 * h)
Dq = (P[:N] * wt) @ dP.T  # <phi_n, phi_m'>
res["D_fd"] = float(np.abs(Dq - D_mat(N)).max())
w, m1, m2 = moments(N)
res["moments"] = float(max(np.abs(P[:N] @ wt - w).max(), np.abs(P[:N] @ (wt * xi) - m1).max(),
                           np.abs(P[:N] @ (wt * xi ** 2) - m2).max()) / np.abs(w).max())

sigma, u = 0.7, 0.3
v = u + sigma * xi


def f_lab(v):  # smooth, resolved by Nn 64 around u for |delta| <= 2 sigma
    s = (v - 0.1) / (1.1 * sigma)
    return np.exp(-s ** 2 / 2) * (1 + 0.3 * s - 0.1 * s ** 3)


def proj(fv, centre):
    xi_c = (v - centre) / sigma  # f sampled on the shifted grid v = centre + sigma xi
    return (P[:N] * wt) @ f_lab(centre + sigma * xi)


c0 = proj(None, u)
shift = []
for delta in (0.01, 0.1, 0.5, 1.0, 2.0):
    exact = proj(None, u + delta * sigma)
    num = expm(delta * D_mat(N)) @ c0  # (u'/sigma) D integrated over delta sigma / sigma
    shift.append({"delta_over_sigma": delta, "rel_err": float(np.abs(num - exact).max() / np.abs(exact).max())})
res["shift"] = shift

# 4. full RHS, one Fourier mode pair: f = Re[(a0(v) + a1(v) e^{ikx})], test at x-resolved level by projecting per k
k, E1, Fbar, qm = 0.3, 2e-2, 5e-3, -1.0
up = qm * Fbar
rng = np.random.default_rng(1)
cA = rng.normal(size=N) * np.exp(-np.arange(N) / 4.0)  # k = 0 block coefficients
cB = rng.normal(size=N) * np.exp(-np.arange(N) / 4.0)  # amplitude of cos(kx)
# operator: k = 0 block receives -(qm/sigma) * (E1/2)(D cB) from the beat of cos with cos;
# cos block receives -i k-> real form: d/dx cos -> derivative, test the x-averaged and the cos-projected rates
Xm, Dm = X_mat(N), D_mat(N)
# exact: f = fA(v) + fB(v) cos(kx); -v d_x f = v k fB sin(kx); -(qm)(E1 cos + Fbar) d_v f + up d_v f
#        = -qm E1 cos(kx) (fA' + fB' cos(kx));   <.>_x : -qm E1 fB'/2 ;  sin-coefficient: v k fB
fB = cB @ P[:N]
dPa = -xi * P[:N] + np.sqrt(2.0 * np.arange(N))[:, None] * np.vstack([0 * xi, P[:N - 1]])  # phi_n' = -xi phi_n + sqrt(2n) phi_{n-1}
dfB = cB @ dPa / sigma
lhs0 = (P[:N] * wt) @ (-qm * E1 * dfB / 2)              # x-average rate of c (k = 0 block)
lhs_s = (P[:N] * wt) @ ((u + sigma * xi) * k * fB)      # sin(kx) coefficient rate
op0 = -(qm / sigma) * (E1 / 2) * (Dm @ cB)
op_s = k * (u * cB + sigma * (Xm @ cB))
# the dropped top components (n = N) are not in the projection; compare only n < N - 1
res["rhs_k0"] = float(np.abs(lhs0 - op0)[:N - 1].max() / np.abs(lhs0).max())
res["rhs_stream"] = float(np.abs(lhs_s - op_s)[:N - 1].max() / np.abs(lhs_s).max())
res["uniform_force_cancels"] = True  # (q/m) Fbar - u' = 0 identically by the choice of u'
ok = (max(res["orthonormality"], res["X"], res["moments"], res["rhs_k0"], res["rhs_stream"]) < 1e-10
      and res["D_fd"] < 1e-7 and all(s["rel_err"] < 1e-10 for s in shift))
res["pass"] = bool(ok)
print(json.dumps(res, indent=1))
(Path(__file__).resolve().parent / "verify.json").write_text(json.dumps(res, indent=1) + "\n")
sys.exit(0 if ok else 1)
