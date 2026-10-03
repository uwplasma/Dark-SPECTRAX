"""Moments, complete energies, Gauss residuals and a damped-mode fit.

Moments implement the Cartesian 3V formulas directly from parent coefficients
``C`` (``n_s = a_x a_y a_z C_000``); they are independent of the parent's own
diagnostics and are checked against Gauss-Hermite quadrature in the tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._model import Model, hermite_index, inner

__all__ = ["moments", "energies", "gauss_residuals", "charge_density", "fit_modes"]


def _coef(model, Ck, s, n, m, p):
    """Coefficient array C^s_{n,m,p}(k), zero if outside the retained orders."""
    if n >= model.Nn or m >= model.Nm or p >= model.Np:
        return jnp.zeros(Ck.shape[1:], Ck.dtype)
    H = model.Nn * model.Nm * model.Np
    return Ck[s * H + hermite_index(n, m, p, model.Nn, model.Nm)]


def moments(model: Model, Ck):
    """Per-species Fourier coefficients of density, flux M_i and second moments M_ij.

    Returns ``n`` with shape ``(Ns, *grid)``, ``M`` ``(Ns, 3, *grid)`` and
    ``M2`` ``(Ns, 3, 3, *grid)`` (velocity moments of f, not multiplied by mass).
    """
    a = np.asarray(model.alpha_s, float).reshape(-1, 3)
    u = np.asarray(model.u_s, float).reshape(-1, 3)
    e = np.eye(3, dtype=int)
    r2 = np.sqrt(2.0)
    ns, Ms, M2s = [], [], []
    for s in range(model.Ns):
        A = a[s].prod()
        C0 = _coef(model, Ck, s, 0, 0, 0)
        C1 = [_coef(model, Ck, s, *e[i]) for i in range(3)]
        ns.append(A * C0)
        Ms.append(jnp.stack([A * (u[s, i] * C0 + a[s, i] / r2 * C1[i]) for i in range(3)]))
        rows = []
        for i in range(3):
            row = []
            for j in range(3):
                Cij = _coef(model, Ck, s, *(e[i] + e[j]))
                if i == j:
                    val = ((u[s, i] ** 2 + a[s, i] ** 2 / 2) * C0
                           + r2 * u[s, i] * a[s, i] * C1[i] + a[s, i] ** 2 / r2 * Cij)
                else:
                    val = (u[s, i] * u[s, j] * C0 + u[s, i] * a[s, j] / r2 * C1[j]
                           + u[s, j] * a[s, i] / r2 * C1[i] + a[s, i] * a[s, j] / 2 * Cij)
                row.append(A * val)
            rows.append(jnp.stack(row))
        M2s.append(jnp.stack(rows))
    return jnp.stack(ns), jnp.stack(Ms), jnp.stack(M2s)


def charge_density(model: Model, Ck):
    """Fourier coefficients of rho = sum_s q_s n_s + rho_background."""
    n, _, _ = moments(model, Ck)
    rho = jnp.tensordot(jnp.asarray(model.qs, float), n, axes=1)
    return rho.at[0, 0, 0].add(model.rho_background)


def energies(model: Model, y):
    """Box-averaged kinetic, Maxwell and complete Proca energies of a state."""
    om0 = model.Omega_cs[0]
    _, _, M2 = moments(model, y["Ck"])
    K_s = 0.5 * jnp.asarray(model.masses) * jnp.real(jnp.trace(M2[..., 0, 0, 0], axis1=1, axis2=2))
    Fk, Dk, Nx = y["Fk"], y["Dk"], model.Nx
    U_gamma = 0.5 * om0 ** 2 * inner(Nx, Fk, Fk)
    U_D = 0.5 * om0 ** 2 * (inner(Nx, Dk[:6], Dk[:6])
                            + model.Omega_D ** 2 * inner(Nx, Dk[6:], Dk[6:]))
    return {"K_species": K_s, "K": jnp.sum(K_s), "U_gamma": U_gamma, "U_D": U_D}


def gauss_residuals(model: Model, y):
    """Max |residual| of ik.E - rho/Om0 and ik.E_D + Omega_D^2 phi_D - eta rho/Om0."""
    nab = model.p["nabla"]
    rho = charge_density(model, y["Ck"]) / model.Omega_cs[0]
    Fk, Dk = y["Fk"], y["Dk"]
    r_ord = 1j * jnp.sum(nab * Fk[:3], axis=0) - rho
    r_dark = 1j * jnp.sum(nab * Dk[:3], axis=0) + model.Omega_D ** 2 * Dk[9] - model.eta * rho
    if model.mode != "self_consistent":
        r_dark = jnp.zeros_like(r_dark)
    return jnp.max(jnp.abs(r_ord)), jnp.max(jnp.abs(r_dark))


def fit_modes(t, z, guesses):
    """Least-squares fit z(t) = sum_j c_j exp(-i w_j t) with complex w_j.

    ``guesses`` are initial complex frequencies; returns fitted frequencies and
    the relative residual norm.  Uses SciPy (a study/test dependency).
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, float)
    z = np.asarray(z, complex)
    z = z / np.linalg.norm(z)  # scale-free residual so tolerances act on the shape
    nw = len(guesses)

    def basis(x):
        w = x[:nw] + 1j * x[nw:]
        return np.exp(-1j * np.outer(t, w))

    def resid(x):
        B = basis(x)
        c = np.linalg.lstsq(B, z, rcond=None)[0]
        r = B @ c - z
        return np.concatenate([r.real, r.imag])

    g = np.asarray(guesses, complex)
    sol = least_squares(resid, np.concatenate([g.real, g.imag]), xtol=1e-14, ftol=1e-14)
    w = sol.x[:nw] + 1j * sol.x[nw:]
    return w, float(np.linalg.norm(resid(sol.x)) / np.linalg.norm(z))
