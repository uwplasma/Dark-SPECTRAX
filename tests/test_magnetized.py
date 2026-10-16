"""B08 rotation: homogeneous drifting Maxwellian in a uniform oblique B0 with a perpendicular mean current."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import darkspectrax as ds


@pytest.mark.parametrize("q", [-1.0, 1.0])
def test_oblique_rotation_mean_flow_and_covariance(q):
    b = 0.5 * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    U0, vt, n = np.array([0.02, -0.01, 0.015]), np.array([0.01, 0.02, 0.015]), 1.0
    model = ds.Model(Nx=1, Nn=3, Nm=3, Np=3, qs=(q,), Omega_cs=(1.0,), alpha_s=tuple(np.sqrt(2) * vt),
                     u_s=tuple(U0), rho_background=-q * n)
    y = ds.maxwellian(model, [n])
    y = {**y, "Fk": y["Fk"].at[3:, 0, 0, 0].set(jnp.asarray(b, complex))}
    out = ds.run(model, y, 6.0, n_save=4, rtol=1e-12, atol=1e-16)
    assert out["status"] == "success"
    C = -np.array([[0, -b[2], b[1]], [b[2], 0, -b[0]], [-b[1], b[0], 0]])  # C v = v x b
    L = np.block([[q * C, q * np.eye(3)], [-q * n * np.eye(3), np.zeros((3, 3))]])
    for i, t in enumerate(out["t"]):
        nn, M, M2 = (np.asarray(x[0, ..., 0, 0, 0]).real for x in ds.moments(model, jnp.asarray(out["Ck"][i])))
        U = M / nn
        X = expm(L * t) @ np.concatenate([U0, np.zeros(3)])
        R = expm(q * C * t)
        np.testing.assert_allclose(U, X[:3], atol=1e-10 * np.linalg.norm(U0))
        np.testing.assert_allclose(np.real(out["Fk"][i, :3, 0, 0, 0]), X[3:], atol=1e-10 * np.linalg.norm(U0))
        np.testing.assert_allclose(M2 / nn - np.outer(U, U), R @ np.diag(vt ** 2) @ R.T, atol=1e-10 * vt.max() ** 2)
    # perpendicular current is nonzero and the magnetic force does no work: K - K0 = W_em only
    assert np.linalg.norm(U0 - (U0 @ b) * b / 0.25) > 1e-2
    assert np.abs(out["ledger_defect"]).max() < 1e-12 * out["K"][0]
    assert np.ptp(out["K"]) > 1e-2 * out["K"][0]
