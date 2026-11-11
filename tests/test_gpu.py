"""Small-size device smoke tests, selected with `pytest -m gpu` (run on a GPU runner in CI, valid on CPU too)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds

pytestmark = pytest.mark.gpu


def _case(Nn=8):
    m = ds.Model(Nx=8, Lx=2 * np.pi, Nn=Nn, alpha_s=(0.5, 0.5, 0.5), u_s=(0.3, 0.0, 0.0), rho_background=1.0, nu=0.0)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 2e-3)]))
    return m, y


def test_rhs_is_finite_and_jit_matches_eager():
    m, y = _case()
    eager = ds.rhs(0.0, y, m)
    jitted = jax.jit(lambda s: ds.rhs(0.0, s, m))(y)
    assert np.isfinite(np.asarray(eager["Ck"])).all()
    np.testing.assert_allclose(np.asarray(jitted["Ck"]), np.asarray(eager["Ck"]), rtol=1e-10, atol=1e-12)


def test_short_run_succeeds():
    m, y = _case()
    out = ds.run(m, y, 0.5, n_save=3, rtol=1e-8, atol=1e-10)
    assert out["status"] == "success"
    assert np.isfinite(np.asarray(out["Ck"])).all()


def test_gradient_matches_finite_difference():
    m, y = _case(Nn=6)

    def loss(x):
        ys = ds.solve(m, y, 0.5, 2, params={"alpha_s": x * jnp.ones(3)}, rtol=1e-8, atol=1e-10)
        return jnp.sum(jnp.abs(ys["Ck"][-1]) ** 2)
    x0 = jnp.array(0.5)
    g = float(jax.grad(loss)(x0))
    fd = (float(loss(x0 + 1e-5)) - float(loss(x0 - 1e-5))) / 2e-5
    assert np.isfinite(g) and abs(g - fd) < 1e-3 * max(abs(fd), 1e-12) + 1e-8
