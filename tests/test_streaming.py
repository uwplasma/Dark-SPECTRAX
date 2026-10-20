"""Exact Hermite streaming integrating factor in Dark-SPECTRAX: generator equals the RHS streaming part (fixed and pump
basis), bitwise-off default, agreement with the plain solve, step-size advantage on pure streaming, gradient."""

import jax
import jax.numpy as jnp
import numpy as np

import darkspectrax as ds
from darkspectrax._streaming import streaming_factor_fn


def _case(frame="fixed", Nn=24, nu=0.0):
    m = ds.Model(Nx=8, Lx=2 * np.pi, Nn=Nn, alpha_s=(0.5, 0.5, 0.5), u_s=(0.3, 0.0, 0.0), rho_background=1.0, nu=nu,
                 frame=frame)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 2e-3), (0, (3, 0, 0), 1e-3)]))
    return m, y


def test_generator_is_the_streaming_part_of_the_rhs_fixed_and_pump_basis():
    for frame in ("fixed", "pump"):
        m, y = _case(frame)
        rng = np.random.default_rng(0)
        Ck = jnp.asarray(rng.normal(size=y["Ck"].shape) + 1j * rng.normal(size=y["Ck"].shape))
        y = {**y, "Ck": Ck, "Fk": jnp.zeros_like(y["Fk"]), "Dk": jnp.zeros_like(y["Dk"])}
        if frame == "pump":
            y["B"] = jnp.array([[0.2, 0, 0], [0.7, 0.6, 0.9]], complex)  # a moving basis different from the model's
        fac = streaming_factor_fn(m)(0.0, y, None)
        ref = ds.rhs(0.0, y, m)["Ck"]  # F = 0, nu = 0: the kinetic part is free streaming
        got = fac.streaming(fac.to_ck(y)).reshape(ref.shape)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_default_bitwise_and_streaming_agrees_with_plain():
    m, y = _case(Nn=20, nu=2.0)
    kw = dict(n_save=5, rtol=1e-10, atol=1e-12)
    plain = ds.run(m, y, 4.0, **kw)
    again = ds.run(m, y, 4.0, integrating_factor=None, **kw)
    assert np.array_equal(plain["Ck"], again["Ck"])
    for mode in (ds.streaming(), ds.streaming(True)):
        lw = ds.run(m, y, 4.0, integrating_factor=mode, **kw)
        assert lw["status"] == "success"
        assert np.abs(lw["Ck"] - plain["Ck"]).max() < 1e-6 * np.abs(plain["Ck"]).max()
        # the hypercollision dissipates energy outside the ledger: the defect must equal the plain run's
        assert np.abs(lw["ledger_defect"] - plain["ledger_defect"]).max() < 1e-9
        # density row of the k = 0 mode: the factor is the identity there
        assert abs(lw["Ck"][-1][0, 0, 0, 0] - lw["Ck"][0][0, 0, 0, 0]) < 1e-11


def test_streaming_gradient_matches_finite_difference():
    m, y = _case(Nn=12)

    def loss(x):
        ys = ds.solve(m, y, 1.0, 2, params={"alpha_s": x * jnp.array([1.0, 1.0, 1.0])}, rtol=1e-9, atol=1e-11,
                      integrating_factor=ds.streaming())
        return jnp.real(ys["W"][-1, 0]) + jnp.sum(jnp.abs(ys["Ck"][-1]) ** 2)
    x0 = jnp.array(0.5)
    g = float(jax.grad(loss)(x0))
    fd = (float(loss(x0 + 1e-6)) - float(loss(x0 - 1e-6))) / 2e-6
    assert np.isfinite(g) and abs(g - fd) < 1e-4 * max(abs(fd), 1e-12) + 1e-9
