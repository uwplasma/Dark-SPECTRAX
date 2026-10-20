"""Integrating-factor (Lawson) solve of the diagonal closure: bitwise default, zero damping, accuracy, speed, ledger, gradient."""

import jax
import jax.numpy as jnp
import numpy as np

import darkspectrax as ds


def _case(nu=0.0, Nn=16):
    m = ds.Model(Nx=8, Lx=2 * np.pi, Nn=Nn, alpha_s=(0.1, 0.1, 0.1), rho_background=1.0, nu=nu)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 2e-3)]))
    return m, y


def test_default_off_and_zero_damping_are_bitwise_the_plain_solve():
    m, y = _case(nu=0.0)
    a = ds.run(m, y, 6.0, n_save=7, rtol=1e-10, atol=1e-12)
    b = ds.run(m, y, 6.0, n_save=7, rtol=1e-10, atol=1e-12, integrating_factor=False)
    c = ds.run(m, y, 6.0, n_save=7, rtol=1e-10, atol=1e-12, integrating_factor=True)  # nu = 0: rate zero
    for k in ("Ck", "Fk", "W"):
        assert np.array_equal(a[k], b[k])
        # zero rate: the Lawson step is bitwise the wrapped step (tests/test_lawson.py in SPECTRAX); only the
        # step clipping onto the saves differs here, so the paths agree to the solver tolerance
        assert np.abs(a[k] - c[k]).max() < 1e-9 * np.abs(a[k]).max()


def test_static_hypercollision_exact_accuracy_ledger_and_fewer_steps():
    m, y = _case(nu=3000.0)  # nu * col reaches 3e3: h_stab ~ 1.7e-3 for the plain Dopri8 step
    plain = ds.run(m, y, 6.0, n_save=7, rtol=1e-10, atol=1e-12)
    lw = ds.run(m, y, 6.0, n_save=7, rtol=1e-10, atol=1e-12, integrating_factor=True)
    assert plain["status"] == lw["status"] == "success"
    scale = np.abs(plain["Ck"]).max()
    assert np.abs(lw["Ck"] - plain["Ck"]).max() < 1e-7 * scale
    assert np.abs(lw["ledger_defect"]).max() < 1e-9
    assert lw["num_steps"] * 3 < plain["num_steps"]
    # density, momentum and energy rows are outside the exponent (col = 0 there): charge is kept to round-off
    assert abs(lw["Ck"][-1][0, 0, 0, 0] - lw["Ck"][0][0, 0, 0, 0]) < 1e-13


def test_field_closure_rate_matches_rhs_damping_and_zero_strength():
    m, y = _case()
    y = {**y, "Ck": y["Ck"] + 1e-3 * (1 + 1j) * jnp.ones_like(y["Ck"])}
    rhs_fn, rate_fn = ds.field_closure(m, 4.0)
    d = rhs_fn(0.0, y, m)["Ck"] - ds.rhs(0.0, y, m)["Ck"]
    rate = rate_fn(0.0, y, m)
    assert set(rate) == set(y) and rate["Fk"] == 0.0 and rate["W"] == 0.0
    np.testing.assert_allclose(d, -rate["Ck"] * y["Ck"], rtol=1e-13, atol=0)
    assert float(rate["Ck"][:3].max()) == 0.0 and float(rate["Ck"].max()) > 0  # rows n <= 2 untouched
    _, rate0 = ds.field_closure(m, 0.0)
    assert float(jnp.abs(rate0(0.0, y, m)["Ck"]).max()) == 0.0


def test_field_closure_lawson_run_agrees_with_plain_and_gradient_smoke():
    m, y = _case()
    rhs_fn, rate_fn = ds.field_closure(m, 2000.0)
    old, ds._simulation.rhs = ds._simulation.rhs, rhs_fn
    try:
        plain = ds.run(m, y, 4.0, n_save=5, rtol=1e-10, atol=1e-12)
        lw = ds.run(m, y, 4.0, n_save=5, rtol=1e-10, atol=1e-12, integrating_factor=rate_fn)
        assert np.abs(lw["Ck"] - plain["Ck"]).max() < 1e-6 * np.abs(plain["Ck"]).max()
        assert lw["num_steps"] <= plain["num_steps"]

        def loss(x):
            ys = ds.solve(m, y, 2.0, 2, params={"alpha_s": x * jnp.array([1.0, 1.0, 1.0])}, rtol=1e-9, atol=1e-11,
                          integrating_factor=rate_fn)
            return jnp.real(ys["W"][-1, 0])
        x0 = jnp.array(0.1)
        g = float(jax.grad(loss)(x0))
        fd = (float(loss(x0 + 1e-6)) - float(loss(x0 - 1e-6))) / 2e-6
        assert np.isfinite(g) and abs(g - fd) < 1e-3 * max(abs(fd), 1e-12) + 1e-9
    finally:
        ds._simulation.rhs = old
