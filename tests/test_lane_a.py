"""Lane A regularization: continuous width growth (never shrinks) and the n > 2 exponential filter."""

import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds


def _m(**kw):
    base = dict(Nx=8, Nn=16, Lx=2 * np.pi, alpha_s=(0.2, 0.2, 0.2), rho_background=1.0,
                mode="prescribed_drive", E_drive=(0.05, 0.0, 0.0), omega_drive=1.0, frame="pump")
    return ds.Model(**{**base, **kw})


def _y0(model):
    return ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), 0.05)]))


def test_width_rate_is_zero_above_the_floor_and_grows_below_it():
    m = _m(width_floor=1.1)
    y = _y0(m)  # Maxwellian with a = sqrt(2) sigma > 1.1 sigma
    assert float(jnp.max(ds.width_rate(m, y["Ck"], y["B"]))) == 0.0
    m2 = _m(width_floor=2.0, width_tau=5.0)
    r = np.asarray(ds.width_rate(m2, y["Ck"], y["B"]))
    np.testing.assert_allclose(r[0], (2.0 / np.sqrt(2) - 1) * 0.2 / 5.0, rtol=1e-12)
    assert np.all(r[1:] == 0)  # y, z axes have one Hermite mode: no measured variance


def test_continuous_widening_is_an_exact_change_of_variables():
    """Physical observables with and without width growth agree to the truncation level (exact in the
    untruncated limit: the width terms only lower the Hermite index); the width grows monotonically."""
    T = 10.0
    ref, grow = _m(Nn=40), _m(Nn=40, width_floor=2.0, width_tau=2.0)
    a = ds.run(ref, _y0(ref), T, n_save=6, rtol=1e-11, atol=1e-13)
    b = ds.run(grow, _y0(grow), T, n_save=6, rtol=1e-11, atol=1e-13)
    assert b["B"][-1, 1, 0] > 1.3 * b["B"][0, 1, 0] and np.all(np.diff(b["B"][:, 1, 0]) >= 0)
    for k in ("K", "U_gamma"):
        assert np.max(np.abs(a[k] - b[k])) < 1e-8 * np.max(np.abs(a[k]))
    assert np.max(np.abs(a["W"] - b["W"])) < 1e-8 * np.max(np.abs(a["W"]))


def test_filter_keeps_low_moments_and_is_the_collision_slot():
    m = _m(filter_rate=1.0, filter_order=8)
    s = np.asarray(m.p["collision_matrix"])
    assert np.all(s[..., :3] == 0) and s[0, 0, -1] == 1.0 and float(m.p["nu"]) == 1.0
    out = ds.run(m, _y0(m), 10.0, n_save=6, rtol=1e-11, atol=1e-13)
    led = out["K"] + out["U_gamma"] - out["W"][:, 2]
    assert np.max(np.abs(led - led[0])) < 1e-7 * out["W"][-1, 2]


def test_bad_combinations_are_refused():
    with pytest.raises(ValueError):
        _m(width_floor=1.1, frame="fixed")
    with pytest.raises(ValueError):
        _m(filter_rate=1.0, nu=1.0)
