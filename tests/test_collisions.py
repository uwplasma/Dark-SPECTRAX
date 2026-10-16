"""Companion option for the parent's conservative Dougherty operator (SPECTRAX #71)."""
import numpy as np
import pytest

from darkspectrax import Model, consistent_fields, maxwellian, run
from darkspectrax._model import rhs


def landau(**kw):
    a = 0.1 * np.sqrt(2)
    m = Model(Nx=5, Nn=24, Lx=2 * np.pi / 5, alpha_s=(a, a, a), rho_background=1.0, **kw)
    y = consistent_fields(m, maxwellian(m, [1.0], [(0, (1, 0, 0), 1e-3)]))
    return m, y


def test_off_is_bitwise_the_collisionless_rhs():
    m0, y = landau()
    m1, _ = landau(collision_model="dougherty", nu_dougherty=0.0)
    a, b = rhs(0.0, y, m0), rhs(0.0, y, m1)
    np.testing.assert_allclose(np.asarray(b["Ck"]), np.asarray(a["Ck"]), atol=1e-15)
    a2 = rhs(0.0, y, landau(collision_model="none", nu_dougherty=3.0)[0])
    assert np.array_equal(np.asarray(a2["Ck"]), np.asarray(a["Ck"]))


def test_collisional_landau_run_keeps_the_ledger():
    m0, y = landau()
    m1, _ = landau(collision_model="dougherty", nu_dougherty=0.1)
    r0, r1 = run(m0, y, 10.0, n_save=11), run(m1, y, 10.0, n_save=11)
    assert r1["status"] == "success"
    assert np.abs(r1["ledger_defect"]).max() < 1e-9
    assert np.abs(r1["U_gamma"] - r0["U_gamma"]).max() > 1e-3 * r0["U_gamma"].max()  # the term acts
    assert abs(r1["K"][-1] + r1["U_gamma"][-1] - r1["K"][0] - r1["U_gamma"][0]) < 1e-9 * r1["K"][0]


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        Model(collision_model="bgk")
    with pytest.raises(ValueError):
        Model(nu_dougherty=(0.1, 0.2))
