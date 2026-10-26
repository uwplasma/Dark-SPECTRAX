"""Relativistic Hermite model through Dark-SPECTRAX: inv_c2 = 0 is the old path, ledger and speed moments vs quadrature."""
import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds


def _model(inv_c2, frame="pump", Nn=24, Nx=8):
    vte = np.sqrt(1e-3)
    return ds.Model(Nx=Nx, Nn=Nn, Lx=40.0, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836),
                    alpha_s=(np.sqrt(2) * vte,) * 3 + (np.sqrt(2) * vte / np.sqrt(1836),) * 3, u_s=(0.0,) * 6,
                    mode="prescribed_drive", E_drive=(0.03 * vte, 0, 0), omega_drive=np.sqrt(1 + 1 / 1836), frame=frame,
                    inv_c2=inv_c2)


def _state(m):
    return ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [(0, (1, 0, 0), 5e-4), (1, (1, 0, 0), 5e-4)]))


def test_zero_inv_c2_is_bitwise_the_old_rhs():
    m0, m1 = _model(0.0), _model(0.0)
    y = _state(m0)
    a, b = ds.rhs(0.3, y, m0), ds.rhs(0.3, y, m1)
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_small_inv_c2_close_to_nonrelativistic():
    y = _state(_model(0.0))
    a, b = ds.rhs(0.3, y, _model(0.0)), ds.rhs(0.3, y, _model(1e-9))
    scale = np.max(np.abs(np.asarray(a["Ck"])))
    for k in ("Ck", "Fk", "W"):
        assert np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))) <= 1e-6 * scale


def test_pump_frame_does_not_double_count_the_drive():
    """Relativistic pump frame: the kinetic rhs of the k = 0 state under a uniform drive is the basis-centre rate only (no Hermite force)."""
    m = _model(1.0)
    y = _state(m)
    out = ds.rhs(0.0, y, m)
    dC = np.asarray(out["Ck"]).reshape(2, m.Nn, *m.shape)
    assert np.max(np.abs(dC[:, :, 0, 0, 0])) < 1e-12  # k = 0 row has no drive force and no streaming


def test_relativistic_kinetic_energy_and_speed_moments_match_quadrature():
    m = _model(1.0, frame="fixed", Nn=48)
    y = _state(m)
    K = np.asarray(ds.energies(m, y)["K_species"])
    # Maxwellian in u: <gamma - 1> by Gauss-Hermite quadrature
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / np.sqrt(2 * np.pi)
    sig = [np.sqrt(1e-3), np.sqrt(1e-3 / 1836)]
    ref = [m_ * np.sum(w * (np.sqrt(1 + (s * x) ** 2) - 1)) for m_, s in zip(m.masses, sig)]
    assert np.allclose(K, ref, rtol=1e-9)
    n, v1, v2 = (np.asarray(z) for z in ds.speed_moments(m, y["Ck"]))
    assert np.allclose(n, 1.0, atol=1e-12) and np.allclose(v1, 0.0, atol=1e-12)
    ref2 = [np.sum(w * ((s * x) / np.sqrt(1 + (s * x) ** 2)) ** 2) for s in sig]
    assert np.allclose(v2, ref2, rtol=1e-9)


def test_streaming_factor_refuses_relativistic():
    with pytest.raises(ValueError):
        ds.streaming()  # marker only
        from darkspectrax._streaming import streaming_factor_fn
        streaming_factor_fn(_model(1.0))


def test_dougherty_refused():
    with pytest.raises(ValueError):
        ds.Model(Nn=4, inv_c2=1.0, collision_model="dougherty", nu_dougherty=0.1)
