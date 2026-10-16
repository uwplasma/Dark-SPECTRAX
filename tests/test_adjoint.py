"""C09: the differentiable solve and traced parameters (AD vs centred finite differences)."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds


def _model():
    return ds.Model(Nx=1, Nn=3, Nm=3, Np=3, alpha_s=(0.05,) * 3, rho_background=1.0,
                    mode="self_consistent", eta=0.25, Omega_D=1.2)


def _wd(x, adjoint=None, fixed_dt=0.05):
    m = _model()
    y = m.zeros()
    y = {**y, "Ck": y["Ck"].at[0, 0, 0, 0].set(1 / 0.05 ** 3), "Fk": y["Fk"].at[0, 0, 0, 0].set(0.01)}
    ys = ds.solve(m, y, 6.0, 2, params={"eta": x[0], "Omega_D": x[1]}, fixed_dt=fixed_dt, max_steps=200,
                  adjoint=adjoint)
    return jnp.real(ys["W"][-1, 1])


def test_solve_matches_run_and_grad_matches_fd():
    m = _model()
    x = jnp.array([0.25, 1.2])
    out = ds.run(m, {**ds.maxwellian(m, [1.0]), "Fk": m.zeros()["Fk"].at[0, 0, 0, 0].set(0.01)}, 6.0, n_save=2,
                 fixed_dt=0.05)
    assert abs(float(_wd(x)) - out["W"][-1, 1]) < 1e-15
    g = np.asarray(jax.grad(_wd)(x))
    fwd = [float(jax.jvp(lambda z: _wd(z, diffrax.ForwardMode()), (x,), (e,))[1]) for e in jnp.eye(2)]
    fd = [(float(_wd(x + h * e)) - float(_wd(x - h * e))) / (2 * h) for e, h in zip(jnp.eye(2), (1e-5, 1e-5))]
    assert np.allclose(g, fwd, rtol=1e-10) and np.allclose(g, fd, rtol=1e-6)
    ga = jax.grad(lambda z: _wd(z, fixed_dt=None))(x)  # adaptive PID path
    assert np.allclose(ga, g, rtol=1e-4)


def test_with_params_alpha_and_rejects_unknown():
    m = _model()
    m2 = ds.with_params(m, alpha_s=(0.06,) * 3, u_s=(0.0,) * 3)
    assert float(m2.p["alpha_s"][0]) == 0.06 and float(m.p["alpha_s"][0]) == 0.05 and m2.eta == m.eta
    with pytest.raises(ValueError, match="not traceable"):
        ds.with_params(m, Nn=4)
