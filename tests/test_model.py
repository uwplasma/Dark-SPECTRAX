"""Operator-level checks: 3V moments and Lorentz operator by independent quadrature (A00),
zero-mixing parent regression (A01), static Yukawa field (A05)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import hermite as H

import darkspectrax as ds
from spectrax._simulation import ode_system as parent_ode_system

ORD = (5, 4, 3)  # unequal (Nn, Nm, Np) catch axis permutations
A_S = (0.7, 1.1, 0.9, 0.35, 0.25, 0.45)
U_S = (0.3, -0.2, 0.5, -0.1, 0.15, 0.05)


def _psi(n, xi):
    c = np.zeros(n + 1)
    c[n] = 1.0
    return np.exp(-xi ** 2) * H.hermval(xi, c) / math.sqrt(math.pi * 2.0 ** n * math.factorial(n))


def _h(n, xi):
    c = np.zeros(n + 1)
    c[n] = 1.0
    return H.hermval(xi, c) / math.sqrt(2.0 ** n * math.factorial(n))


def _dpsi(n, xi):
    """d psi_n / d xi from the polynomial form, independent of ladder identities."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    P, dP = H.hermval(xi, c), H.hermval(xi, H.hermder(c))
    return np.exp(-xi ** 2) * (dP - 2 * xi * P) / math.sqrt(math.pi * 2.0 ** n * math.factorial(n))


def _homogeneous_model(**kw):
    base = dict(Nn=ORD[0], Nm=ORD[1], Np=ORD[2], qs=(-1.0, 2.0), Omega_cs=(1.3, 0.13),
                alpha_s=A_S, u_s=U_S)
    base.update(kw)
    return ds.Model(**base)


def _random_coeffs(model, seed=0):
    rng = np.random.default_rng(seed)
    Ck = rng.normal(size=(model.Ns * model.Nn * model.Nm * model.Np, 1, 1, 1)) * 0.3
    return jnp.asarray(Ck + 0j)


def _quad(model, s, Ck, func, nq=24):
    """Integral of func(v) f_s(v) d^3v by tensor Gauss-Hermite quadrature."""
    xg, wg = H.hermgauss(nq)
    a = np.reshape(model.alpha_s, (-1, 3))[s]
    u = np.reshape(model.u_s, (-1, 3))[s]
    X, Y, Z = np.meshgrid(xg, xg, xg, indexing="ij")
    W = wg[:, None, None] * wg[None, :, None] * wg[None, None, :]
    f = np.zeros_like(X)
    Hs = model.Nn * model.Nm * model.Np
    for p in range(model.Np):
        for m in range(model.Nm):
            for n in range(model.Nn):
                c = float(np.real(Ck[s * Hs + ds.hermite_index(n, m, p, model.Nn, model.Nm), 0, 0, 0]))
                f += c * _psi(n, X) * _psi(m, Y) * _psi(p, Z)
    f *= np.exp(X ** 2 + Y ** 2 + Z ** 2)  # undo the quadrature weight
    V = [u[0] + a[0] * X, u[1] + a[1] * Y, u[2] + a[2] * Z]
    return np.sum(W * func(V) * f) * a.prod()


def test_a00_moments_match_quadrature():
    model = _homogeneous_model()
    Ck = _random_coeffs(model)
    n, M, M2 = ds.moments(model, Ck)
    for s in range(model.Ns):
        assert np.isclose(n[s, 0, 0, 0].real, _quad(model, s, Ck, lambda V: 1.0), atol=1e-13)
        for i in range(3):
            assert np.isclose(M[s, i, 0, 0, 0].real, _quad(model, s, Ck, lambda V: V[i]), atol=1e-13)
            for j in range(3):
                ref = _quad(model, s, Ck, lambda V: V[i] * V[j])
                assert np.isclose(M2[s, i, j, 0, 0, 0].real, ref, atol=1e-12)
    # Full pressure tensor and kinetic energy (masses from Omega_cs ratios).
    y = {**model.zeros(), "Ck": Ck}
    K = ds.energies(model, y)["K_species"]
    for s in range(model.Ns):
        ms = model.masses[s]
        ref = 0.5 * ms * _quad(model, s, Ck, lambda V: V[0] ** 2 + V[1] ** 2 + V[2] ** 2)
        assert np.isclose(K[s], ref, rtol=1e-12)
        ns_ = n[s, 0, 0, 0].real
        P = ms * (M2[s, :, :, 0, 0, 0].real - np.outer(M[s, :, 0, 0, 0].real, M[s, :, 0, 0, 0].real) / ns_)
        Pref = np.array([[ms * (_quad(model, s, Ck, lambda V: V[i] * V[j])
                                - _quad(model, s, Ck, lambda V: V[i]) * _quad(model, s, Ck, lambda V: V[j]) / ns_)
                          for j in range(3)] for i in range(3)])
        assert np.allclose(P, Pref, atol=1e-12)


@pytest.mark.parametrize("seed", [1, 2])
def test_a00_lorentz_operator_by_quadrature(seed):
    """Galerkin projection of -(q/m)(E + v x B).grad_v f for each field component."""
    model = _homogeneous_model()
    Ck = _random_coeffs(model, seed)
    rng = np.random.default_rng(10 + seed)
    for comp in range(6):  # each E component and each B component (both Levi-Civita terms)
        F = np.zeros(6)
        F[comp] = rng.uniform(0.5, 1.5)
        y = {**model.zeros(), "Ck": Ck, "Fk": model.zeros()["Fk"].at[:, 0, 0, 0].set(F)}
        dC = np.asarray(ds.rhs(0.0, y, model)["Ck"])
        xg, wg = H.hermgauss(24)
        X, Y, Z = np.meshgrid(xg, xg, xg, indexing="ij")
        W = wg[:, None, None] * wg[None, :, None] * wg[None, None, :] * np.exp(X ** 2 + Y ** 2 + Z ** 2)
        Hs = model.Nn * model.Nm * model.Np
        for s in range(model.Ns):
            a = np.reshape(model.alpha_s, (-1, 3))[s]
            u = np.reshape(model.u_s, (-1, 3))[s]
            V = [u[0] + a[0] * X, u[1] + a[1] * Y, u[2] + a[2] * Z]
            grad = [np.zeros_like(X) for _ in range(3)]
            for p in range(model.Np):
                for m in range(model.Nm):
                    for n in range(model.Nn):
                        c = float(Ck[s * Hs + ds.hermite_index(n, m, p, model.Nn, model.Nm), 0, 0, 0].real)
                        grad[0] += c * _dpsi(n, X) * _psi(m, Y) * _psi(p, Z) / a[0]
                        grad[1] += c * _psi(n, X) * _dpsi(m, Y) * _psi(p, Z) / a[1]
                        grad[2] += c * _psi(n, X) * _psi(m, Y) * _dpsi(p, Z) / a[2]
            E, B = F[:3], F[3:]
            force = [E[0] + V[1] * B[2] - V[2] * B[1], E[1] + V[2] * B[0] - V[0] * B[2],
                     E[2] + V[0] * B[1] - V[1] * B[0]]
            qm = model.qs[s] * model.Omega_cs[s]  # normalized q_s/m_s times Omega_cs[0]
            rhs_v = -qm * sum(force[i] * grad[i] for i in range(3))
            for p in range(model.Np):
                for m in range(model.Nm):
                    for n in range(model.Nn):
                        ref = np.sum(W * rhs_v * _h(n, X) * _h(m, Y) * _h(p, Z))
                        got = dC[s * Hs + ds.hermite_index(n, m, p, model.Nn, model.Nm), 0, 0, 0]
                        assert abs(got - ref) < 1e-12 * (1 + abs(ref)), (comp, s, n, m, p)


def _random_state(model, seed=3, amp=1e-2):
    """Real physical fields projected to the parent's active band."""
    rng = np.random.default_rng(seed)
    mask = model.p["mask23"]
    shp = (model.Ny, model.Nx, model.Nz)

    def fourier(nc):
        f = rng.normal(size=(nc, *shp))
        return jnp.fft.rfftn(f, axes=(-1, -3, -2), norm="forward") * mask

    Ck = fourier(model.Ns * model.Nn * model.Nm * model.Np) * amp
    H = model.Nn * model.Nm * model.Np
    Ck = Ck.at[::H, 0, 0, 0].add(1.0)
    return {**model.zeros(), "Ck": Ck, "Fk": fourier(6) * amp, "Dk": fourier(10) * amp}


def _parent_rhs(model, y):
    p = model.p
    args = (None,) * 7 + (p["qs"], p["nu"], p["D"], p["Omega_cs"], p["alpha_s"], p["u_s"],
                          p["Lx"], p["Ly"], p["Lz"], p["kx_grid"], p["ky_grid"], p["kz_grid"],
                          p["k2_grid"], p["nabla"], p["collision_matrix"], p["sqrt_n_plus"],
                          p["sqrt_n_minus"], p["sqrt_m_plus"], p["sqrt_m_minus"], p["sqrt_p_plus"],
                          p["sqrt_p_minus"])
    flat = jnp.concatenate([y["Ck"].ravel(), y["Fk"].ravel()])
    out = parent_ode_system(model.Nx, model.Ny, model.Nz, model.Nn, model.Nm, model.Np, model.Ns,
                            0.0, flat, args)
    nC = y["Ck"].size
    return out[:nC].reshape(y["Ck"].shape), out[nC:].reshape(y["Fk"].shape)


@pytest.mark.parametrize("mode", ["ordinary", "self_consistent"])
def test_a01_zero_mixing_matches_parent_rhs(mode):
    model = ds.Model(Nx=8, Ny=5, Nz=4, Nn=4, Nm=3, Np=2, Lx=2.1, Ly=1.7, Lz=3.3,
                     qs=(-1.0, -1.0, 1.0), Omega_cs=(1.0, 1.0, 0.05),
                     alpha_s=(0.2, 0.25, 0.3, 0.15, 0.2, 0.1, 0.05, 0.06, 0.04),
                     u_s=(0.1, -0.05, 0.02, -0.1, 0.03, 0.0, 0.0, 0.01, -0.02),
                     nu=0.5, mode=mode, eta=0.0, Omega_D=0.8)
    y = _random_state(model)
    r = ds.rhs(0.0, y, model)
    dC, dF = _parent_rhs(model, y)
    scale = max(float(jnp.max(jnp.abs(dC))), 1.0)
    assert float(jnp.max(jnp.abs(r["Ck"] - dC))) <= 1e-14 * scale
    assert float(jnp.max(jnp.abs(r["Fk"] - dF))) <= 1e-14 * max(float(jnp.max(jnp.abs(dF))), 1.0)
    if mode == "self_consistent":
        # Zero mixing: the dark sector evolves as free Proca and does not feed back.
        y2 = {**y, "Dk": 7.0 * y["Dk"]}
        r2 = ds.rhs(0.0, y2, model)
        assert jnp.array_equal(r2["Ck"], r["Ck"]) and jnp.array_equal(r2["Fk"], r["Fk"])
        assert float(r["W"][1].real) == 0.0


def test_a01_zero_mixing_trajectory_matches_parent():
    from spectrax import simulation

    model = ds.Model(Nx=8, Nn=10, Lx=2 * np.pi / 2.5, qs=(-1.0, 1.0), Omega_cs=(1.0, 1.0 / 25),
                     alpha_s=(0.14, 0.14, 0.14, 0.03, 0.03, 0.03), u_s=(0.0,) * 6,
                     mode="self_consistent", eta=0.0)
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0, 1.0], [(0, (1, 0, 0), 0.01j)]))
    out = ds.run(model, y, 3.0, n_save=4, rtol=1e-11, atol=1e-13)
    par = simulation({"Ck_0": y["Ck"], "Fk_0": y["Fk"], "qs": jnp.array(model.qs),
                      "Omega_cs": jnp.array(model.Omega_cs), "alpha_s": jnp.array(model.alpha_s),
                      "u_s": jnp.array(model.u_s), "Lx": model.Lx, "nu": 0.0, "D": 0.0,
                      "t_max": 3.0, "ode_tolerance": 1e-12},
                     Nx=8, Ny=1, Nz=1, Nn=10, Nm=1, Np=1, Ns=2, timesteps=4)
    assert np.allclose(out["Fk"], np.asarray(par["Fk"]), atol=1e-11)
    assert np.allclose(out["Ck"], np.asarray(par["Ck"]), atol=1e-10)


@pytest.mark.parametrize("Omega_D", [0.5, 2.0])
def test_a05_static_yukawa(Omega_D):
    eta, k, rho0 = 0.4, 1.5, 0.02
    model = ds.Model(Nx=8, Nn=3, Lx=2 * np.pi / k, alpha_s=(0.1, 0.1, 0.1), rho_background=1.0,
                     Omega_cs=(1.7,), mode="self_consistent", eta=eta, Omega_D=Omega_D)
    # electron density n = 1 - rho0 cos(kx)  ->  rho = rho0 cos(kx)
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), -rho0 / 2)]))
    g_ord, g_dark = ds.gauss_residuals(model, y)
    assert g_ord < 1e-16 and g_dark < 1e-16
    r = ds.rhs(0.0, y, model)
    assert float(jnp.max(jnp.abs(r["Dk"]))) < 1e-17  # static: J = 0 and fields stationary
    # Independent fields: E = rho0 sin(kx)/(Om0 k), eta E_D = eta^2 rho0 k sin(kx)/(Om0 (k^2+M^2)).
    om0 = model.Omega_cs[0]
    Ex = np.fft.irfft(np.asarray(y["Fk"][0, 0, :, 0]) * 8, n=8)
    EDx = np.fft.irfft(np.asarray(y["Dk"][0, 0, :, 0]) * 8, n=8)
    x = np.arange(8) * model.Lx / 8
    assert np.allclose(Ex, rho0 * np.sin(k * x) / (om0 * k), atol=1e-16)
    assert np.allclose(eta * EDx, eta ** 2 * rho0 * k * np.sin(k * x) / (om0 * (k ** 2 + Omega_D ** 2)),
                       atol=1e-16)
    # potential energy term is part of U_D
    U = ds.energies(model, y)["U_D"]
    phi = eta * rho0 / (2 * om0 * (k ** 2 + Omega_D ** 2))
    ref = 0.5 * om0 ** 2 * 2 * (k ** 2 * phi ** 2 + Omega_D ** 2 * phi ** 2)
    assert np.isclose(float(U), ref, rtol=1e-12)


def test_model_validation_errors():
    with pytest.raises(ValueError, match="mode"):
        ds.Model(mode="dark")
    with pytest.raises(ValueError, match="inconsistent"):
        ds.Model(qs=(-1.0, 1.0))
    with pytest.raises(ValueError, match="E_drive"):
        ds.Model(E_drive=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="eta"):
        ds.Model(eta=0.1)
    m = ds.Model(Nx=4, Lx=1.0, alpha_s=(0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="neutral"):
        ds.consistent_fields(m, ds.maxwellian(m, [1.0]))
    with pytest.raises(ValueError, match="ix > 0"):
        ds.proca_mode(ds.Model(Ny=4, mode="self_consistent"), ds.Model(Ny=4).zeros(), (0, 1, 0),
                      [1.0, 0, 0])


def test_kx0_perturbation_keeps_conjugate_pair():
    model = ds.Model(Nx=1, Ny=4, Ly=2.0, alpha_s=(0.1, 0.2, 0.1), rho_background=1.0)
    y = ds.maxwellian(model, [1.0], [(0, (0, 1, 0), 0.01 + 0.02j)])
    n = np.fft.irfftn(np.asarray(y["Ck"][0]) * 4, s=(1, 4, 1), axes=(-1, -3, -2)) * 0.1 * 0.2 * 0.1
    yy = np.arange(4) * 2.0 / 4
    ref = 1.0 + 2 * np.real((0.01 + 0.02j) * np.exp(2j * np.pi * yy / 2.0))
    assert np.allclose(n[:, 0, 0], ref, atol=1e-15)
    yc = ds.consistent_fields(model, y)
    assert ds.gauss_residuals(model, yc)[0] < 1e-16
