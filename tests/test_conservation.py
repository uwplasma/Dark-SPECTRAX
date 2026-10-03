"""C00 semidiscrete identities on random constraint-satisfying states; C07 restart/chunking."""

import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds


def _model(**kw):
    base = dict(Nx=8, Ny=5, Nz=1, Nn=4, Nm=3, Np=3, Lx=2.3, Ly=1.9, qs=(-1.0, 1.0), Omega_cs=(1.3, 0.13),
                alpha_s=(0.2, 0.25, 0.22, 0.07, 0.08, 0.06), u_s=(0.05, -0.02, 0.01, 0.0, 0.01, -0.01),
                rho_background=0.0, mode="self_consistent", eta=0.35, Omega_D=0.9)
    base.update(kw)
    return ds.Model(**base)


def _random_constrained(model, seed):
    """Random band-limited state with both Gauss laws and B = curl-like fields satisfied exactly."""
    rng = np.random.default_rng(seed)
    mask = model.p["mask23"]
    nab = model.p["nabla"]
    shp = (model.Ny, model.Nx, model.Nz)

    def rnd(nc, amp):
        f = rng.normal(size=(nc, *shp)) * amp
        return jnp.fft.rfftn(f, axes=(-1, -3, -2), norm="forward") * mask

    H = model.Nn * model.Nm * model.Np
    Ck = rnd(model.Ns * H, 1e-2)
    a = np.reshape(model.alpha_s, (-1, 3)).prod(axis=1)
    Ck = Ck.at[0, 0, 0, 0].set(1.0 / a[0]).at[H, 0, 0, 0].set(1.0 / a[1])     # neutral: n_e = n_i = 1
    y = {**model.zeros(), "Ck": Ck}
    y = ds.consistent_fields(model, y, E_mean=tuple(rng.normal(size=3) * 1e-2))
    k2 = jnp.sum(nab ** 2, axis=0)
    safe = jnp.where(k2 > 0, k2, 1.0)

    def transverse(v):  # remove the longitudinal part of a random vector field (k != 0)
        return v - nab * jnp.where(k2 > 0, jnp.sum(nab * v, axis=0) / safe, 0.0)

    Fk = y["Fk"].at[:3].add(transverse(rnd(3, 1e-2))).at[3:].set(transverse(rnd(3, 1e-2)))
    AD, phi = rnd(3, 1e-2), rnd(1, 1e-2)[0]
    rho = ds.charge_density(model, Ck) / model.Omega_cs[0]
    ED_L = -1j * nab * jnp.where(k2 > 0, (model.eta * rho - model.Omega_D ** 2 * phi) / safe, 0.0)
    phi = phi.at[0, 0, 0].set(model.eta * rho[0, 0, 0] / model.Omega_D ** 2)
    ED = ED_L + transverse(rnd(3, 1e-2))
    BD = 1j * jnp.stack([nab[1] * AD[2] - nab[2] * AD[1], nab[2] * AD[0] - nab[0] * AD[2],
                         nab[0] * AD[1] - nab[1] * AD[0]])
    Dk = jnp.concatenate([ED, BD, AD, phi[None]])
    return {**y, "Fk": Fk, "Dk": Dk}


def _ddt(model, y, dy, key):
    """Directional derivative of an energy along the RHS (energies are affine/quadratic)."""
    h = 1.0  # central difference is exact for affine K and quadratic U
    yp = {k: y[k] + h * dy[k] for k in y}
    ym = {k: y[k] - h * dy[k] for k in y}
    return (float(ds.energies(model, yp)[key]) - float(ds.energies(model, ym)[key])) / (2 * h)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c00_energy_work_continuity_gauss(seed):
    m = _model()
    y = _random_constrained(m, seed)
    g = ds.gauss_residuals(m, y)
    assert g[0] < 1e-15 and g[1] < 1e-15
    dy = ds.rhs(0.3, y, m)
    P_em, P_D = float(dy["W"][0].real), float(dy["W"][1].real)
    dK, dU, dUD = (_ddt(m, y, dy, k) for k in ("K", "U_gamma", "U_D"))
    scale = abs(P_em) + abs(P_D)
    assert abs(dK - P_em - P_D) < 1e-11 * scale          # particle work theorem (exact central difference)
    assert abs(dU + P_em) < 1e-11 * scale               # Poynting (periodic: no flux)
    assert abs(dUD + P_D) < 1e-11 * scale               # Proca energy including the potential terms
    # continuity and both constraints are preserved by the RHS
    n, M, _ = ds.moments(m, y["Ck"])
    dn, _, _ = ds.moments(m, dy["Ck"])
    q = jnp.asarray(m.qs)
    nab = m.p["nabla"]
    J = jnp.tensordot(q, M, axes=1)
    drho = jnp.tensordot(q, dn, axes=1)
    assert float(jnp.max(jnp.abs(drho + 1j * jnp.sum(nab * J, axis=0)))) < 1e-14
    rho_dot = drho / m.Omega_cs[0]
    gE = 1j * jnp.sum(nab * dy["Fk"][:3], axis=0) - rho_dot
    gD = 1j * jnp.sum(nab * dy["Dk"][:3], axis=0) + m.Omega_D ** 2 * dy["Dk"][9] - m.eta * rho_dot
    assert float(jnp.max(jnp.abs(gE))) < 1e-14 and float(jnp.max(jnp.abs(gD))) < 1e-14
    BD = dy["Dk"][3:6]
    curlA = 1j * jnp.stack([nab[1] * dy["Dk"][8] - nab[2] * dy["Dk"][7], nab[2] * dy["Dk"][6] - nab[0] * dy["Dk"][8],
                            nab[0] * dy["Dk"][7] - nab[1] * dy["Dk"][6]])
    assert float(jnp.max(jnp.abs(BD - curlA))) < 1e-14     # d/dt (B_D - curl A_D) = 0


def test_c00_magnetic_force_does_no_work():
    m = _model()
    y = _random_constrained(m, 5)
    y = {**y, "Fk": y["Fk"].at[:3].set(0.0), "Dk": y["Dk"].at[:3].set(0.0)}
    dy = ds.rhs(0.0, y, m)
    assert abs(_ddt(m, y, dy, "K")) < 1e-12 and float(dy["W"][0].real) == 0.0


def test_c07_chunked_restart_matches_single_run():
    m = ds.Model(Nx=8, Nn=10, Lx=2 * np.pi / 2.0, alpha_s=(0.12, 0.12, 0.12), rho_background=1.0,
                 mode="prescribed_drive", E_drive=(1e-3, 0.0, 0.0), omega_drive=1.1, phase_drive=0.3)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 0.01)]))
    full = ds.run(m, y, 6.0, n_save=7, rtol=1e-12, atol=1e-14)
    a = ds.run(m, y, 3.0, n_save=4, rtol=1e-12, atol=1e-14)
    last = {"Ck": a["Ck"][-1], "Fk": a["Fk"][-1], "Dk": a["Dk"][-1], "W": a["W"][-1].astype(complex)}
    b = ds.run(m, last, 3.0, n_save=4, rtol=1e-12, atol=1e-14, t0=3.0)
    assert np.allclose(b["t"], full["t"][3:])
    assert np.max(np.abs(b["Fk"] - full["Fk"][3:])) < 1e-12
    assert np.max(np.abs(b["W"] - full["W"][3:])) < 1e-13
    led_full = full["K"] + full["U_gamma"] - full["W"][:, 2]
    led_b = b["K"] + b["U_gamma"] - b["W"][:, 2]
    assert np.max(np.abs(led_b - led_full[0])) < 1e-12        # ledger continues across the restart
    assert np.abs(b["ledger_defect"]).max() < 1e-12           # segment ledger closes from its own start
