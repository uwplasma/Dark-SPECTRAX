"""Time-domain controls: vacuum Proca (A02), coupled oscillator (A03), eta-sign symmetry
(A06), exact mean pump (A07), populations and grids (A09), ledger closure and the CLI."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import darkspectrax as ds
from darkspectrax.__main__ import main


def _max(a):
    return float(np.max(np.abs(a)))


@pytest.mark.parametrize("index,A", [((1, 0, 0), (0.2, 0.1j, -0.05)),    # mixed polarization
                                     ((1, 1, 0), (0.1, -0.2, 0.3j)),     # oblique k in x-y
                                     ((0, 0, 0), (0.1, 0.2j, 0.05))])    # k = 0 mass oscillation
def test_a02_vacuum_proca(index, A):
    OD, eta = 1.3, 0.4
    model = ds.Model(Nx=8, Ny=4, Lx=2 * np.pi, Ly=2 * np.pi / 1.5, qs=(0.0,), Nn=2,
                     mode="self_consistent", eta=eta, Omega_D=OD)
    y = ds.proca_mode(model, ds.maxwellian(model, [1.0]), index, A)
    out = ds.run(model, y, 6.0, n_save=13, rtol=1e-12, atol=1e-14)
    assert out["status"] == "success"
    ix, iy, iz = index
    k = np.asarray(model.p["nabla"][:, iy, ix, iz])
    w = np.sqrt(k @ k + OD ** 2)
    D0 = np.asarray(y["Dk"][:, iy, ix, iz])
    for t, D in zip(out["t"], out["Dk"][:, :, iy, ix, iz]):
        ref = D0 * np.exp(-1j * w * t) if ix else None
        if ix == 0:  # real mean field: A(t) = Re(A e^{-iwt}), E = -dA/dt
            Ac = np.asarray(A)
            ref = np.concatenate([np.real(1j * w * Ac * np.exp(-1j * w * t)), np.zeros(3),
                                  np.real(Ac * np.exp(-1j * w * t)), [0.0]])
        assert np.allclose(D, ref, atol=1e-10)
        # B_D = i k x A_D and phi_D = k.A_D / w stay satisfied
        assert np.allclose(D[3:6], 1j * np.cross(k, D[6:9]), atol=1e-10)
    assert out["gauss"][:, 1].max() < 1e-10                      # dark Gauss law in vacuum
    U = out["U_D"]
    assert U[0] > 0 and _max(U - U[0]) < 1e-8 * U[0]            # complete energy constant
    if ix and abs(np.dot(k, A)) > 0:
        assert abs(D0[9]) > 0                                    # longitudinal part carries phi_D


def _homogeneous(eta, OD, drive=None):
    kw = dict(mode="prescribed_drive", E_drive=drive[0], omega_drive=drive[1]) if drive else \
        dict(mode="self_consistent", eta=eta, Omega_D=OD)
    # Omega_cs[0] != 1 exercises the normalized source -J/Omega_cs[0]
    return ds.Model(Nx=1, Nn=3, Nm=3, Np=3, alpha_s=(0.05, 0.07, 0.06), Omega_cs=(1.7,),
                    rho_background=1.0, **kw)


def test_a03_coupled_oscillator_roots_beats_and_work():
    eta, OD = 0.25, 1.2
    model = _homogeneous(eta, OD)
    y = ds.maxwellian(model, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(0.01)}
    out = ds.run(model, y, 60.0, n_save=601, rtol=1e-12, atol=1e-14)
    t = out["t"]
    # Independent linear solution for (J, E, E_D, A_D) at k = 0 (omega_L = 1, Om0 = 1).
    Mtx = np.array([[0, 1, eta, 0], [-1, 0, 0, 0], [-eta, 0, 0, OD ** 2], [0, 0, -1, 0]], float)
    ref = np.array([expm(Mtx * tt) @ [0, 0.01, 0, 0] for tt in t])
    assert _max(out["Fk"][:, 0, 0, 0, 0].real - ref[:, 1]) < 1e-11
    assert _max(out["Dk"][:, 0, 0, 0, 0].real - ref[:, 2]) < 1e-11
    # Both roots of (w^2 - wL^2)(w^2 - OD^2) - eta^2 wL^2 w^2 = 0
    roots = np.sqrt(np.sort(np.roots([1, -(1 + OD ** 2 + eta ** 2), OD ** 2]).real))
    w, res = ds.fit_modes(t, out["Fk"][:, 0, 0, 0, 0].real, [0.9, -0.9, 1.3, -1.3])
    got = np.sort(np.abs(w.real))
    assert res < 1e-9 and np.allclose(got, np.repeat(roots, 2), rtol=1e-9)
    assert np.max(np.abs(w.imag)) < 1e-9
    # Beats: the dark energy exchanges with the plasma at the root difference; work balance.
    W_D = out["W"][:, 1]
    assert _max(W_D - (out["K"] + out["U_gamma"] - out["K"][0] - out["U_gamma"][0])) < 1e-12
    assert _max(W_D + out["U_D"] - out["U_D"][0]) < 1e-12
    assert W_D.min() < -1e-7  # energy really flows to the dark field


def test_a06_eta_sign_symmetry():
    def go(eta):
        model = ds.Model(Nx=8, Nn=12, Lx=2 * np.pi / 2.0, alpha_s=(0.12, 0.12, 0.12),
                         rho_background=1.0, mode="self_consistent", eta=eta, Omega_D=0.9)
        y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), 0.02)]))
        y = ds.proca_mode(model, y, (2, 0, 0), np.sign(eta) * np.array([0.01, 0.0, 0.003]))
        return ds.run(model, y, 8.0, n_save=5, rtol=1e-11, atol=1e-13)

    a, b = go(0.3), go(-0.3)
    assert _max(a["Ck"] - b["Ck"]) < 1e-12 and _max(a["Fk"] - b["Fk"]) < 1e-12
    assert _max(a["Dk"] + b["Dk"]) < 1e-12
    assert _max(a["U_D"] - b["U_D"]) < 1e-14


@pytest.mark.parametrize("seed_amp", [0.0, 0.05])
def test_a07_exact_mean_pump_with_immobile_ions(seed_amp):
    E0 = 0.005  # quiver speed stays below the basis width over the run
    model = ds.Model(Nx=8, Nn=16, Lx=2 * np.pi / 1.0, alpha_s=(0.2, 0.2, 0.2), rho_background=1.0,
                     nu=1.0,  # parent closure; it leaves Hermite orders 0-2 untouched
                     mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=1.0)
    pert = [(0, (1, 0, 0), seed_amp), (0, (2, 0, 0), 0.5j * seed_amp)] if seed_amp else []
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], pert))
    out = ds.run(model, y, 30.0, n_save=61, rtol=1e-11, atol=1e-13)
    assert out["status"] == "success"
    t = out["t"]
    Ebar = out["Fk"][:, 0, 0, 0, 0].real
    assert _max(Ebar + E0 * t / 2 * np.sin(t)) < 1e-9
    led = out["K"] + out["U_gamma"] - out["W"][:, 2]
    assert _max(led - led[0]) < 1e-13 and out["W"][-1, 2] > 1e-4
    assert out["gauss"][:, 0].max() < 1e-12


@pytest.mark.parametrize("Nx,pops", [(7, 1), (8, 2), (10, 3), (1, 2)])
def test_a09_populations_grids_and_ledger(Nx, pops):
    q = (-1.0, -1.0, 1.0)[:pops] if pops != 1 else (-1.0,)
    Om = (1.0, 1.0, 1.0 / 30)[:pops]
    a = (0.1, 0.1, 0.1, 0.08, 0.1, 0.1, 0.02, 0.02, 0.02)[:3 * pops]
    u = (0.2, 0.0, 0.01, -0.2, 0.0, -0.01, 0.0, 0.0, 0.0)[:3 * pops]
    dens = [0.5, 0.5, 1.0][:pops] if pops == 3 else [1.0] * pops
    rho_b = -float(np.dot(q, dens))
    model = ds.Model(Nx=Nx, Nn=6, Nm=3, Np=3, Lx=2 * np.pi / 1.3, qs=q, Omega_cs=Om, alpha_s=a,
                     u_s=u, rho_background=rho_b, mode="self_consistent", eta=0.2, Omega_D=1.1)
    pert = [(0, (1, 0, 0), 0.01 + 0.004j)] if Nx > 1 else []
    y = ds.consistent_fields(model, ds.maxwellian(model, dens, pert), E_mean=(0.003, 0.0, 0.001))
    # mean current from the populations' drifts is kept, not deleted
    Jx = sum(qs * ns * us for qs, ns, us in zip(q, dens, u[::3]))
    _, M, _ = ds.moments(model, y["Ck"])
    assert np.isclose(float(jnp.sum(jnp.asarray(q)[:, None] * M[:, :, 0, 0, 0].real, axis=0)[0]), Jx)
    out = ds.run(model, y, 4.0, n_save=9, rtol=1e-11, atol=1e-13)
    assert out["status"] == "success"
    assert _max(out["ledger_defect"]) < 1e-13
    assert out["gauss"].max() < 1e-12
    E_tot = out["K"] + out["U_gamma"] + out["U_D"]
    assert _max(E_tot - E_tot[0]) < 1e-13


@pytest.mark.parametrize("Nx", [6, 7, 9, 10])
def test_parseval_weights_even_odd(Nx):
    rng = np.random.default_rng(Nx)
    f, g = rng.normal(size=(2, 3, 2, Nx, 3))
    fk = jnp.fft.rfftn(f, axes=(-1, -3, -2), norm="forward")
    gk = jnp.fft.rfftn(g, axes=(-1, -3, -2), norm="forward")
    assert np.isclose(float(ds.inner(Nx, fk, gk)), np.mean(np.sum(f * g, axis=0)), rtol=1e-13)


def test_failure_is_reported():
    model = ds.Model(Nx=1, mode="self_consistent", eta=0.1, rho_background=1.0)
    y = ds.maxwellian(model, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(0.1)}
    out = ds.run(model, y, 50.0, n_save=3, max_steps=4)
    assert out["status"] == "failure"


def test_massless_yukawa_zero_mode_and_cli(tmp_path):
    m = ds.Model(Nx=4, Lx=1.0, alpha_s=(0.1, 0.1, 0.1), rho_background=1.0,
                 mode="self_consistent", eta=0.3, Omega_D=0.0)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 0.01)]))
    assert np.isfinite(np.asarray(y["Dk"])).all() and y["Dk"][9, 0, 0, 0] == 0
    toml = tmp_path / "in.toml"
    toml.write_text("""
[model]
Nx = 8
Nn = 6
Lx = 4.0
alpha_s = [0.1, 0.1, 0.1]
rho_background = 1.0
mode = "self_consistent"
eta = 0.2
Omega_D = 1.0
[init]
perturbations = [[0, [1, 0, 0], 0.01, 0.0]]
[[init.proca]]
index = [1, 0, 0]
A_re = [0.0, 0.01, 0.0]
[run]
t_max = 1.0
n_save = 3
""")
    assert main([str(toml), "--out", str(tmp_path / "o")]) == 0
    rec = json.loads((tmp_path / "o" / "run.json").read_text())
    assert rec["status"] == "success" and rec["parent_commit"] == ds.PARENT_COMMIT
    assert rec["max_abs_ledger_defect"] < 1e-11  # default rtol 1e-10, atol 1e-12
    assert main([str(toml), "--out", str(tmp_path / "r"), "--t-max", "0.5",
                 "--resume", str(tmp_path / "o" / "run.npz")]) == 0
