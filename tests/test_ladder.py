"""A04 finite-k cold branches, A08 exact-solution convergence, A10 failure policy."""

import numpy as np
import pytest

import darkspectrax as ds


def _cold_model(k, eta, OD, vt=0.005, Nn=6):
    a = np.sqrt(2) * vt
    return ds.Model(Nx=5, Nn=Nn, Nm=3, Np=3, Lx=2 * np.pi / k, alpha_s=(a, a, a),
                    rho_background=1.0, mode="self_consistent", eta=eta, Omega_D=OD)


def _fit_two(t, z, roots):
    g = [w for r in roots for w in (r, -r)]
    w, res = ds.fit_modes(t, z, g)
    return np.sort(np.abs(w.real)), res


@pytest.mark.parametrize("k,OD", [(1.0, 1.2), (0.5, 0.3)])   # second: near the avoided crossing
def test_a04_transverse_cold_branches(k, OD):
    """(z - wL^2)(z - OD^2) - eta^2 wL^2 z = 0, z = w^2 - k^2, with the full Hermite/Proca system."""
    eta = 0.3
    m = _cold_model(k, eta, OD)
    y = ds.maxwellian(m, [1.0])
    y = {**y, "Fk": y["Fk"].at[1, 0, 1, 0].set(1e-3)}           # transverse E_y seed
    out = ds.run(m, y, 60.0, n_save=601, rtol=1e-11, atol=1e-11)
    zs = np.sort(np.roots([1, -(1 + OD ** 2 + eta ** 2), OD ** 2]).real)
    roots = np.sqrt(zs + k ** 2)
    got, res = _fit_two(out["t"], out["Fk"][:, 1, 0, 1, 0], roots)
    assert res < 1e-5
    # thermal correction to the cold root is O(k^2 v_t^2 / w^2) ~ 1e-5 here
    assert np.allclose(got, np.repeat(roots, 2), rtol=2e-5)
    assert out["gauss"].max() < 1e-12 and np.abs(out["ledger_defect"]).max() < 1e-12


@pytest.mark.parametrize("k,OD", [(1.0, 0.8), (0.3, 1.0)])
def test_a04_longitudinal_cold_branches(k, OD):
    """(w^2 - wL^2)(w^2 - k^2 - OD^2) - eta^2 wL^2 (w^2 - k^2) = 0 from a density seed."""
    eta, vt = 0.3, 0.005
    m = _cold_model(k, eta, OD, vt=vt, Nn=8)
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 1e-3j)]))
    # also excite the massive branch with a longitudinal free Proca mode
    y = ds.proca_mode(m, y, (1, 0, 0), [1e-3, 0.0, 0.0])
    out = ds.run(m, y, 60.0, n_save=601, rtol=1e-11, atol=1e-11)
    W = np.roots([1, -(1 + k ** 2 + OD ** 2 + eta ** 2), k ** 2 + OD ** 2 + eta ** 2 * k ** 2])
    roots = np.sqrt(np.sort(W.real))
    got, res = _fit_two(out["t"], out["Fk"][:, 0, 0, 1, 0], roots)
    assert res < 1e-4
    # Bohm-Gross-type warm shift 3 k^2 v_t^2 / 2 ~ 4e-5 relative on the plasma branch
    assert np.allclose(got, np.repeat(roots, 2), rtol=1e-4)
    assert out["gauss"].max() < 1e-12


def test_a08_time_convergence_order():
    """Fixed-step Dopri8 against the exact homogeneous coupled-oscillator solution."""
    from scipy.linalg import expm
    eta, OD = 0.25, 1.2
    m = ds.Model(Nx=1, Nn=3, Nm=3, Np=3, alpha_s=(0.05,) * 3, rho_background=1.0,
                 mode="self_consistent", eta=eta, Omega_D=OD)
    y = ds.maxwellian(m, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(0.01)}
    M = np.array([[0, 1, eta, 0], [-1, 0, 0, 0], [-eta, 0, 0, OD ** 2], [0, 0, -1, 0]], float)
    ref = (expm(M * 10.0) @ [0, 0.01, 0, 0])[1]
    errs = []
    for dt in (0.5, 0.25, 0.125):
        out = ds.run(m, y, 10.0, n_save=2, fixed_dt=dt)
        errs.append(abs(out["Fk"][-1, 0, 0, 0, 0].real - ref))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 7.5), orders  # Dopri8 is eighth order


def test_a08_hermite_convergence_free_streaming():
    """Exact free streaming of a Maxwellian density perturbation: dn(t) = dn0 exp(-(k v t)^2/2).

    The error falls with Hermite order until the front n ~ (k v t)^2 is resolved."""
    k, vt, t_end = 1.0, 1.0, 3.0
    errs = []
    for Nn in (8, 16, 32):
        m = ds.Model(Nx=5, Nn=Nn, Lx=2 * np.pi / k, qs=(0.0,), alpha_s=(np.sqrt(2) * vt,) * 3)
        y = ds.maxwellian(m, [1.0], [(0, (1, 0, 0), 1e-3)])
        out = ds.run(m, y, t_end, n_save=7, rtol=1e-12, atol=1e-16)
        n_k = out["Ck"][:, 0, 0, 1, 0] * 2 * vt ** 3 * np.sqrt(2) ** 3 / 2
        exact = 1e-3 * np.exp(-(k * vt * out["t"]) ** 2 / 2)
        errs.append(np.max(np.abs(n_k - exact)))
    assert errs[0] > 10 * errs[1] > 100 * errs[2] and errs[2] < 1e-12


def test_a10_nonfinite_initial_state_fails():
    m = ds.Model(Nx=1, mode="self_consistent", eta=0.1, rho_background=1.0)
    y = ds.maxwellian(m, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(np.nan)}
    out = ds.run(m, y, 1.0, n_save=3, max_steps=50)
    assert out["status"] == "failure" and out["failure_reason"]


def test_a10_short_stiff_transient_succeeds_without_floor():
    """A fast dark mode (Omega_D = 200) needs tiny steps; the run still finishes inside its budget."""
    m = ds.Model(Nx=1, Nn=3, Nm=3, Np=3, alpha_s=(0.05,) * 3, mode="self_consistent", eta=0.2,
                 Omega_D=200.0, rho_background=1.0)
    y = ds.proca_mode(m, ds.maxwellian(m, [1.0]), (0, 0, 0), [1e-3, 0, 0])
    out = ds.run(m, y, 0.5, n_save=3, rtol=1e-9, atol=1e-14)
    assert out["status"] == "success" and out["failure_reason"] is None
    assert np.abs(out["ledger_defect"]).max() < 1e-7 * out["U_D"][0]  # rtol-level


def test_a10_step_budget_failure_reason():
    m = ds.Model(Nx=1, mode="self_consistent", eta=0.1, rho_background=1.0)
    y = ds.maxwellian(m, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(0.1)}
    out = ds.run(m, y, 50.0, n_save=3, max_steps=4)
    assert out["status"] == "failure" and "max_steps" in out["failure_reason"]
    assert 1 <= out["num_valid_times"] < 3
    floor = ds.run(m, y, 50.0, n_save=3, dtmin=10.0)
    assert floor["status"] == "failure" and "minimum step size" in floor["failure_reason"]
    ok = ds.run(m, y, 1.0, n_save=3)
    assert ok["status"] == "success" and ok["num_valid_times"] == 3


def test_a10_fixed_step_nonfinite_is_not_success():
    """A constant-step run cannot reject a NaN step; the driver still refuses to report success."""
    m = ds.Model(Nx=1, mode="self_consistent", eta=0.1, rho_background=1.0)
    y = ds.maxwellian(m, [1.0])
    y = {**y, "Fk": y["Fk"].at[0, 0, 0, 0].set(np.inf)}
    out = ds.run(m, y, 1.0, n_save=3, fixed_dt=0.1)
    assert out["status"] == "failure" and out["failure_reason"] == "nonfinite state"
