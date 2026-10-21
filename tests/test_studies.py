"""Study helpers that delegate to the parent must match the copies the records were made with."""
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))


def _copy_rate(F, u_s, a_s, m, c, magnetic):
    """The b09_closure.py EB/E rate as committed before the dd80b0f repin (records were made with it)."""
    dev = jnp.max(jnp.abs(F - jnp.mean(F, axis=(-3, -2, -1), keepdims=True)), axis=(-3, -2, -1))
    dE, dB = dev[:3], dev[3:]
    N = jnp.array([m.Nn, m.Nm, m.Np], dtype=float)
    a = a_s.reshape(m.Ns, 3)
    V = jnp.abs(u_s.reshape(m.Ns, 3)) + a * jnp.sqrt(2 * N)[None]
    s = jnp.broadcast_to(dE[None], (m.Ns, 3))
    if magnetic:
        s = s + jnp.stack([V[:, 1] * dB[2] + V[:, 2] * dB[1],
                           V[:, 2] * dB[0] + V[:, 0] * dB[2],
                           V[:, 0] * dB[1] + V[:, 1] * dB[0]], axis=1)
    per = jnp.where((N > 3)[None], jnp.sqrt(2 * N)[None] * s / a, 0.0)
    return c * jnp.abs(m.p["qs"] * m.p["Omega_cs"]) * jnp.max(per, axis=1)


@pytest.mark.parametrize("magnetic", [True, False])
@pytest.mark.parametrize("seed,Nn,Nm,Np", [(0, 32, 16, 1), (1, 64, 32, 1), (2, 8, 8, 8)])
def test_b09_closure_rate_is_parent(seed, Nn, Nm, Np, magnetic):
    b09c = pytest.importorskip("b09_closure")
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    Ns = 2
    F = jax.random.normal(k[0], (6, 4, 16, 2)) * jnp.array([1e-3, 1e-2, 1e-4, 1e-5, 1e-6, 3e-2])[:, None, None, None]
    u_s = jax.random.normal(k[1], (3 * Ns,)) * 0.1
    a_s = 0.05 + jax.random.uniform(k[2], (3 * Ns,)) * 0.2
    m = SimpleNamespace(Nn=Nn, Nm=Nm, Np=Np, Ns=Ns,
                        p={"qs": jnp.array([-1.0, 1.0]), "Omega_cs": jnp.array([1.0, 1 / 1836])})
    ref = np.asarray(_copy_rate(F, u_s, a_s, m, 1.3, magnetic))
    got = np.asarray(b09c.closure_rate(F, u_s, a_s, m, 1.3, magnetic))
    assert got.shape == (Ns,)
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=0)


def test_r1_phased_drive_and_linear_theory():
    """R1 helpers: the jax drive equals the numpy phase formula; the derived expectation is the exact tau_c = inf work and,
    for a finite line, below it; the 4x4 reservoir matrix equals an independent ODE solve."""
    r1 = pytest.importorskip("r1_common")
    z = r1.noise_knots(30.0, 3, T=60.0, fac=2.0)
    m = r1.mobile_model(Nx=1, Nn=3, E0=1e-3, cls=r1.PhasedModel, z=tuple(z))
    t = np.linspace(0.0, 60.0, 37)
    got = np.array([float(m.drive(tt)[0]) for tt in t])
    assert np.allclose(got, 1e-3 * np.cos(r1.w_L * t + r1.phase_np(t, z)), atol=1e-15)
    assert abs(r1.phase_variance_rate(30.0, 2.0) * 30.0 - 2.0) < 0.04
    te, We, _ = r1.expected_work(np.inf, 1e-3, T=60.0, out_every=10)
    assert np.allclose(We[1:], r1.prescribed_linear(te, 1e-3)[1:], rtol=1e-3)
    _, Wf, _ = r1.expected_work(30.0, 1e-3, T=60.0, out_every=10, fac=2.0)
    assert 0 < Wf[-1] < We[-1]
    tt = np.linspace(0, 100, 11)
    X, _ = r1.coupled_solution(tt, 0.02, 1.0)
    assert np.allclose(X, r1.coupled_solution_ivp(tt, 0.02, 1.0), atol=1e-8)


def test_h08_run_restores_patches_and_continues_time_axis(tmp_path):
    """h08.run patches _sim.run/_sim.rhs; repeated in-process calls must not stack offsets or closures."""
    import json

    import h08
    from darkspectrax import _simulation as _sim

    run0, rhs0 = _sim.run, _sim.rhs
    h08.RUNS = tmp_path
    args = SimpleNamespace(vq=0.03, nu=0.0, Nx=8, Nn=8, T=40.0, chunk=20.0, control=False)
    for _ in range(2):
        assert h08.run(args) == 0
        assert _sim.run is run0 and _sim.rhs is rhs0
    t = [np.load(tmp_path / f"vq0.03_nu0_Nx8_Nn8.part{i}.npz")["t"] for i in (0, 1)]
    assert t[0][0] == 0.0 and np.isclose(t[1][0], t[0][-1]) and np.isclose(t[1][-1], 40.0)
    assert json.loads((tmp_path / "vq0.03_nu0_Nx8_Nn8.part1.json").read_text())["t_start"] == pytest.approx(t[0][-1])


def test_b10_reference_limits():
    """B10 reference: eq 2.9 hydrodynamic asymptote, eq 2.8 static maximum (w/4 with the exact algebra), the dark
    generalization collapsing to the ordinary one at eta -> 0, and the R0/R1 tiers agreeing for a small-k0 ordinary pump."""
    ref = pytest.importorskip("b10_ref")
    pl = ref.Plasma(vte=0.1, mi=1836.0, tau=10.0)
    mu = 1 / 1836.0
    w = 50 * mu
    E0 = 0.1 * np.sqrt(2 * w)
    g = ref.growth(pl, 1e-3, 5.0, E0, tier="R0", cs2=0.0)[0]
    assert abs(g / np.sqrt(mu * w / 3) - 1) < 0.01
    w = 1e-6
    Kc, gm = ref.closed_form_static(pl, w)
    assert abs(gm - w / 4 * pl.Te / (pl.Te + pl.Ti)) < 1e-15  # the exact maximum of eq 2.8, not the printed w/16
    E0 = 0.1 * np.sqrt(2 * w)
    g = max(ref.growth(pl, 1e-3, Kc * f, E0, tier="R1")[0] for f in np.linspace(0.6, 1.6, 21))
    assert abs(g / gm - 1) < 0.10  # the record's G0a gate is 5%; the root finder differs slightly across platforms
    E0 = 0.1 * np.sqrt(2 * 0.04)
    w0 = pl.omega(0.05)
    r0 = ref.growth(pl, 0.05, 0.2, E0, w0, "R0")[0]
    r1 = ref.growth(pl, 0.05, 0.2, E0, w0, "R1")[0]
    assert r1 > 0 and abs(r1 / r0 - 1) < 0.02
    pd = ref.Plasma(vte=0.1, mi=1836.0, tau=10.0, eta=1e-8, OmegaD=1.0)
    rd = ref.growth(pd, 0.05, 0.2, E0, pd.omega(0.05), "R1")[0]
    assert abs(rd / r1 - 1) < 1e-4


def test_r2_linear_drag_law_and_case_list():
    import r2_gates
    import r2_theory as th
    assert len(r2_gates.cases()) == 192
    t = np.linspace(0, 400, 20001)
    nu, E0 = 0.6, 0.03 * th.vte
    for om in (th.wL, 0.9):
        W = th.exact_linear(0.03, nu, om, t)[1]
        slope = (W[-1] - W[-6001]) / (t[-1] - t[-6001])  # 120 time units: > 15 periods, endpoint ripple at 2 omega is about 1%
        assert abs(slope / th.p_ss(E0, nu, om) - 1) < 3e-2
    assert abs(th.p_ss(E0, 1e-2, th.wL) * 2 * th.nu_eff(1e-2) / (th.wL ** 2 * E0 ** 2) - 1) < 1e-3  # resonant limit
