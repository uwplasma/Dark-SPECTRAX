"""Pump (oscillating-centre) frame and basis remaps on the parent's moving-basis operators."""

import jax.numpy as jnp
import numpy as np
import pytest

import darkspectrax as ds


def _max(a):
    return float(np.max(np.abs(a)))


def _pumped(frame, E0=0.05, seed_amp=0.05, Nn=16):
    return ds.Model(Nx=8, Nn=Nn, Lx=2 * np.pi, alpha_s=(0.2, 0.2, 0.2), rho_background=1.0, nu=1.0,
                    mode="prescribed_drive", E_drive=(E0, 0.0, 0.0), omega_drive=1.0, frame=frame)


def _y0(model, seed_amp=0.05):
    pert = [(0, (1, 0, 0), seed_amp), (0, (2, 0, 0), 0.5j * seed_amp)] if seed_amp else []
    return ds.consistent_fields(model, ds.maxwellian(model, [1.0], pert))


def test_pump_frame_without_uniform_field_is_the_fixed_rhs():
    """With no uniform force the pump-frame kinetic RHS equals the fixed-frame one exactly."""
    fixed, pump = _pumped("fixed", E0=0.0), _pumped("pump", E0=0.0)
    y = _y0(pump)
    a, b = ds.rhs(0.0, {k: v for k, v in y.items() if k != "B"}, fixed), ds.rhs(0.0, y, pump)
    # the box mean of the k != 0 fields is round-off, not exactly zero
    assert _max(a["Ck"] - b["Ck"]) < 1e-15 * _max(a["Ck"]) and jnp.array_equal(a["Fk"], b["Fk"])
    assert _max(b["B"]) < 1e-15


@pytest.mark.parametrize("seed_amp", [0.0, 0.05])
def test_a07_mean_pump_identity_in_the_pump_frame(seed_amp):
    """A07 in the pump frame: E_bar = -E0 t sin(t)/2 to 1e-9, and without seeds the electrons stay a single
    Hermite mode while the centre carries the quiver. K is quadratic in (u, C) here, so the explicit
    scheme closes the ledger to solver tolerance (fixed frame: to round-off, K is affine in C)."""
    E0 = 0.005
    model = _pumped("pump", E0=E0)
    out = ds.run(model, _y0(model, seed_amp), 30.0, n_save=61, rtol=1e-11, atol=1e-13)
    assert out["status"] == "success"
    t, Ebar = out["t"], out["Fk"][:, 0, 0, 0, 0].real
    assert _max(Ebar + E0 * t / 2 * np.sin(t)) < 1e-9
    led = out["K"] + out["U_gamma"] - out["W"][:, 2]
    assert _max(led - led[0]) < 1e-7 * out["W"][-1, 2] and out["W"][-1, 2] > 1e-4
    assert out["gauss"][:, 0].max() < 1e-12
    # centre = mean velocity: u_dot = -(E_bar + E_drive)
    if not seed_amp:
        assert _max(out["Ck"][:, 1:16]) < 1e-13 * _max(out["Ck"][:, 0, 0, 0, 0])
        # electron centre: u_dot = -(E_bar + E0 cos t), E_bar = -E0 t sin(t)/2
        u_ref = E0 / 2 * (np.sin(t) - t * np.cos(t)) - E0 * np.sin(t)
        assert _max(out["B"][:, 0, 0] - u_ref) < 1e-9


def test_remapped_run_closes_the_ledger_and_replays_bitwise():
    """Strong resonant pump in the pump frame with remaps between segments: every remap preserves the
    low moments, the work ledger closes over the whole run, and the frozen schedule replays bitwise."""
    model = _pumped("pump", E0=0.05, Nn=16)
    y0 = _y0(model)
    live = ds.run_adaptive(model, y0, 40.0, 4.0, rtol=1e-10, atol=1e-13,
                           trigger={"shift_on": 0.05, "width_on": 0.01})
    assert live["status"] == "success" and len(live["events"]) >= 1
    assert all(ev["moment_defect"] < 1e-12 for ev in live["events"])
    scale = np.abs(live["W"][:, 2]).max()
    assert _max(live["ledger_defect"]) < 1e-9 * scale
    schedule = [(ev["t"], ev["new_basis"]) for ev in live["events"]]
    rep = ds.run_adaptive(model, y0, 40.0, 4.0, rtol=1e-10, atol=1e-13, schedule=schedule)
    assert np.array_equal(rep["Ck"], live["Ck"]) and np.array_equal(rep["W"], live["W"])


def test_run_adaptive_needs_the_pump_frame():
    with pytest.raises(ValueError, match="pump"):
        ds.run_adaptive(_pumped("fixed"), None, 1.0, 1.0)
    with pytest.raises(ValueError, match="frame"):
        _pumped("lab")


def test_run_adaptive_stops_at_a_failed_segment():
    model = _pumped("pump", E0=0.05)
    out = ds.run_adaptive(model, _y0(model), 4.0, 2.0, max_steps=3)
    assert out["status"] == "failure" and out["failure_reason"].startswith("segment 0")
