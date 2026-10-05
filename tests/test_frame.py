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


def test_run_adaptive_compiles_once(monkeypatch):
    """All segments of a run share one compiled solve (t0 is traced)."""
    calls = []
    compile_ = ds._simulation._compile
    monkeypatch.setattr(ds._simulation, "_compile", lambda *a: calls.append(1) or compile_(*a))
    model = _pumped("pump", E0=0.05)
    out = ds.run_adaptive(model, _y0(model), 8.0, 2.0, rtol=1e-10, atol=1e-13)
    assert out["status"] == "success" and len(calls) == 1 and out["num_steps"] > 0


def test_run_adaptive_stops_when_positivity_is_lost():
    """A state with negative k=0 temperature stops after its first segment with a positivity failure_reason."""
    model = _pumped("pump", E0=0.05)
    y0 = _y0(model)
    y0 = {**y0, "Ck": y0["Ck"].at[2, 0, 0, 0].set(-y0["Ck"][0, 0, 0, 0])}  # T_x < 0 from C_2 = -C_0
    out = ds.run_adaptive(model, y0, 6.0, 2.0)
    assert out["status"] == "failure" and out["failure_reason"] == "segment 0: positivity lost at t = 0"
    assert out["t_reached"] == 2.0
    off = ds.run_adaptive(model, y0, 4.0, 2.0, stop_on_negative=False)
    assert "positivity" not in str(off["failure_reason"])


def test_noise_floor_removes_the_unseeded_pump_frame_stall():
    """Unseeded e-i plasma in the pump frame: the k=0 ion coefficients are round-off of C_000,i ~ 1/a_i^3, and
    plain PID at atol 1e-14 chases that noise. A per-species floor of 1e-14 |C_000,s| restores normal steps,
    and the uniform two-fluid work stays the exact resonant oscillator's."""
    vte, eps = np.sqrt(1e-3), 1 / 1836
    w = np.sqrt(1 + eps)
    model = ds.Model(Nx=8, Nn=16, Lx=40.0, qs=(-1.0, 1.0), Omega_cs=(1.0, eps),
                     alpha_s=(np.sqrt(2) * vte,) * 3 + (np.sqrt(2 * eps) * vte,) * 3, u_s=(0.0,) * 6,
                     mode="prescribed_drive", E_drive=(0.1 * vte, 0.0, 0.0), omega_drive=w, frame="pump")
    y0 = ds.consistent_fields(model, ds.maxwellian(model, [1.0, 1.0]))
    kw = dict(n_save=21, rtol=1e-10, atol=1e-14, max_steps=3000)
    plain, floored = ds.run(model, y0, 20.0, **kw), ds.run(model, y0, 20.0, noise_floor=1e-14, **kw)
    assert plain["status"] == "failure"
    assert floored["status"] == "success" and floored["num_steps"] < 300
    t, A = floored["t"], 0.1 * vte * (1 + eps) / (2 * w)
    r, rd = -A * t * np.sin(w * t), -A * (np.sin(w * t) + w * t * np.cos(w * t))
    W_lin = 0.5 / (1 + eps) * (rd ** 2 + w ** 2 * r ** 2)
    assert _max(floored["W"][:, 2] - W_lin) < 1e-8 * W_lin.max()
