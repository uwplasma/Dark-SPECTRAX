"""Initialization, constraint-consistent fields and the Diffrax driver."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time as _time
import warnings
from dataclasses import fields
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from ._diagnostics import charge_density, energies, gauss_residuals, moments
from ._model import PARENT_COMMIT, Model, rhs

__all__ = ["maxwellian", "consistent_fields", "proca_mode", "run", "save_record", "adapt_basis", "run_adaptive"]


def maxwellian(model: Model, densities, perturbations=()):
    """Basis-matched drifting Maxwellians plus density perturbations.

    ``densities[s]`` is the mean density of species ``s``. Each perturbation is
    ``(s, (ix, iy, iz), amplitude)``: it adds ``amplitude*exp(i k.x) + c.c.`` to
    ``n_s``, with ``ix >= 0`` the stored rfft index and ``iy, iz`` signed indices.
    """
    y = model.zeros()
    a = np.asarray(model.alpha_s, float).reshape(-1, 3).prod(axis=1)
    H = model.Nn * model.Nm * model.Np
    Ck = y["Ck"]
    for s, n0 in enumerate(densities):
        Ck = Ck.at[s * H, 0, 0, 0].add(n0 / a[s])
    for s, (ix, iy, iz), amp in perturbations:
        if ix == 0 and (iy, iz) != (0, 0):
            Ck = Ck.at[s * H, -iy, 0, -iz].add(np.conj(amp) / a[s])
        Ck = Ck.at[s * H, iy, ix, iz].add(amp / a[s])
    return {**y, "Ck": Ck}


def consistent_fields(model: Model, y, E_mean=(0.0, 0.0, 0.0)):
    """Longitudinal ordinary field and static Yukawa dark near field from rho.

    E_k = -i k rho_k/(Om0 k^2); phi_D = eta rho_k/(Om0 (k^2 + Omega_D^2));
    E_D = -i k phi_D; A_D = B_D = 0. Rejects nonzero mean charge.
    """
    om0 = model.Omega_cs[0]
    rho = charge_density(model, y["Ck"], y.get("B"))
    q0 = abs(complex(rho[0, 0, 0]))
    if q0 > 1e-12 * max(1.0, float(jnp.max(jnp.abs(rho)))):
        raise ValueError(f"periodic box is not neutral: mean charge {q0:.3e}")
    nab = model.p["nabla"]
    k2 = jnp.sum(nab ** 2, axis=0)
    safe = jnp.where(k2 > 0, k2, 1.0)
    E = jnp.where(k2 > 0, -1j * nab * rho / (om0 * safe), 0.0)
    E = E.at[:, 0, 0, 0].set(jnp.asarray(E_mean, complex))
    Fk = y["Fk"].at[:3].set(E)
    Dk = y["Dk"]
    if model.mode == "self_consistent" and model.eta != 0:
        den = k2 + model.Omega_D ** 2
        if model.Omega_D == 0:
            den = jnp.where(k2 > 0, den, 1.0)  # massless zero mode: gauge phi_0 = 0
        phi = jnp.where(k2 > 0, model.eta * rho / (om0 * den), 0.0)
        Dk = Dk.at[9].set(phi).at[:3].set(-1j * nab * phi)
    return {**y, "Fk": Fk, "Dk": Dk}


def proca_mode(model: Model, y, index, A):
    """Add a free vacuum Proca mode A_D exp(i(k.x - w t)) + c.c., w^2 = k^2 + Omega_D^2.

    ``index = (ix, iy, iz)`` with ``ix > 0`` (stored rfft half) or ``(0, 0, 0)``.
    At k = 0 the real mean field A_D = Re(A), E_D = Re(i w A) is used.
    """
    ix, iy, iz = index
    if ix == 0 and (iy, iz) != (0, 0):
        raise ValueError("use ix > 0 or the k = 0 mode")
    nab = model.p["nabla"][:, iy, ix, iz]
    A = jnp.asarray(A, complex)
    w = jnp.sqrt(jnp.sum(nab ** 2) + model.Omega_D ** 2)
    phi = jnp.sum(nab * A) / w
    E = 1j * w * A - 1j * nab * phi
    B = 1j * jnp.cross(nab, A)
    vals = jnp.concatenate([E, B, A, phi[None]])
    if ix == 0:
        vals = jnp.real(vals).astype(complex)
    return {**y, "Dk": y["Dk"].at[:, iy, ix, iz].add(vals)}


def _compile(model, y0, t_max, n_save, rtol, atol, dt0, solver, max_steps, progress, fixed_dt, dtmin, noise_floor):
    term = diffrax.ODETerm(lambda t, y, args: rhs(t, y, model))
    meter = diffrax.TqdmProgressMeter() if progress else diffrax.NoProgressMeter()
    if fixed_dt is not None:
        dt0 = fixed_dt

    @jax.jit
    def solve(y0, t0):
        if fixed_dt is not None:
            controller = diffrax.ConstantStepSize()
        elif noise_floor is None:
            controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmin=dtmin, force_dtmin=False)
        else:
            controller = _FlooredPID(rtol=rtol, atol=atol, dtmin=dtmin, force_dtmin=False,
                                     floor=_row_floor(model, y0["Ck"], noise_floor))
        ts = t0 + jnp.linspace(0.0, t_max, n_save)
        return diffrax.diffeqsolve(
            term, solver, t0, t0 + t_max, dt0, y0, saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller, max_steps=max_steps, throw=False, progress_meter=meter)

    return solve.lower(y0, jnp.asarray(0.0)).compile()


class _FlooredPID(diffrax.PIDController):
    """PID control with an absolute error floor per Ck row (``floor``): err / (atol + floor + rtol |y|)."""

    floor: jax.Array = None

    def adapt_step_size(self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state):
        y = jnp.maximum(jnp.abs(y0["Ck"]), jnp.abs(jnp.nan_to_num(y1_candidate["Ck"])))
        shrink = (self.atol + self.rtol * y) / (self.atol + self.floor + self.rtol * y)
        y_error = {**y_error, "Ck": y_error["Ck"] * shrink}
        return super().adapt_step_size(t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state)


def _row_floor(model: Model, Ck, noise_floor):
    """``noise_floor * |C_000(k=0)|`` of each row's species (the round-off scale of that species' coefficients)."""
    H = model.Nn * model.Nm * model.Np
    c0 = jnp.abs(Ck[::H, 0, 0, 0])
    return (noise_floor * jnp.repeat(c0, H))[:, None, None, None]


def run(model: Model, y0, t_max, n_save=101, rtol=1e-10, atol=1e-12, dt0=1e-3,
        solver=None, max_steps=200_000, progress=False, fixed_dt=None, t0=0.0, dtmin=None,
        noise_floor=None, cache=None):
    """Integrate from ``t0`` to ``t0 + t_max`` with Dopri8 (adaptive PID, or constant ``fixed_dt``).

    Restart a run by passing its final state (including the work ledger ``W``) and final time as ``t0``.

    Returns a dict of saved states, diagnostics, solver statistics, timings and
    the work ledger. ``status`` is ``"success"`` only when Diffrax reports
    success *and* every saved array is finite; otherwise ``failure_reason`` says why.
    Step limits follow the parent's PR #50 semantics: ``max_steps`` is the step budget and
    ``dtmin`` (default None: no floor) stops the solve with ``dt_min_reached`` instead of
    crawling; ``num_valid_times`` counts the saves that hold a solution.
    ``noise_floor`` (default None: plain PID) adds ``noise_floor * |C_000,s(k=0)|`` to ``atol`` on the
    rows of species ``s``, so round-off in coefficients far below a species' own scale cannot drive the
    step size (see docs/results.md, unseeded pump-frame stall). ``cache`` (a dict) reuses the compiled
    solve across calls with equal settings and shapes; ``t0`` is a traced argument.
    """
    solver = diffrax.Dopri8() if solver is None else solver
    key = (id(model), t_max, n_save, rtol, atol, dt0, id(solver), max_steps, progress, fixed_dt, dtmin, noise_floor,
           tuple((k, jnp.shape(v), str(jnp.result_type(v))) for k, v in sorted(y0.items())))
    compiled = None if cache is None else cache.get(key)
    tic = _time.perf_counter()
    with warnings.catch_warnings():  # complex states are used exactly as in the parent
        warnings.filterwarnings("ignore", message="Complex dtype support")
        if compiled is None:
            compiled = _compile(model, y0, t_max, n_save, rtol, atol, dt0, solver, max_steps, progress, fixed_dt,
                                dtmin, noise_floor)
            if cache is not None:
                cache[key] = compiled
        t1 = _time.perf_counter()
        sol = jax.block_until_ready(compiled(y0, jnp.asarray(t0, float)))
    t2 = _time.perf_counter()

    ys = sol.ys
    states = [jax.tree_util.tree_map(lambda a, i=i: a[i], ys) for i in range(n_save)]
    en = [energies(model, s) for s in states]
    gs = [gauss_residuals(model, s) for s in states]
    out = {key: np.array([float(e[key]) for e in en]) for key in ("K", "U_gamma", "U_D")}
    out.update({
        "t": np.asarray(sol.ts), "Ck": np.asarray(ys["Ck"]), "Fk": np.asarray(ys["Fk"]),
        "Dk": np.asarray(ys["Dk"]), "W": np.asarray(ys["W"]).real,
        "B": np.asarray(ys["B"]).real if "B" in ys else None,
        "gauss": np.array([[float(a), float(b)] for a, b in gs]),
        "status": "success", "failure_reason": None,
        "num_valid_times": int(np.isfinite(np.asarray(sol.ts)).sum()),
        "num_steps": int(sol.stats["num_steps"]),
        "num_accepted": int(sol.stats["num_accepted_steps"]),
        "num_rejected": int(sol.stats["num_rejected_steps"]),
        "compile_time": t1 - tic, "run_time": t2 - t1,
    })
    finite = all(np.isfinite(out[k]).all() for k in ("Ck", "Fk", "Dk", "W"))
    if sol.result != diffrax.RESULTS.successful:
        out["status"], out["failure_reason"] = "failure", str(sol.result)
    elif not finite:
        out["status"], out["failure_reason"] = "failure", "nonfinite state"
    W = out["W"] - out["W"][0]  # work done in this segment (a restart carries W(t0) != 0)
    # Ledger: K gains W_em + W_D + W_ext; U_gamma loses W_em; U_D loses W_D.
    out["ledger_defect"] = ((out["K"] - out["K"][0]) - W.sum(axis=1),
                            (out["U_gamma"] - out["U_gamma"][0]) + W[:, 0],
                            (out["U_D"] - out["U_D"][0]) + W[:, 1])
    out["ledger_defect"] = np.array(out["ledger_defect"])
    return out


def adapt_basis(model: Model, y, t=None, schedule_basis=None, **trigger):
    """Remap the pump-frame basis between segments with the parent's exact triangular remap.

    Live (``schedule_basis=None``): the parent's ``remap_trigger`` decides from the measured moments
    (keyword arguments are passed to it), and ``remap_event`` applies the capped move and returns
    its record. Replay: ``schedule_basis`` is a recorded basis, applied without any decision.
    Returns ``(y, record)``; ``record`` is None when nothing fired.
    """
    from spectrax._remap import remap, remap_event, remap_trigger

    shape = (model.Nn, model.Nm, model.Np, model.Ns)
    if schedule_basis is not None:
        new = jnp.asarray(schedule_basis, complex)
        return {**y, "Ck": remap(y["Ck"], y["B"], new, *shape), "B": new}, None
    fire, new, info = remap_trigger(y["Ck"], y["B"], *shape, **trigger)
    if not fire:
        return y, None
    Ck, rec = remap_event(y["Ck"], y["B"], jnp.asarray(new), *shape, t=t,
                          max_shift=trigger.get("max_shift", 1.0), max_narrow=trigger.get("max_narrow", 1.1))
    rec["trigger"] = info
    return {**y, "Ck": Ck, "B": jnp.asarray(new, complex)}, rec


_SAVED = ("t", "K", "U_gamma", "U_D", "W", "B", "Ck", "Fk", "Dk")


def _first_negative(model: Model, seg):
    """First saved time at which some species has K_s <= 0 or k=0 T_x <= 0 (None if never)."""
    for i in range(seg["t"].size):
        st = {key: jnp.asarray(seg[key][i]) for key in ("Ck", "Fk", "Dk", "B")}
        n, M, M2 = (np.asarray(x)[..., 0, 0, 0].real for x in moments(model, st["Ck"], st["B"]))
        K = np.asarray(energies(model, st)["K_species"])
        if np.any(K <= 0) or np.any(M2[:, 0, 0] - M[:, 0] ** 2 / n <= 0):
            return float(seg["t"][i])
    return None


def run_adaptive(model: Model, y0, t_max, segment, n_save_segment=11, schedule=None, trigger=None,
                 stop_on_negative=True, **run_kw):
    """Integrate in segments of length ``segment`` with a basis remap check after each one.

    ``model.frame`` must be ``"pump"``. ``schedule`` (a list of ``(t, basis)`` from a previous
    run's ``events``) replays a frozen remap schedule instead of deciding live. Returns the
    concatenated saves (``t, K, U_gamma, U_D, W, B, Ck, Fk``), the work ledger closed over the
    whole run, the event list, summed solver statistics (``num_steps, num_rejected, compile_time,
    run_time``) and ``status``. The run stops at a failed segment, and (``stop_on_negative``) after
    the first segment in which a saved kinetic energy or k=0 temperature is not positive: the
    Hermite solution is then unresolved and further segments only spend the step budget.
    The compiled solve is reused across segments (one compilation per run).
    """
    if model.frame != "pump":
        raise ValueError("run_adaptive needs model.frame == 'pump'")
    y, t, parts, events = y0, 0.0, [], []
    nseg = int(round(t_max / segment))
    replay = {round(float(tt), 9): b for tt, b in (schedule or [])}
    out = {"status": "success", "failure_reason": None, "num_steps": 0, "num_rejected": 0,
           "compile_time": 0.0, "run_time": 0.0}
    cache = {}
    for k in range(nseg):
        seg = run(model, y, segment, n_save=n_save_segment, t0=t, cache=cache, **run_kw)
        for key in ("num_steps", "num_rejected", "compile_time", "run_time"):
            out[key] += seg[key]
        parts.append({key: seg[key][(0 if k == 0 else 1):] for key in _SAVED})  # drop the repeated start
        if seg["status"] != "success":
            out.update(status="failure", failure_reason=f"segment {k}: {seg['failure_reason']}")
            break
        t = t + segment
        t_neg = _first_negative(model, seg) if stop_on_negative else None
        if t_neg is not None:
            out.update(status="failure", failure_reason=f"segment {k}: positivity lost at t = {t_neg:g}")
            break
        y = {"Ck": jnp.asarray(seg["Ck"][-1]), "Fk": jnp.asarray(seg["Fk"][-1]), "Dk": jnp.asarray(seg["Dk"][-1]),
             "W": jnp.asarray(seg["W"][-1], complex), "B": jnp.asarray(seg["B"][-1], complex)}
        if k == nseg - 1:
            break
        if schedule is not None:
            b = replay.get(round(t, 9))
            if b is not None:
                y, _ = adapt_basis(model, y, schedule_basis=b)
                events.append({"t": t, "new_basis": np.asarray(b).tolist()})
        else:
            y, rec = adapt_basis(model, y, t=t, **(trigger or {}))
            if rec is not None:
                events.append(rec)
        parts[-1]["B"] = np.concatenate([parts[-1]["B"][:-1], np.asarray(y["B"]).real[None]])
        parts[-1]["Ck"] = np.concatenate([parts[-1]["Ck"][:-1], np.asarray(y["Ck"])[None]])
    for key in _SAVED:
        out[key] = np.concatenate([p[key] for p in parts])
    W = out["W"] - out["W"][0]
    out["ledger_defect"] = np.array(((out["K"] - out["K"][0]) - W.sum(axis=1),
                                     (out["U_gamma"] - out["U_gamma"][0]) + W[:, 0],
                                     (out["U_D"] - out["U_D"][0]) + W[:, 1]))
    out["events"], out["t_reached"] = events, float(t)
    return out


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        return "unknown"


def save_record(path, model: Model, out, extra=None):
    """Write ``run.json`` (provenance, settings, scalar metrics) and ``run.npz`` (arrays)."""
    import diffrax as _dfx
    import spectrax

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = {f.name: getattr(model, f.name) for f in fields(model) if f.name != "_p"}
    rec = {
        "model": cfg, "status": out["status"],
        "repository_commit": _git_sha(), "parent_commit": PARENT_COMMIT,
        "versions": {"python": platform.python_version(), "jax": jax.__version__,
                     "diffrax": _dfx.__version__, "spectrax": spectrax.__version__,
                     "numpy": np.__version__},
        "backend": jax.default_backend(), "dtype": "complex128",
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "solver": {k: out[k] for k in ("num_steps", "num_accepted", "num_rejected",
                                       "compile_time", "run_time")},
        "failure_reason": out["failure_reason"],
        "max_abs_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
        "max_gauss_residual": [float(v) for v in out["gauss"].max(axis=0)],
        **(extra or {}),
    }
    (path / "run.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    np.savez_compressed(path / "run.npz", **{k: out[k] for k in (
        "t", "K", "U_gamma", "U_D", "W", "gauss", "ledger_defect")},
        Ck_final=out["Ck"][-1], Fk=out["Fk"], Dk=out["Dk"], W_final=out["W"][-1])
    return rec
