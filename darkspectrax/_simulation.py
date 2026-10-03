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

from ._diagnostics import charge_density, energies, gauss_residuals
from ._model import PARENT_COMMIT, Model, rhs

__all__ = ["maxwellian", "consistent_fields", "proca_mode", "run", "save_record"]


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
    rho = charge_density(model, y["Ck"])
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


def run(model: Model, y0, t_max, n_save=101, rtol=1e-10, atol=1e-12, dt0=1e-3,
        solver=None, max_steps=200_000, progress=False, fixed_dt=None, t0=0.0):
    """Integrate from ``t0`` to ``t0 + t_max`` with Dopri8 (adaptive PID, or constant ``fixed_dt``).

    Restart a run by passing its final state (including the work ledger ``W``) and final time as ``t0``.

    Returns a dict of saved states, diagnostics, solver statistics, timings and
    the work ledger. ``status`` is ``"success"`` only when Diffrax reports
    success *and* every saved array is finite; otherwise ``failure_reason`` says why.
    """
    solver = diffrax.Dopri8() if solver is None else solver
    ts = jnp.linspace(t0, t0 + t_max, n_save)
    term = diffrax.ODETerm(lambda t, y, args: rhs(t, y, model))
    meter = diffrax.TqdmProgressMeter() if progress else diffrax.NoProgressMeter()

    if fixed_dt is None:
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
    else:
        controller, dt0 = diffrax.ConstantStepSize(), fixed_dt

    @jax.jit
    def solve(y0):
        return diffrax.diffeqsolve(
            term, solver, t0, t0 + t_max, dt0, y0, saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=max_steps, throw=False, progress_meter=meter)

    tic = _time.perf_counter()
    with warnings.catch_warnings():  # complex states are used exactly as in the parent
        warnings.filterwarnings("ignore", message="Complex dtype support")
        compiled = solve.lower(y0).compile()
        t1 = _time.perf_counter()
        sol = jax.block_until_ready(compiled(y0))
    t2 = _time.perf_counter()

    ys = sol.ys
    states = [jax.tree_util.tree_map(lambda a, i=i: a[i], ys) for i in range(len(ts))]
    en = [energies(model, s) for s in states]
    gs = [gauss_residuals(model, s) for s in states]
    out = {key: np.array([float(e[key]) for e in en]) for key in ("K", "U_gamma", "U_D")}
    out.update({
        "t": np.asarray(sol.ts), "Ck": np.asarray(ys["Ck"]), "Fk": np.asarray(ys["Fk"]),
        "Dk": np.asarray(ys["Dk"]), "W": np.asarray(ys["W"]).real,
        "gauss": np.array([[float(a), float(b)] for a, b in gs]),
        "status": "success", "failure_reason": None,
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
    W = out["W"]
    # Ledger: K gains W_em + W_D + W_ext; U_gamma loses W_em; U_D loses W_D.
    out["ledger_defect"] = ((out["K"] - out["K"][0]) - W.sum(axis=1),
                            (out["U_gamma"] - out["U_gamma"][0]) + W[:, 0],
                            (out["U_D"] - out["U_D"][0]) + W[:, 1])
    out["ledger_defect"] = np.array(out["ledger_defect"])
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
