"""Field-scaled Hermite closure (SPECTRAX PR #66 rate) and its exact diagonal rate for the integrating-factor solve."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from spectrax._model import field_scaled_closure_rate

from ._model import Model, basis_of, rhs as _rhs

__all__ = ["field_closure", "static_rate"]


def _spectrum(model: Model):
    """Per-row factor ``s(n) = n(n-1)(n-2)/((N-1)(N-2)(N-3))`` (density, momentum and energy rows untouched)."""
    n = np.arange(model.Nn, dtype=float)
    return n * (n - 1) * (n - 2) / ((model.Nn - 1) * (model.Nn - 2) * (model.Nn - 3))


def field_closure(model: Model, c: float, nu_fixed=None):
    """``(rhs_fn, rate_fn)`` for the field-scaled closure ``-nu_s s(n) C_s`` with ``nu_s`` from SPECTRAX
    :func:`spectrax.field_scaled_closure_rate` (strength ``c``, 1D1V: electric field ``E_x`` only; ``nu_fixed``
    of shape ``(Ns,)`` freezes the rate).

    ``rhs_fn(t, y, model)`` is the Dark right-hand side plus that damping (identical to the study wrapper
    ``studies/nxconv_run.install_closure``). ``rate_fn(t, y, model)`` returns the same damping as a non-negative
    rate pytree (zero for ``Fk, Dk, W, B``) for ``run(..., integrating_factor=rate_fn)``, so it can be integrated exactly.
    Needs ``Nn > 3`` and ``Nm = Np = 1``.
    """
    Ns, Nn = model.Ns, model.Nn
    if Nn <= 3 or model.Nm != 1 or model.Np != 1:
        raise ValueError("field_closure needs Nn > 3 and Nm = Np = 1")
    sn = jnp.asarray(_spectrum(model))
    qs, Om = jnp.asarray(np.asarray(model.qs, float)), jnp.asarray(np.asarray(model.Omega_cs, float))
    mask = model.p["mask23"]

    def nu_of(y, mdl):
        if nu_fixed is not None:
            return jnp.asarray(nu_fixed, float)
        Ex = jnp.fft.irfftn(y["Fk"][0] * mask, s=(model.Nz, model.Ny, model.Nx), axes=(-1, -3, -2), norm="forward")
        F = jnp.stack([Ex, jnp.zeros_like(Ex), jnp.zeros_like(Ex)])
        return field_scaled_closure_rate(F, basis_of(mdl, y)[1], qs, Om, Nn, 1, 1, Ns, c).reshape(Ns)

    def rhs_fn(t, y, mdl):
        out = _rhs(t, y, mdl)
        Ck = y["Ck"].reshape(Ns, Nn, *y["Ck"].shape[1:])
        damp = -(nu_of(y, mdl)[:, None, None, None, None] * sn[None, :, None, None, None]) * Ck
        return {**out, "Ck": out["Ck"] + damp.reshape(y["Ck"].shape)}

    def rate_fn(t, y, mdl):
        rate = (nu_of(y, mdl)[:, None] * sn[None, :]).reshape(Ns * Nn, 1, 1, 1)
        return {k: (rate if k == "Ck" else 0.0) for k in y}

    return rhs_fn, rate_fn


def static_rate(model: Model):
    """Rate pytree function of the model's constant numerical hypercollision ``nu * col`` (the parent
    ``Col = -nu col Ck``) for ``run(..., integrating_factor=True)``."""
    def rate_fn(t, y, mdl):
        p = mdl.p
        rate = (p["nu"] * jnp.tile(jnp.asarray(p["collision_matrix"]).reshape(-1), mdl.Ns)).reshape(-1, 1, 1, 1)
        return {k: (rate if k == "Ck" else 0.0) for k in y}
    return rate_fn
