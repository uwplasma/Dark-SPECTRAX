"""Exact Hermite free-streaming integrating factor for ``run(..., integrating_factor=streaming(...))``.

Thin Dark-SPECTRAX adapter of :class:`spectrax.StreamingFactor` (SPECTRAX PRs #82/#83, derivation in
``spectrax/_lawson_streaming.py``): the Hermite widths and drifts come from the state's basis ``B`` (pump frame) or the model
constants and are frozen at the step start; ``nabla`` is the model's wave-number grid. Fields, dark fields, the work ledger
and the basis state have the identity as factor. Optionally the diagonal rate (``True``: the model's constant
hypercollision; a callable ``rate_fn(t, y, model)`` such as the second output of :func:`darkspectrax.field_closure`) is
put inside the factor (Strang composition); ``None`` leaves any damping explicit in the Runge-Kutta stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from spectrax import StreamingFactor
from spectrax._lawson_streaming import hermite_axis_eig

from ._closure import static_rate
from ._model import Model, basis_of

__all__ = ["streaming", "Streaming", "streaming_factor_fn"]


@dataclass(frozen=True)
class Streaming:
    """Marker for ``integrating_factor=``: exact streaming, with ``rate`` None (damping explicit), True (the model's
    constant hypercollision) or a callable ``rate_fn(t, y, model)`` (rate pytree with an ``"Ck"`` entry) inside the factor."""

    rate: object = None


def streaming(rate=None):
    """``integrating_factor`` value: exact Hermite streaming, optionally with the diagonal ``rate`` inside the factor."""
    return Streaming(rate)


def streaming_factor_fn(model: Model, rate=None):
    """``factor_fn(t, y, args) -> StreamingFactor`` for :class:`spectrax.LawsonSolver` on the Dark dict state."""
    Ns, Np, Nm, Nn = model.Ns, model.Np, model.Nm, model.Nn
    shape = (Ns, Np, Nm, Nn, *model.shape)
    ck_shape = (Ns * Nn * Nm * Np, *model.shape)
    rate_fn = static_rate(model) if rate is True else rate
    eig = [hermite_axis_eig(shape[ax]) for ax in (3, 2, 1)]
    lam = tuple(tuple(float(x) for x in e[0]) for e in eig)
    V = tuple(tuple(tuple(float(x) for x in row) for row in e[1]) for e in eig)
    nabla = model.p["nabla"]

    def to_ck(y):
        return y["Ck"].reshape(shape)

    def from_ck(y, ck):
        return {**y, "Ck": ck.reshape(ck_shape).astype(y["Ck"].dtype)}

    def factor_fn(t, y, args):
        u_s, a_s = basis_of(model, y)
        r = None
        if rate_fn is not None:
            r = jnp.asarray(rate_fn(t, y, model)["Ck"], float).reshape(Ns, Np, Nm, Nn, 1, 1, 1)
        return StreamingFactor(
            kappa=(nabla[0], nabla[1], nabla[2]), a=jnp.asarray(a_s, float).reshape(Ns, 3),
            u=jnp.asarray(u_s, float).reshape(Ns, 3), rate=r, lam=lam, V=V, to_ck=to_ck, from_ck=from_ck)

    return factor_fn
