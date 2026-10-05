"""highk: the study closure (studies/highk_run.py::_install_field_closure, which now calls the pinned parent's
spectrax.field_scaled_closure_rate from SPECTRAX PR #66) equals the original study implementation kept below
(_reference_damp: E_x and a_x only, s(n) on the n axis). Run: python studies/highk_parent_equiv.py"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from darkspectrax._model import basis_of  # noqa: E402
import highk_run  # noqa: E402
from lane_c_run import model, seeds  # noqa: E402


def _reference_damp(m, y, c):
    """Original study implementation (Dark-SPECTRAX highk 99ee01f), kept as the equivalence reference."""
    Ns, Nn = m.Ns, m.Nn
    n = np.arange(Nn, dtype=float)
    sn = jnp.asarray(n * (n - 1) * (n - 2) / ((Nn - 1) * (Nn - 2) * (Nn - 3)))
    qm = jnp.asarray(np.abs(np.asarray(m.qs)) * np.asarray(m.Omega_cs))
    Ex = jnp.fft.irfftn(y["Fk"][0] * m.p["mask23"], s=(m.Nz, m.Ny, m.Nx), axes=(-1, -3, -2), norm="forward")
    Emax = jnp.max(jnp.abs(Ex - jnp.mean(Ex)))
    a_x = basis_of(m, y)[1].reshape(Ns, 3)[:, 0]
    rate = c * qm * jnp.sqrt(2.0 * Nn) * Emax / a_x
    Ck = y["Ck"].reshape(Ns, Nn, *y["Ck"].shape[1:])
    return (-(rate[:, None, None, None, None] * sn[None, :, None, None, None]) * Ck).reshape(y["Ck"].shape)


m = model(0.1, 32, 16)
y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [(s, k, 30 * A) for s, k, A in seeds(0)]))
rng = np.random.default_rng(0)
noise = 1e-3 * (rng.normal(size=y["Ck"].shape) + 1j * rng.normal(size=y["Ck"].shape)) * float(jnp.abs(y["Ck"]).max())
y = {**y, "Ck": y["Ck"] + noise, "B": y["B"].at[0, 0].set(0.3).at[1, 0].set(0.06)}
r0 = _sim.rhs(1.3, y, m)["Ck"]
highk_run._install_field_closure(m, 0.8)
r1 = _sim.rhs(1.3, y, m)["Ck"]
ref = _reference_damp(m, y, 0.8)
err = float(jnp.abs((r1 - r0) - ref).max() / jnp.abs(ref).max())
print("parent:", ds.PARENT_COMMIT, " max|ref|", float(jnp.abs(ref).max()), " nan:", bool(jnp.isnan(r1).any()))
print("max rel diff parent-helper closure vs original study closure:", f"{err:.2e}")
assert err < 1e-12
