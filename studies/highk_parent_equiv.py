"""highk: the study closure (studies/highk_run.py::_install_field_closure) equals the parent PR helper
spectrax.field_scaled_closure_rate + hypercollision_spectrum(order=2) used as a per-species nu. Run with the
SPECTRAX PR checkout first on PYTHONPATH:  PYTHONPATH=<spectrax-pr>:. python studies/highk_parent_equiv.py"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectrax  # noqa: E402
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from darkspectrax._model import basis_of  # noqa: E402
import highk_run  # noqa: E402
from lane_c_run import model, seeds  # noqa: E402

m = model(0.1, 32, 16)
y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [(s, k, 30 * A) for s, k, A in seeds(0)]))
rng = np.random.default_rng(0)
noise = 1e-3 * (rng.normal(size=y["Ck"].shape) + 1j * rng.normal(size=y["Ck"].shape)) * float(jnp.abs(y["Ck"]).max())
y = {**y, "Ck": y["Ck"] + noise, "B": y["B"].at[0, 0].set(0.3).at[1, 0].set(0.06)}
r0 = _sim.rhs(1.3, y, m)["Ck"]
highk_run._install_field_closure(m, 0.8)
r1 = _sim.rhs(1.3, y, m)["Ck"]
F = jnp.fft.irfftn(y["Fk"] * m.p["mask23"], s=(m.Nz, m.Ny, m.Nx), axes=(-1, -3, -2), norm="forward")
u, a = basis_of(m, y)
nu = spectrax.field_scaled_closure_rate(F, a, m.p["qs"], m.p["Omega_cs"], m.Nn, m.Nm, m.Np, m.Ns, 0.8)
col = spectrax.hypercollision_spectrum(m.Nn, m.Nm, m.Np, 2)
Ck = y["Ck"].reshape(m.Ns, m.Np, m.Nm, m.Nn, *y["Ck"].shape[1:])
ref = (-nu * col[None, :, :, :, None, None, None] * Ck).reshape(r0.shape)
print(float(jnp.abs(ref).max()), float(jnp.abs(r1-r0).max()), np.asarray(nu).ravel(), bool(jnp.isnan(r0).any()))
err = float(jnp.abs((r1 - r0) - ref).max() / jnp.abs(ref).max())
print("spectrax:", spectrax.__file__, " max rel diff study closure vs parent helper:", f"{err:.2e}")
assert err < 1e-12
