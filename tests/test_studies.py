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
