"""highk (a): de-aliasing consistency of the pump-frame RHS. A random two-species state band-limited to |k| <= 5 at
Nx = 16 (strict 2/3 mask keeps |k| <= 5) and the SAME state zero-padded to Nx = 24 with the mask overridden to
|k| <= 5 (padded-convolution reference: products reach |k| <= 10 < 24 - 5, so nothing aliases onto a kept mode).
Rejection criterion: if dCk, dFk, u_dot differ by more than 1e-12 relative on the kept modes, de-aliasing is
inconsistent somewhere (force, current, moving-basis terms).  Also: Nx = 8 vs Nx = 16 with |k| <= 2 kept.
Run: python studies/highk_alias_check.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from lane_c_run import model  # noqa: E402

rng = np.random.default_rng(1)


def state(m, K, base=None):
    y = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], []))
    nk = m.Nx // 2 + 1
    Ck = np.array(y["Ck"]).reshape(m.Ns, m.Nn, nk)
    if base is None:
        pert = (rng.normal(size=(m.Ns, m.Nn, K + 1)) + 1j * rng.normal(size=(m.Ns, m.Nn, K + 1))) * 1e-3
        pert[:, :, 0] = pert[:, :, 0].real * 0
        pert *= np.abs(Ck[:, :1, :1])
    else:
        pert = base
    Ck[:, :, :K + 1] += pert
    y = {**y, "Ck": jnp.asarray(Ck.reshape(-1, 1, nk, 1)), "B": y["B"].at[0, 0].set(0.4).at[0, 3].set(-0.4 / 1836)}
    return ds.consistent_fields(m, y), pert


for Na, Nb, K in ((16, 24, 5), (8, 16, 2)):
    ma, mb = model(0.1, 16, Na), model(0.1, 16, Nb)
    mb.p["mask23"] = jnp.asarray(np.arange(Nb // 2 + 1)[None, :, None] <= K)
    ya, pert = state(ma, K)
    yb, _ = state(mb, K, pert)
    ra, rb = ds.rhs(3.7, ya, ma), ds.rhs(3.7, yb, mb)
    for key in ("Ck", "Fk", "B"):
        A, Bv = np.asarray(ra[key]), np.asarray(rb[key])
        if key != "B":
            A, Bv = A[..., :K + 1, :], Bv[..., :K + 1, :]
        print(f"Nx {Na} vs padded {Nb}, kept |k|<={K}: {key} max rel diff {np.abs(A - Bv).max() / np.abs(A).max():.2e}")
    hi = np.abs(np.asarray(rb["Ck"])[..., K + 1:, :]).max()
    print(f"  padded run: largest RHS on a dropped mode = {hi:.2e}")
