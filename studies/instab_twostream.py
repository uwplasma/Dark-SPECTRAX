"""instab task 2: is the README two-stream movie negativity (-31% of max f by t = 29.25, Nn/2Nn agreeing to 0.8%)
the AW truncation instability (highk.md), or x-truncation?

Same case as studies/figures.py phase two_stream (ordinary model, u = +-1, v_t = 0.3, k = 0.4, nu = 1 order-1 parent
hypercollision, Dopri8 rtol 1e-8), varying Nx and adding the field-scaled AW closure of SPECTRAX PR #66
(c |q/m| sqrt(2N) max|E - <E>| / a on n(n-1)(n-2)/((N-1)(N-2)(N-3)); same form as studies/highk_run.py on branch highk).
Records min f / max f (reconstructed on 96 x 160 like the movie) and |E_k1| at each frame up to T.
Rejection criteria (declared before running):
  - AW truncation instability: negativity must drop by >= 2x with c = 1 at fixed Nx and grow with Nx at c = 0.
  - x-truncation (unresolved filamentation): negativity must drop by >= 2x from Nx 16 -> 64 at c = 0, c-insensitive.
  - neither: negativity changes < 2x across both -> intrinsic to the velocity reconstruction at this Nn.
Run: python studies/instab_twostream.py NX NN C T -> studies/instab/twostream_Nx<NX>_Nn<NN>_c<C>.json
"""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
import figures as F  # noqa: E402
from darkspectrax._model import basis_of  # noqa: E402


def install_closure(m, c, Nx):
    Ns, Nn = m.Ns, m.Nn
    n = np.arange(Nn, dtype=float)
    sn = jnp.asarray(n * (n - 1) * (n - 2) / ((Nn - 1) * (Nn - 2) * (Nn - 3)))
    qm = jnp.asarray(np.abs(np.asarray(m.qs)) * np.asarray(m.Omega_cs))
    rhs0 = _sim.rhs
    mask = m.p["mask23"]

    def rhs(t, y, model):
        out = rhs0(t, y, model)
        Ex = jnp.fft.irfftn(y["Fk"][0] * mask, s=(m.Nz, m.Ny, m.Nx), axes=(-1, -3, -2), norm="forward")
        Emax = jnp.max(jnp.abs(Ex - jnp.mean(Ex)))
        a_x = basis_of(model, y)[1].reshape(Ns, 3)[:, 0]
        rate = c * qm * jnp.sqrt(2.0 * Nn) * Emax / a_x
        Ck = y["Ck"].reshape(Ns, Nn, *y["Ck"].shape[1:])
        damp = -(rate[:, None, None, None, None] * sn[None, :, None, None, None]) * Ck
        return {**out, "Ck": out["Ck"] + damp.reshape(y["Ck"].shape)}

    _sim.rhs = rhs


if __name__ == "__main__":
    Nx, Nn, c, T = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    F.PHASE_CASES["two_stream"] = {**F.PHASE_CASES["two_stream"], "Nx": Nx, "T": T}
    if c > 0:
        m, _ = F.phase_model("two_stream", False, Nn)
        install_closure(m, c, Nx)
    r = F.phase_simulate("two_stream", False, Nn)
    ok = np.isfinite(r["t_frames"]) & np.array([np.all(np.isfinite(C)) for C in r["Ck"]])
    x, v, f = F.phase_f("two_stream", False, Nn, r["Ck"][ok])
    neg = f.min(axis=(1, 2)) / f.max(axis=(1, 2))
    out = Path(__file__).resolve().parent / "instab"
    out.mkdir(exist_ok=True)
    rec = {"Nx": Nx, "Nn": Nn, "c": c, "T": T, "status": str(r["status"]),
           "failure_reason": str(r["failure_reason"]), "steps": int(r["steps"]), "run_time": float(r["run_time"]),
           "t": r["t_frames"][ok].tolist(), "minf_over_maxf": neg.tolist(),
           "E1abs": np.abs(np.interp(r["t_frames"][ok], r["t"][np.isfinite(r["t"])],
                                     np.abs(r["E1"])[np.isfinite(r["t"])])).tolist(),
           "command": "python studies/instab_twostream.py " + " ".join(sys.argv[1:])}
    (out / f"twostream_Nx{Nx}_Nn{Nn}_c{c:g}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    i = np.searchsorted(rec["t"], 29.25)
    print(json.dumps({k: rec[k] for k in ("Nx", "Nn", "c", "status", "failure_reason", "steps", "run_time")}),
          "t", rec["t"][min(i, len(neg) - 1)], "min/max f", neg[min(i, len(neg) - 1)], "worst", neg.min(), flush=True)
