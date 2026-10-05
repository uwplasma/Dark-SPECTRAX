"""highk: diagnostic H05 pump-frame Hermite runs at v_q/v_te = 0.1 (lane_c_run.py r0 setup, nu = 0) that record, per
save, the Hermite spectrum per species and per Fourier mode, |C_{s,n}(k)|^2, plus the basis (u_s, a_s) and step
statistics. Options select the decisive experiments:
  --Nx, --Nn                  resolution
  --kmax K                    override the 2/3 mask to keep |k| <= K (padded-convolution reference: Nx=24, K=5
                              must reproduce Nx=16 to round-off if de-aliasing is consistent)
  --fixed-dt DT               constant-step Dopri8 instead of PID (stiffness/step-size test)
  --no-remap                  disable segment remaps (pump frame shift only, initial width)
Run: python studies/highk_run.py --vq 0.1 --Nx 16 --Nn 64 --T 480 [--tag X]  ->  studies/highk/runs/<case>.npz/json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from lane_c_run import NSAVE, SEG, model, seeds  # noqa: E402

OUT = Path(__file__).resolve().parent / "highk" / "runs"


def _install_field_closure(m, c):
    """Field-scaled order-2 hypercollision on the AW top modes (study implementation, wraps the Dark RHS).

    The truncated AW Fourier-Hermite operator in a field E has spurious growth modes living in the top third of the
    Hermite ladder at the highest kept |k| (studies/highk_eig.py), with rate ~0.3 sqrt(2 Nn) |E|/a. A fixed nu is
    overtaken as the H05 k1 field grows secularly, so the rate here follows the field:
        dC_s/dt += -c |q/m|_s sqrt(2 Nn) max_x|E_x - <E_x>| / a_s * s(n) C_s,
    s(n) = n(n-1)(n-2)/((N-1)(N-2)(N-3)) (zero for n <= 2: density, momentum and energy rows untouched).
    """
    import jax.numpy as jnp
    from darkspectrax._model import basis_of

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, default=0.1)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--Nx", type=int, default=16)
    ap.add_argument("--T", type=float, default=480.0)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--fixed-dt", type=float, default=None)
    ap.add_argument("--no-remap", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed-scale", type=float, default=1.0)
    ap.add_argument("--field-nu", type=float, default=0.0,
                    help="c in the field-scaled closure rate nu_s = c |q/m|_s sqrt(2 Nn) max|E - E0| / a_s on s(n)")
    a = ap.parse_args(argv)
    m = model(a.vq, a.Nn, a.Nx)
    if a.kmax is not None:
        kx = np.arange(a.Nx // 2 + 1)[None, :, None]
        m.p["mask23"] = jnp.asarray(kx <= a.kmax)
    stats = []
    run0 = _sim.run

    def counted(*args, **kw):
        if a.fixed_dt is not None:
            kw["fixed_dt"] = a.fixed_dt
            kw["max_steps"] = int(SEG / a.fixed_dt) + 10
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    if a.field_nu > 0:
        _install_field_closure(m, a.field_nu)
    trig = {"shift_on": 1e9, "width_on": 1e9} if a.no_remap else None
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [(sp, k, A * a.seed_scale) for sp, k, A in seeds(0)]))
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, a.T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000,
                          trigger=trig)
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    Ck = out["Ck"][good].reshape(good.sum(), m.Ns, m.Nn, m.Nx // 2 + 1)
    spec = np.abs(Ck) ** 2
    case = f"vq{a.vq:g}_Nx{a.Nx}_Nn{a.Nn}" + (f"_kmax{a.kmax}" if a.kmax is not None else "") \
        + (f"_dt{a.fixed_dt:g}" if a.fixed_dt else "") + ("_noremap" if a.no_remap else "") + (f"_seed{a.seed_scale:g}" if a.seed_scale != 1 else "") + (f"_fnu{a.field_nu:g}" if a.field_nu else "") + a.tag
    st = np.array(stats)
    rec = {"case": case, "status": out["status"], "failure_reason": out["failure_reason"],
           "t_reached": float(out["t"][good][-1]), "events": len(out["events"]),
           "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()), "compile_time": float(st[:, 2].sum()),
           "run_time": float(st[:, 3].sum()), "wall_time": wall, "repository_commit": _sim._git_sha(),
           "parent_commit": ds.PARENT_COMMIT, "command": "python studies/highk_run.py " + " ".join(sys.argv[1:])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{case}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(OUT / f"{case}.npz", t=out["t"][good], spec=spec, B=out["B"][good], W=out["W"][good],
                        K=out["K"][good], Ek2=np.abs(out["Fk"][good][:, 0, 0, :, 0]) ** 2, seg_stats=st,
                        Ck_last=out["Ck"][good][-1])
    print(json.dumps(rec, default=float), flush=True)


if __name__ == "__main__":
    main()
