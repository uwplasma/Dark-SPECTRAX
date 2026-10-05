"""nxconv: H05 strong-drive Hermite runs for x-convergence (Nx 32/64/128), mobile ions, pump frame + remaps,
SPECTRAX PR #66 field-scaled closure (c = 1 by default), rtol 1e-10, atol 1e-14, optional parent noise floor.

Setup = studies/lane_c_run.py r0 (the gate seeds, omega = sqrt(1 + 1/1836)). The closure rate is the parent helper
spectrax._model.field_scaled_closure_rate (PR #66, in integration/dark-baseline 16400f8) applied on the order-2
spectrum s(n) = n(n-1)(n-2)/((N-1)(N-2)(N-3)) (density, momentum and energy rows untouched); this is the study
wrapper of studies/highk_run.py (highk branch) with the rate taken from the parent instead of re-implemented.

Restart/resume: --chunk C integrates [t0, min(t0 + C, T)] and stores the final state in
studies/nxconv/runs/<case>.part<i>.npz; rerunning the same command continues from the last part (the parent run's
t0 is offset, so the drive phase and the time axis are absolute). Each part is one process under the 15 min cap.
The run stops at the parent's positivity check (K_s or k=0 T_x <= 0) or at a solver failure; further reruns
then do nothing.

Per save: t, K_species, k=0 thermal energy, T_x, W (ledger), |E_x(k)|^2 for all rfft modes, and per species the
Hermite energy per order summed over k (Nn) and the top-third Hermite energy fraction per k (Nx/2 + 1).
Run: python studies/nxconv_run.py --vq 0.1 --Nx 64 --Nn 64 --T 1000 --chunk 200 [--noise-floor 1e-14]
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
from lane_c_run import NSAVE, SEG, diagnostics, model, seeds  # noqa: E402

OUT = Path(__file__).resolve().parent / "nxconv" / "runs"


def install_closure(m, c):
    """Wrap the Dark RHS with -nu_s s(n) C_s, nu_s from the parent PR #66 helper."""
    from spectrax._model import field_scaled_closure_rate

    from darkspectrax._model import basis_of

    Ns, Nn = m.Ns, m.Nn
    n = np.arange(Nn, dtype=float)
    sn = jnp.asarray(n * (n - 1) * (n - 2) / ((Nn - 1) * (Nn - 2) * (Nn - 3)))
    qs, Om = jnp.asarray(np.asarray(m.qs, float)), jnp.asarray(np.asarray(m.Omega_cs, float))
    rhs0, mask = _sim.rhs, m.p["mask23"]

    def rhs(t, y, model):
        out = rhs0(t, y, model)
        Ex = jnp.fft.irfftn(y["Fk"][0] * mask, s=(m.Nz, m.Ny, m.Nx), axes=(-1, -3, -2), norm="forward")
        F = jnp.stack([Ex, jnp.zeros_like(Ex), jnp.zeros_like(Ex)])
        nu = field_scaled_closure_rate(F, basis_of(model, y)[1], qs, Om, Nn, 1, 1, Ns, c).reshape(Ns)
        Ck = y["Ck"].reshape(Ns, Nn, *y["Ck"].shape[1:])
        damp = -(nu[:, None, None, None, None] * sn[None, :, None, None, None]) * Ck
        return {**out, "Ck": out["Ck"] + damp.reshape(y["Ck"].shape)}

    _sim.rhs = rhs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, default=0.1)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--chunk", type=float, default=1e9)
    ap.add_argument("--field-nu", type=float, default=1.0)
    ap.add_argument("--noise-floor", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    case = f"vq{a.vq:g}_Nx{a.Nx}_Nn{a.Nn}_c{a.field_nu:g}" + (f"_nf{a.noise_floor:g}" if a.noise_floor else "") + a.tag
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sorted(OUT.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    m = model(a.vq, a.Nn, a.Nx)
    if parts:
        last = np.load(parts[-1])
        info = json.loads(Path(str(parts[-1])[:-4] + ".json").read_text())
        if info["status"] not in ("success", "short") or info["t_end"] >= a.T - 1e-9:
            print(f"{case}: nothing to do ({info['status']}, t_end {info['t_end']:g})")
            return 0
        t0 = info["t_end"]
        y0 = {"Ck": jnp.asarray(last["y_Ck"]), "Fk": jnp.asarray(last["y_Fk"]), "Dk": jnp.asarray(last["y_Dk"]),
              "W": jnp.asarray(last["y_W"]), "B": jnp.asarray(last["y_B"])}
    else:
        t0 = 0.0
        y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(0)))
    ipart = len(parts)
    t1 = min(a.T, t0 + max(SEG, SEG * (a.chunk // SEG)))  # whole segments
    stats, run0 = [], _sim.run

    def counted(*args, **kw):  # absolute time: offset the parent's segment start
        kw["t0"] = kw.get("t0", 0.0) + t0
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    if a.field_nu > 0:
        install_closure(m, a.field_nu)
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, t1 - t0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=a.max_steps,
                          **({"noise_floor": a.noise_floor} if a.noise_floor else {}))
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    Ks, Th, Tx, Q = diagnostics(m, out)
    nt = out["t"].size
    Ck = np.asarray(out["Ck"]).reshape(nt, m.Ns, m.Nn, m.Nx // 2 + 1)
    E2 = np.abs(Ck) ** 2
    herm_n = E2.sum(-1)  # (nt, Ns, Nn)
    top = E2[:, :, (2 * m.Nn) // 3:, :].sum(2) / np.maximum(E2.sum(2), 1e-300)  # (nt, Ns, Nk)
    st = np.array(stats)
    t_end = float(out["t"][-1])
    status = out["status"] if (out["status"] != "success" or t_end >= t1 - 1e-9) else "short"
    rec = {"case": case, "part": ipart, "t_start": t0, "t_end": t_end, "status": status,
           "failure_reason": out["failure_reason"], "events": len(out["events"]),
           "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()), "compile_time": float(st[:, 2].sum()),
           "run_time": float(st[:, 3].sum()), "wall_time": wall, "repository_commit": _sim._git_sha(),
           "parent_commit": ds.PARENT_COMMIT, "command": "python studies/nxconv_run.py " + " ".join(sys.argv[1:])}
    stem = OUT / f"{case}.part{ipart}"
    np.savez_compressed(f"{stem}.npz", t=out["t"], K_species=Ks, Th=Th, Tx=Tx, W=out["W"],
                        Ek2=np.abs(out["Fk"][:, 0, 0, :, 0]) ** 2, herm_n=herm_n, top_frac=top, seg_stats=st,
                        y_Ck=out["Ck"][-1], y_Fk=out["Fk"][-1], y_Dk=out["Dk"][-1], y_W=out["W"][-1].astype(complex),
                        y_B=out["B"][-1].astype(complex))
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
