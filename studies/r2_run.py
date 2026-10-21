"""R2: one chunked, resumable collisional driven-plasma run (gates and case list: studies/r2_gates.py, declared before production).

Setup = h08.py (H05 mobile-ion setup, Dougherty collisions) with the drive frequency free, closure field_closure(model, c = 64)
integrated exactly (Lawson, as r1nl_run.py) and the h08 per-save diagnostics (collisional power, mean-field identity residual).
Run: python studies/r2_run.py --conv EM --vq 0.03 --nu 1e-2 --omega res --Nx 32 --Nn 64 [--control] [--chunk 200] [--T 1000]
Rerun continues from the last stored part (absolute time).
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
import h08  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from lane_c_run import Lx, NSAVE, SEG, a_e, a_i, diagnostics, seeds, vte, w_tot  # noqa: E402

OUT = Path(__file__).resolve().parent / "r2" / "runs"
MI, NF = 1836.0, 1e-14
COLL = {"EM": "dougherty", "G": "dougherty_g"}


def omega_of(tag):
    return w_tot if tag == "res" else float(tag)


def case_name(conv, vq, nu, om, Nx, Nn, control, seed_scale=1.0):
    return (f"{conv}_vq{vq:g}_nu{nu:g}_w{om}_Nx{Nx}_Nn{Nn}" + ("_control" if control else "")
            + (f"_seed{seed_scale:g}" if seed_scale != 1.0 else ""))


def model(conv, vq, Nx, Nn, nu, om, collisional=True):
    kw = dict(collision_model=COLL[conv], nu_dougherty=h08.nu_matrix(nu)) if (nu > 0 and collisional) else {}
    return ds.Model(Nx=Nx, Nn=Nn, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / MI), alpha_s=(a_e,) * 3 + (a_i,) * 3,
                    u_s=(0.0,) * 6, mode="prescribed_drive", E_drive=(vq * vte, 0.0, 0.0), omega_drive=omega_of(om),
                    frame="pump", **kw)


def run(a):
    case = case_name(a.conv, a.vq, a.nu, a.omega, a.Nx, a.Nn, a.control, a.seed_scale)
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(out_dir.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    m = model(a.conv, a.vq, a.Nx, a.Nn, a.nu, a.omega)
    mfree = model(a.conv, a.vq, a.Nx, a.Nn, a.nu, a.omega, collisional=False)
    if parts:
        last = np.load(parts[-1])
        info = json.loads(Path(str(parts[-1])[:-4] + ".json").read_text())
        if info["status"] not in ("success", "short") or info["t_end"] >= a.T - 1e-9:
            print(f"{case}: nothing to do ({info['status']}, t_end {info['t_end']:g})")
            return 0
        t0 = info["t_end"]
        y0 = {k: jnp.asarray(last[f"y_{k}"]) for k in ("Ck", "Fk", "Dk", "W", "B")}
    else:
        t0 = 0.0
        y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], [] if a.control else [(sp, k, A * a.seed_scale) for sp, k, A in seeds(0)]))
    ipart = len(parts)
    t1 = min(a.T, t0 + max(SEG, SEG * (a.chunk // SEG)))
    stats, run0, rhs0 = [], _sim.run, _sim.rhs

    def counted(*args, **kw):
        kw["t0"] = kw.get("t0", 0.0) + t0
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    try:
        _sim.rhs, rate = ds.field_closure(m, a.c)
        tic = time.perf_counter()
        out = ds.run_adaptive(m, y0, t1 - t0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=a.max_steps,
                              noise_floor=NF, integrating_factor=rate)
    finally:
        _sim.run, _sim.rhs = run0, rhs0
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    Ks, Th, Tx, Q = diagnostics(m, out)
    f = h08.diag_fn(m, mfree)
    rows = []
    for i in range(out["t"].size):
        y = {"Ck": jnp.asarray(out["Ck"][i]), "Fk": jnp.asarray(out["Fk"][i]), "Dk": jnp.asarray(out["Dk"][i]),
             "W": jnp.asarray(out["W"][i], complex), "B": jnp.asarray(out["B"][i], complex)}
        rows.append([np.asarray(v) for v in f(out["t"][i], y)])
    P, R, R0, dJc, Eb = (np.array(x) for x in zip(*rows))
    st = np.array(stats)
    t_end = float(out["t"][-1])
    status = out["status"] if (out["status"] != "success" or t_end >= t1 - 1e-9) else "short"
    rec = {"case": case, "conv": a.conv, "vq_over_vte": a.vq, "nu_ei": a.nu, "omega": a.omega, "omega_value": omega_of(a.omega),
           "Nx": a.Nx, "Nn": a.Nn, "control": a.control, "seed_scale": a.seed_scale, "closure_c": a.c, "noise_floor": NF, "part": ipart,
           "t_start": t0, "t_end": t_end, "status": status, "failure_reason": out["failure_reason"], "events": len(out["events"]),
           "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()), "compile_time": float(st[:, 2].sum()),
           "run_time": float(st[:, 3].sum()), "wall_time": wall, "repository_commit": _sim._git_sha(),
           "parent_commit": ds.PARENT_COMMIT,
           "command": f"python studies/r2_run.py --conv {a.conv} --vq {a.vq:g} --nu {a.nu:g} --omega {a.omega} --Nx {a.Nx} "
                      f"--Nn {a.Nn} --c {a.c:g} --T {a.T:g} --chunk {a.chunk:g}" + (" --control" if a.control else "")}
    stem = out_dir / f"{case}.part{ipart}"
    np.savez_compressed(f"{stem}.npz", t=out["t"], K_species=Ks, Th=Th, Tx=Tx, W=out["W"], ledger=out["ledger_defect"][0],
                        Ek2=np.abs(out["Fk"][:, 0, 0, :, 0]) ** 2, P=P, R=R, R0=R0, dJc=dJc, Ebar=Eb, seg_stats=st,
                        y_Ck=out["Ck"][-1], y_Fk=out["Fk"][-1], y_Dk=out["Dk"][-1], y_W=out["W"][-1].astype(complex),
                        y_B=out["B"][-1].astype(complex))
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", choices=("EM", "G"), default="EM")
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--nu", type=float, required=True)
    ap.add_argument("--omega", default="res")
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--c", type=float, default=64.0)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--chunk", type=float, default=200.0)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--seed-scale", type=float, default=1.0)  # follow-up F (studies/r2_followup.py); default unchanged
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--out", default=None)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
