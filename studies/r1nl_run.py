"""r1nl: one chunked, resumable H05-setup run with the strong field-scaled closure integrated exactly (Lawson).

Gates and case lists: studies/r1nl_gates.py (declared before any production run). Setup = studies/r1_common.py (mobile ions,
m_i/m_e = 1836, T_i = T_e, L = 40, pump frame + remaps, seeds r0, omega = omega_L, noise floor 1e-14, rtol 1e-10, atol 1e-14),
closure c via darkspectrax.field_closure(model, c) with run(..., integrating_factor=rate_fn). v_q/v_te is the effective force amplitude.
kinds: pre (coherent prescribed drive), phase (finite coherence, v2 definition: variance 2/(tau h)), sc (self-consistent finite reservoir).

Restart: --chunk C integrates [t0, t0 + C]; rerunning the same command continues from the last stored part (absolute time).
Run: python studies/r1nl_run.py --kind pre --vq 0.05 --Nx 32 --Nn 64 --c 64 --chunk 200 [--out DIR]
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
import lane_c_run as lc  # noqa: E402
import r1_common as c1  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402

OUT = Path(__file__).resolve().parent / "r1nl" / "runs"
SEG, NSAVE, FLOOR = lc.SEG, lc.NSAVE, 1e-14


def eta_of(vq, u):
    return vq / np.sqrt(2 * u)


def case_name(a):
    s = f"{a.kind}_vq{a.vq:g}_Nx{a.Nx}_Nn{a.Nn}_c{a.c:g}"
    if a.kind == "phase":
        s += f"_tau{a.tau:g}_r{a.real}"
    if a.kind == "sc":
        s += f"_u{a.u:g}"
    return s + a.tag


def build(a):
    E0 = a.vq * c1.vte
    if a.kind == "pre":
        m = c1.mobile_model(a.Nx, a.Nn, mode="prescribed_drive", E0=E0)
        return m, c1.initial(m), 0.0
    if a.kind == "phase":
        z = c1.noise_knots(a.tau, a.real, fac=2.0)
        m = c1.mobile_model(a.Nx, a.Nn, mode="prescribed_drive", E0=E0, cls=c1.PhasedModel, z=tuple(z))
        return m, c1.initial(m), 0.0
    eta = eta_of(a.vq, a.u)
    m = c1.mobile_model(a.Nx, a.Nn, mode="self_consistent", eta=eta, Omega_D=c1.w_L)
    return m, c1.initial(m, E_D0=E0 / eta), eta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("pre", "phase", "sc"), required=True)
    ap.add_argument("--vq", type=float, required=True)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--c", type=float, default=64.0)
    ap.add_argument("--tau", type=float, default=np.inf)
    ap.add_argument("--real", type=int, default=0)
    ap.add_argument("--u", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--chunk", type=float, default=1e9)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else OUT
    case = case_name(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(out_dir.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    m, y0, eta = build(a)
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
    ipart = len(parts)
    t1 = min(a.T, t0 + max(SEG, SEG * (a.chunk // SEG)))
    stats, run0 = [], _sim.run

    def counted(*args, **kw):
        kw["t0"] = kw.get("t0", 0.0) + t0
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    _sim.rhs, rate = ds.field_closure(m, a.c)
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, t1 - t0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=a.max_steps,
                          noise_floor=FLOOR, integrating_factor=rate)
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    Ks, Th, Tx, Q = lc.diagnostics(m, out)
    st = np.array(stats)
    t_end = float(out["t"][-1])
    status = out["status"] if (out["status"] != "success" or t_end >= t1 - 1e-9) else "short"
    rec = {"case": case, "part": ipart, "t_start": t0, "t_end": t_end, "status": status, "failure_reason": out["failure_reason"],
           "events": len(out["events"]), "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()),
           "compile_time": float(st[:, 2].sum()), "run_time": float(st[:, 3].sum()), "wall_time": wall,
           "kind": a.kind, "vq": a.vq, "Nx": a.Nx, "Nn": a.Nn, "c": a.c, "tau": a.tau, "real": a.real, "u": a.u, "eta": eta,
           "repository_commit": _sim._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "command": "python studies/r1nl_run.py " + " ".join(sys.argv[1:])}
    stem = out_dir / f"{case}.part{ipart}"
    np.savez_compressed(f"{stem}.npz", t=out["t"], K_species=Ks, Th=Th, Tx=Tx, Q=Q, W=out["W"], U_gamma=out["U_gamma"],
                        U_D=out["U_D"], Ebar=np.asarray(out["Fk"][:, 0, 0, 0, 0]).real,
                        EDbar=np.asarray(out["Dk"][:, 0, 0, 0, 0]).real, Ek2=np.abs(np.asarray(out["Fk"][:, 0, 0, :, 0])) ** 2,
                        seg_stats=st, y_Ck=out["Ck"][-1], y_Fk=out["Fk"][-1], y_Dk=out["Dk"][-1],
                        y_W=out["W"][-1].astype(complex), y_B=out["B"][-1].astype(complex))
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
