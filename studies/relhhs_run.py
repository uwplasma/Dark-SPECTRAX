"""relhhs: relativistic Hermite (Dark-SPECTRAX inv_c2 = 1, velocities in c, T_e = T_i = 1e-3 m_e c^2) on the HHS resonant set-up.

Set-up = studies/lane_c_run.py r0 (v_q/v_te = 0.03, mobile ions m_i/m_e = 1836, L = 40 c/omega_pe, omega = sqrt(1 + 1/1836), deterministic
nxconv seeds, pump frame + remap check every 20), relativistic RHS (SPECTRAX integration/dark-baseline-2 relativistic Hermite #88 + its pump-frame F0
support), field-scaled closure of strength c (SPECTRAX #66 rate, exact in the Lawson integrating factor; streaming explicit: the exact streaming factor is
the nonrelativistic ladder), rtol 1e-10, atol 1e-14. The closure is the study wrapper of darkspectrax.field_closure, unchanged by relativity.

Per save: t, K_species ((gamma - 1) m c^2 via the nodal Galerkin matrix), W (em, D, ext), Ebar = <E_x>, |E_x(k)|^2 for all rfft modes, the HHS measure
E1D_e = (1/2) m (<v^2> - <v>^2/n) (v = u/gamma, k = 0, speed_moments), the electron basis (u_s, alpha_s), Hermite energy per order, top-third Hermite
fraction per k, and f_e positivity (min and max of the reconstructed f_e(x, xi), |xi| <= 10). Chunked and resumable as nxconv_run.py.
Run: python studies/relhhs_run.py --Nx 32 --Nn 64 --field-nu 64 --T 1000 --chunk 40 [--inv-c2 1] -> studies/relhhs/runs/<case>.part<i>.{npz,json}
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402
from darkspectrax._model import basis_of  # noqa: E402
from lane_c_run import NSAVE, SEG, model as lane_model, seeds  # noqa: E402

OUT = Path(__file__).resolve().parent / "relhhs" / "runs"
XI = np.linspace(-10.0, 10.0, 321)


def hermite_functions(Nn, xi):
    """h_n(xi) = H_n(xi) exp(-xi^2) / sqrt(2^n n!) for n < Nn (three-term recurrence of the orthonormal Hermite polynomials)."""
    h = np.zeros((Nn, xi.size))
    h[0] = 1.0
    if Nn > 1:
        h[1] = np.sqrt(2.0) * xi
    for n in range(1, Nn - 1):
        h[n + 1] = (np.sqrt(2.0) * xi * h[n] - np.sqrt(n) * h[n - 1]) / np.sqrt(n + 1.0)
    return h * np.exp(-xi ** 2)[None]


def f_extrema(m, Ck, B):
    """(min, max) over x and xi of the electron distribution at each save (shape (nt,)); the overall positive constant is irrelevant."""
    H = hermite_functions(m.Nn, XI)
    mn, mx = [], []
    for i in range(Ck.shape[0]):
        c = np.asarray(Ck[i]).reshape(m.Ns, m.Nn, m.Nx // 2 + 1)[0]
        c = np.fft.irfft(c, n=m.Nx, axis=-1, norm="forward")
        f = np.einsum("nx,nj->xj", c, H)
        mn.append(f.min())
        mx.append(f.max())
    return np.array(mn), np.array(mx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", type=float, default=0.03)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--Nx", type=int, default=32)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--chunk", type=float, default=40.0)
    ap.add_argument("--field-nu", type=float, default=64.0)
    ap.add_argument("--inv-c2", type=float, default=1.0)
    ap.add_argument("--noise-floor", type=float, default=1e-14)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--real", type=int, default=0, help="seed realization of lane_c_run (-1: unseeded homogeneous control)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else OUT
    case = f"rel_vq{a.vq:g}_Nx{a.Nx}_Nn{a.Nn}_c{a.field_nu:g}" + ("" if a.inv_c2 == 1.0 else f"_ic{a.inv_c2:g}") + a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(out_dir.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    m = dataclasses.replace(lane_model(a.vq, a.Nn, a.Nx), inv_c2=a.inv_c2)
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
        y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(a.real)))
    ipart = len(parts)
    t1 = min(a.T, t0 + max(SEG, SEG * (a.chunk // SEG)))
    stats, run0 = [], _sim.run

    def counted(*args, **kw):
        kw["t0"] = kw.get("t0", 0.0) + t0
        o = run0(*args, **kw)
        stats.append([o["num_steps"], o["num_rejected"], o["compile_time"], o["run_time"]])
        return o

    _sim.run = counted
    li = {}
    if a.field_nu > 0:
        _sim.rhs, li["integrating_factor"] = ds.field_closure(m, a.field_nu)
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, t1 - t0, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=a.max_steps,
                          noise_floor=a.noise_floor, **li)
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    nt = out["t"].size
    Ks = np.array([np.asarray(ds.energies(m, {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "B": out["B"][i].astype(complex)})["K_species"])
                   for i in range(nt)])
    sm = jax.jit(jax.vmap(lambda c, b: ds.speed_moments(m, c, b)))(jnp.asarray(out["Ck"]), jnp.asarray(out["B"], complex))
    n0, v1, v2 = (np.asarray(x) for x in sm)
    e1d = 0.5 * np.asarray(m.masses)[None] * (v2 - v1 ** 2 / n0)
    Ck = np.asarray(out["Ck"]).reshape(nt, m.Ns, m.Nn, m.Nx // 2 + 1)
    E2 = np.abs(Ck) ** 2
    herm_n = E2.sum(-1)
    top = E2[:, :, (2 * m.Nn) // 3:, :].sum(2) / np.maximum(E2.sum(2), 1e-300)
    fmin, fmax = f_extrema(m, out["Ck"], out["B"])
    Fx = np.abs(out["Fk"][:, 0, 0, :, 0]) ** 2
    Uk = Fx[:, 1:].sum(1)  # k != 0 electrostatic energy: (1/2) * 2 |E_k|^2 per retained mode, om0 = 1
    st = np.array(stats)
    t_end = float(out["t"][-1])
    status = out["status"] if (out["status"] != "success" or t_end >= t1 - 1e-9) else "short"
    rec = {"case": case, "part": ipart, "t_start": t0, "t_end": t_end, "status": status, "failure_reason": out["failure_reason"],
           "events": len(out["events"]), "steps": int(st[:, 0].sum()), "rejected": int(st[:, 1].sum()),
           "compile_time": float(st[:, 2].sum()), "run_time": float(st[:, 3].sum()), "wall_time": wall,
           "inv_c2": a.inv_c2, "Nx": a.Nx, "Nn": a.Nn, "field_nu": a.field_nu, "vq": a.vq, "real": a.real,
           "repository_commit": _sim._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "command": "python studies/relhhs_run.py " + " ".join(sys.argv[1:])}
    stem = out_dir / f"{case}.part{ipart}"
    np.savez_compressed(f"{stem}.npz", t=out["t"], K_species=Ks, W=out["W"], Ebar=np.real(out["Fk"][:, 0, 0, 0, 0]), Ek2=Fx, Uk=Uk, E1D=e1d,
                        B=out["B"], herm_n=herm_n, top_frac=top, fmin=fmin, fmax=fmax, seg_stats=st,
                        y_Ck=out["Ck"][-1], y_Fk=out["Fk"][-1], y_Dk=out["Dk"][-1], y_W=out["W"][-1].astype(complex),
                        y_B=out["B"][-1].astype(complex))
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
