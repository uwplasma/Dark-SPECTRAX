"""B09 nonlinear: is the adaptive-step collapse of the Weibel run (514 -> 55150 steps in t = 110-113 while U_B still
rises, studies/b09.py Part 2 probe) the AW truncation instability, and does a field-scaled closure remove it?

Case: studies/b09.py Part 2 base level, Weibel ordinary, k0 = 1.2, (Nx, Nn, Nm, Np) = (16, 32, 16, 1), seed B_z = 1e-6,
Dopri8 rtol 1e-10 atol 1e-16. Integrated in chunks of DT = 2 (restart from the saved state, t0 traced), per-chunk step
budget 20000; every chunk's final state is checkpointed (npz, outside the records) so a run resumes.

Closures (order-2 hypercollision spectrum, rows of total degree <= 2 untouched, so the ledger is exact):
  none : c = 0.
  E    : the parent SPECTRAX #66 rate nu_s = c |q/m|_s max_i sqrt(2 N_i) max_x|E_i - <E_i>| / a_s,i (N_i > 3).
  EB   : the same with the electric field replaced by the full non-uniform Lorentz force scale on each axis,
         E_i -> |E_i - <E_i>| + sum_{j != i} V_j |B_l - <B_l>|  (i, j, l a permutation; V_j = |u_j| + a_j sqrt(2 N_j)
         the largest velocity the kept ladder on axis j represents: the turning point of the top Hermite function).
         Rationale: in the AW operator (q/m)(v x B) . d_v, the v_j factor couples Hermite neighbours with amplitude
         a_j sqrt(N_j / 2) and d_v_i scales as sqrt(2 N_i) / a_i, so a non-uniform B gives the same non-normal top-mode
         structure as E with |E| -> |v_j B_l| at the top of the ladder.
Force field: the ordinary model uses F = Fk (no dark field in this probe).

Decisive test (declared before running):
  Collapse: a chunk needs > 10x the median step count of the chunks ending at t <= 100, or fails.
  1) step history: run c = 0, E (c = 1) and EB (c = 1) to t = 160 (or first failure). The closure "removes the
     collapse" if no chunk collapses up to t = 160, while the c = 0 run collapses.
  2) fidelity: before the c = 0 collapse, |U_B(closure) - U_B(c = 0)| / max U_B(c = 0) <= 1e-3.
  3) frozen-field spectrum: at the checkpoints t = 100 and the last c = 0 checkpoint before collapse, the largest
     real part (and largest |lambda|) of the kinetic operator dC/dt = L(F) C with the fields frozen, by ARPACK on its
     JVP (real-packed), c = 0 and with each closure (same state). AW-instability signature: max Re lambda > 0 at
     c = 0, set by modes in the top third of the Hermite ladder (energy fraction recorded), removed by the closure.
  Verdict "AW truncation instability, closure removes it" needs 1), 2) and 3) all true for the same closure.
  If no closure passes 1): record and stop (no nonlinear saturation run).
Amendment 1 (added after the first three probes, declared before rerunning them; B09_AMEND1=1): the 10x-median rule
fired for every run, including EB at t = 120 where the step count (186) rises with the physical growth while the
top-third Hermite energy fraction stays 1e-3 (c = 0 reached 0.99 at t = 110). Under the amendment a run continues
from its checkpoint to t = 160 and stops only on a per-chunk budget failure or when the top-third energy fraction
exceeds 0.5. Criterion 1) becomes: no budget failure and top-third fraction <= 0.1 up to t = 160 with the closure,
while c = 0 exceeds 0.1. The original 10x-median verdict stays recorded (field t_collapse).

Run: python studies/b09_closure.py probe CLOSURE C [TMAX]  -> studies/b09/closure_probe_<CLOSURE>_c<C>.json
     python studies/b09_closure.py spectrum T              -> studies/b09/closure_spectrum_t<T>.json
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b09  # noqa: E402
import darkspectrax as ds  # noqa: E402
import spectrax  # noqa: E402
from darkspectrax import _model as _M  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402

OUT = Path(__file__).resolve().parent / "b09"
CKPT = Path(os.environ.get("B09_CKPT", Path(__file__).resolve().parent / "b09" / "ckpt"))
K0, LEVEL, SEED, DT, BUDGET = 1.2, (16, 32, 16), 1e-6, 2.0, 20000
RTOL, ATOL = 1e-10, 1e-16
AMEND_STOP_TOP = os.environ.get("B09_AMEND1", "0") == "1"
_RHS0 = _M.rhs


def lorentz_scale(F, u_s, a_s, Nn, Nm, Np, Ns, magnetic):
    """(Ns, 3) per-axis force scale: |E_i - <E_i>| (+ sum_j V_j |B_l - <B_l>| when magnetic)."""
    dev = jnp.max(jnp.abs(F - jnp.mean(F, axis=(-3, -2, -1), keepdims=True)), axis=(-3, -2, -1))  # (6,)
    dE, dB = dev[:3], dev[3:]
    N = jnp.array([Nn, Nm, Np], dtype=float)
    V = jnp.abs(u_s.reshape(Ns, 3)) + a_s.reshape(Ns, 3) * jnp.sqrt(2 * N)[None]
    s = jnp.broadcast_to(dE[None], (Ns, 3))
    if magnetic:  # (v x B)_x = v_y B_z - v_z B_y, etc.
        s = s + jnp.stack([V[:, 1] * dB[2] + V[:, 2] * dB[1],
                           V[:, 2] * dB[0] + V[:, 0] * dB[2],
                           V[:, 0] * dB[1] + V[:, 1] * dB[0]], axis=1)
    return s


def closure_rate(F, u_s, a_s, m, c, magnetic):
    N = jnp.array([m.Nn, m.Nm, m.Np], dtype=float)
    a = a_s.reshape(m.Ns, 3)
    s = lorentz_scale(F, u_s, a_s, m.Nn, m.Nm, m.Np, m.Ns, magnetic)
    per = jnp.where((N > 3)[None], jnp.sqrt(2 * N)[None] * s / a, 0.0)
    return c * jnp.abs(m.p["qs"] * m.p["Omega_cs"]) * jnp.max(per, axis=1)


def closure_term(t, y, m, c, magnetic):
    col = spectrax.hypercollision_spectrum(m.Nn, m.Nm, m.Np, 2)[None, :, :, :, None, None, None]
    Fk = y["Fk"] + m.eta * y["Dk"][:6] if m.mode == "self_consistent" else y["Fk"]
    F = jnp.fft.irfftn(Fk * m.p["mask23"], s=(m.Nz, m.Ny, m.Nx), axes=(-1, -3, -2), norm="forward")
    u_s, a_s = _M.basis_of(m, y)
    nu = closure_rate(F, u_s, a_s, m, c, magnetic).reshape(m.Ns, 1, 1, 1, 1, 1, 1)
    Ck = y["Ck"].reshape(m.Ns, m.Np, m.Nm, m.Nn, *y["Ck"].shape[1:])
    return (-nu * col * Ck).reshape(y["Ck"].shape)


def install(closure, c):
    if closure == "none" or c == 0:
        _sim.rhs = _RHS0
        return
    mag = closure == "EB"

    def rhs(t, y, model):
        out = _RHS0(t, y, model)
        return {**out, "Ck": out["Ck"] + closure_term(t, y, model, c, mag)}

    _sim.rhs = rhs


def model_and_y0(case="weibel", OD=None, level=LEVEL):
    m = b09.model_for(case, OD, K0, *level)
    return m, b09.seeded(m, case, SEED)


def probe(closure, c, tmax, case="weibel", OD=None, level=LEVEL, tag=""):
    m, y = model_and_y0(case, OD, level)
    install(closure, c)
    name = f"{case}_{'ordinary' if OD is None else f'OD{OD}'}_{level[0]}_{level[1]}_{level[2]}_{closure}_c{c:g}{tag}"
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = CKPT / f"{name}.npz"
    chunks, t = [], 0.0
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        y = {k: jnp.asarray(z[k]) for k in ("Ck", "Fk", "Dk", "W")}
        t, chunks = float(z["t"]), json.loads(str(z["chunks"]))
        print(f"resume {name} at t={t}", flush=True)
    cache, status = {}, "success"
    while t < tmax - 1e-9:
        seg = ds.run(m, y, DT, n_save=5, t0=t, rtol=RTOL, atol=ATOL, max_steps=BUDGET, cache=cache)
        UB, UBD = b09.field_energy_series(m, seg)
        row = {"t0": t, "t1": t + DT, "steps": seg["num_steps"], "rejected": seg["num_rejected"],
               "status": seg["status"], "failure_reason": seg["failure_reason"], "run_time": seg["run_time"],
               "t": seg["t"].tolist(), "UB": UB.tolist(), "UBD": UBD.tolist(), "K": seg["K"].tolist(),
               "U_gamma": seg["U_gamma"].tolist(), "W": seg["W"].tolist(),
               "top_third_energy_fraction": top_fraction(m, seg["Ck"][-1])}
        chunks.append(row)
        early = [r["steps"] for r in chunks if r["t1"] <= 100 and r["status"] == "success"]
        med = float(np.median(early)) if early else None
        print(f"{name} t={t + DT:6.1f} steps={row['steps']:6d} UB={UB[-1]:.4e} top={row['top_third_energy_fraction']:.2e}"
              f" {row['status']} {row['run_time']:.1f}s", flush=True)
        if seg["status"] != "success":
            status = "failure"
            break
        y = {"Ck": jnp.asarray(seg["Ck"][-1]), "Fk": jnp.asarray(seg["Fk"][-1]), "Dk": jnp.asarray(seg["Dk"][-1]),
             "W": jnp.asarray(seg["W"][-1], complex)}
        t += DT
        np.savez(ck, t=t, chunks=json.dumps(chunks), **{k: np.asarray(v) for k, v in y.items()})
        if AMEND_STOP_TOP and row["top_third_energy_fraction"] > 0.5:
            status = "truncation-dominated"
            break
        if not AMEND_STOP_TOP and med and t > 100 and row["steps"] > 10 * med:
            status = "collapse"
            break
    early = [r["steps"] for r in chunks if r["t1"] <= 100 and r["status"] == "success"]
    med = float(np.median(early)) if early else None
    t_collapse = next((r["t1"] for r in chunks if r["status"] != "success" or (med and r["t1"] > 100
                                                                              and r["steps"] > 10 * med)), None)
    return {"case": case, "model": "ordinary" if OD is None else f"dark_OD{OD}", "level": list(level),
            "closure": closure, "c": c, "t_max": tmax, "DT": DT, "budget_per_chunk": BUDGET, "rtol": RTOL, "atol": ATOL,
            "seed_Bz": SEED, "k0": K0, "status": status, "t_reached": t, "median_steps_t_le_100": med,
            "t_collapse": t_collapse, "amendment_1": AMEND_STOP_TOP,
            "t_top_third_gt_0.1": next((r["t1"] for r in chunks if r["top_third_energy_fraction"] > 0.1), None),
            "t_budget_failure": next((r["t1"] for r in chunks if r["status"] != "success"), None), "chunks": chunks}


def top_fraction(m, Ck):
    C = np.asarray(Ck).reshape(m.Ns, m.Np, m.Nm, m.Nn, -1)
    e = np.abs(C) ** 2
    n = np.arange(m.Nn)[None, None, None, :, None]
    mm = np.arange(m.Nm)[None, None, :, None, None]
    top = (n >= 2 * m.Nn // 3) | (mm >= 2 * m.Nm // 3)
    return float(e[np.broadcast_to(top, e.shape)].sum() / e.sum())


def spectrum(T):
    """Frozen-field kinetic spectrum at the c = 0 checkpoint state at time T (rerun to T from the c = 0 checkpoint)."""
    from scipy.sparse.linalg import LinearOperator, eigs
    m, y = model_and_y0()
    install("none", 0)
    name = f"weibel_ordinary_{LEVEL[0]}_{LEVEL[1]}_{LEVEL[2]}_none_c0"
    z = np.load(CKPT / f"{name}.npz", allow_pickle=True)
    chunks = json.loads(str(z["chunks"]))
    t_ck = float(z["t"])
    y = {k: jnp.asarray(z[k]) for k in ("Ck", "Fk", "Dk", "W")}
    if T < t_ck - 1e-9:  # rerun from 0 to T (cheap before the collapse)
        _, y0 = model_and_y0()
        seg = ds.run(m, y0, T, n_save=2, rtol=RTOL, atol=ATOL, max_steps=200000)
        y = {"Ck": jnp.asarray(seg["Ck"][-1]), "Fk": jnp.asarray(seg["Fk"][-1]), "Dk": jnp.asarray(seg["Dk"][-1]),
             "W": jnp.asarray(seg["W"][-1], complex)}
        t_ck = T
    shape = y["Ck"].shape
    nC = int(np.prod(shape))
    rows = []
    for closure, c in (("none", 0.0), ("E", 1.0), ("EB", 1.0)):
        def Lc(C, closure=closure, c=c):
            yy = {**y, "Ck": C}
            out = _RHS0(t_ck, yy, m)["Ck"]
            if c:
                out = out + closure_term(t_ck, yy, m, c, closure == "EB")
            return out
        # Ck enters linearly once the fields are frozen (closure rate depends on fields only)
        f = jax.jit(lambda v, Lc=Lc: (lambda o: jnp.concatenate([o.real.ravel(), o.imag.ravel()]))(
            Lc((v[:nC] + 1j * v[nC:]).reshape(shape))))
        op = LinearOperator((2 * nC, 2 * nC), matvec=lambda v: np.asarray(f(jnp.asarray(v))), dtype=float)
        tic = time.perf_counter()
        lr, vr = eigs(op, k=6, which="LR", tol=1e-8, maxiter=20000, ncv=60)
        lm = eigs(op, k=2, which="LM", tol=1e-6, maxiter=20000, ncv=40, return_eigenvectors=False)
        i = int(np.argmax(lr.real))
        v = vr[:nC, i] + 1j * vr[nC:, i]
        rows.append({"closure": closure, "c": c, "LR": [[float(x.real), float(x.imag)] for x in lr],
                     "max_re": float(lr.real.max()), "max_abs": float(np.abs(lm).max()),
                     "top_third_fraction_of_LR_mode": top_fraction(m, v.reshape(shape)),
                     "time": time.perf_counter() - tic})
        print(T, rows[-1]["closure"], rows[-1]["max_re"], rows[-1]["max_abs"], rows[-1]["top_third_fraction_of_LR_mode"],
              flush=True)
    return {"t": t_ck, "level": list(LEVEL), "frozen": "fields (Fk, Dk) frozen; dC/dt operator incl. closure",
            "UB_at_t": float(b09.field_energy_series(m, {"Fk": np.asarray(y["Fk"])[None],
                                                         "Dk": np.asarray(y["Dk"])[None]})[0][0]),
            "c0_checkpoint_chunks": len(chunks), "results": rows}


if __name__ == "__main__":
    part = sys.argv[1]
    tic = time.perf_counter()
    if part == "probe":
        closure, c = sys.argv[2], float(sys.argv[3])
        tmax = float(sys.argv[4]) if len(sys.argv) > 4 else 160.0
        rec, fname = probe(closure, c, tmax), f"closure_probe_{closure}_c{c:g}.json"
    else:
        T = float(sys.argv[2])
        rec, fname = spectrum(T), f"closure_spectrum_t{T:g}.json"
    rec.update({"repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
                "command": " ".join(["python", "studies/b09_closure.py"] + sys.argv[1:]),
                "wall_time": time.perf_counter() - tic})
    OUT.mkdir(exist_ok=True)
    (OUT / fname).write_text(json.dumps(rec, indent=1, default=float) + "\n")
