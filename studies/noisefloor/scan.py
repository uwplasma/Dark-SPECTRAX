"""noise_floor scan: choose the control-run default with evidence.

(a) unseeded pump-frame H05-type controls (lane_c_run.model, real = -1), v_q/v_te 0.01 and 0.1, Nn64, T = 1000:
    max |W_ext / W_lin - 1| for t > 20 vs the exact uniform two-fluid oscillator, steps, wall.
(b) the H00 / H01 records of studies/hhs (hhs_ladder.py inputs) rerun with the floor vs their committed values.
(c) one seeded case (v_q/v_te 0.03, Nn64, r0) vs the committed Lane C record and the r0/r1/r2 spread.
Floor 'none' = plain PID (for (a) the committed Lane C records are used instead: they fail at the step budget).
Usage: python studies/noisefloor/scan.py PART FLOOR...   (PART in a, a:RTOL, b, c)  -> studies/noisefloor/scan_<PART>.jsonl
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hhs_ladder as H  # noqa: E402
from lane_c_run import NSAVE, SEG, diagnostics, model, seeds  # noqa: E402

part, floors = sys.argv[1].split(":")[0], [None if f == "none" else float(f) for f in sys.argv[2:]]
PLAIN = {0.01: 158.0, 0.1: 31.5}  # t reached by the committed plain-PID controls
RTOL = float(sys.argv[1].split(":")[1]) if ":" in sys.argv[1] else 1e-10  # PART a:RTOL overrides rtol
out_file = HERE / f"scan_{sys.argv[1].replace(':', '_rtol')}.jsonl"


def W_lin(t, vq):
    eps, E0, w = 1 / 1836, vq * np.sqrt(1e-3), np.sqrt(1 + 1 / 1836)
    A, mu = E0 * (1 + eps), 1 / (1 + eps)
    r, rd = -(A / (2 * w)) * t * np.sin(w * t), -(A / (2 * w)) * (np.sin(w * t) + w * t * np.cos(w * t))
    return 0.5 * mu * (rd ** 2 + w ** 2 * r ** 2)


def emit(rec):
    print(json.dumps(rec), flush=True)
    with out_file.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def pump(vq, Nn, real, nf, rtol=1e-10):
    m = model(vq, Nn)
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds(real)))
    tic = time.perf_counter()
    o = ds.run_adaptive(m, y0, 1000.0, SEG, n_save_segment=NSAVE, rtol=rtol, atol=1e-14, max_steps=200_000,
                        noise_floor=nf)
    wall = time.perf_counter() - tic
    good = np.isfinite(o["t"]) & np.all(np.isfinite(o["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        o[k] = o[k][good]
    return m, o, wall


for nf in floors:
    if part == "a":
        for vq in (0.01, 0.1):
            _, o, wall = pump(vq, 64, -1, nf, RTOL)
            t, W = o["t"], o["W"][:, 2]
            sel = t > 20
            emit({"part": "a", "rtol": RTOL, "floor": nf, "vq": vq, "status": o["status"], "failure_reason": o["failure_reason"],
                  "t_reached": float(t[-1]), "steps": o["num_steps"], "rejected": o["num_rejected"], "wall": wall,
                  "max_rel_dev_W_lin": float(np.abs(W[sel] / W_lin(t[sel], vq) - 1).max()),
                  "t_at_max_dev": float(t[sel][np.argmax(np.abs(W[sel] / W_lin(t[sel], vq) - 1))]),
                  # same window as the committed plain-PID control (which stopped at the step budget)
                  "max_rel_dev_W_lin_plainPID_window": float(np.abs(W[sel & (t <= PLAIN[vq])] /
                                                                    W_lin(t[sel & (t <= PLAIN[vq])], vq) - 1).max())})
    elif part == "b":
        old = json.loads((HERE.parent / "hhs" / "run.json").read_text())
        for key, r in H.h00(nf).items():
            o = old["H00"][key]
            emit({"part": "b", "floor": nf, "case": f"H00_{key}", "status": r["status"], "steps": r["steps"],
                  "Ebar_err": r["max_abs_Ebar_err_over_max"], "Ebar_err_committed": o["max_abs_Ebar_err_over_max"],
                  "W_rel_err": r["W_ext_rel_err"], "W_rel_err_committed": o["W_ext_rel_err"],
                  "steps_committed": o["steps"]})
        for key, r in H.h01(nf).items():
            o = old["H01"][key]
            emit({"part": "b", "floor": nf, "case": f"H01_{key}", "status": r["status"], "steps": r["steps"],
                  "Ebar_err": r["max_abs_Ebar_err_over_max"], "Ebar_err_committed": o["max_abs_Ebar_err_over_max"],
                  "identity": r["max_identity_residual_over_wL2E0"],
                  "identity_committed": o["max_identity_residual_over_wL2E0"], "steps_committed": o["steps"]})
    elif part == "c":
        case = "vq0.03_Nn64_r0"
        oz = np.load(HERE.parent / "lane_c" / "runs" / f"{case}.npz")
        old = json.loads((HERE.parent / "lane_c" / "runs" / f"{case}.json").read_text())
        m, o, wall = pump(0.03, 64, 0, nf)
        K = diagnostics(m, o)[0]
        n = min(o["t"].size, oz["t"].size)
        dKn, dKo = K[:n, 0] - K[0, 0], oz["K"][:n, 0] - oz["K"][0, 0]
        win = o["t"][:n] > 50
        run_max = np.maximum.accumulate(np.abs(dKo))
        emit({"part": "c", "floor": nf, "case": case, "status": o["status"], "failure_reason": o["failure_reason"],
              "t_reached": float(o["t"][-1]), "t_reached_committed": old["t_reached"], "steps": o["num_steps"],
              "wall": wall, "common_saves": int(n),
              "max_rel_diff_dK_e": float(np.max(np.abs(dKn - dKo)[win] / run_max[win])),
              "max_rel_diff_W_ext": float(np.max(np.abs(o["W"][:n, 2] - oz["W"][:n, 2]) / np.abs(oz["W"][:n, 2]).max()))})
