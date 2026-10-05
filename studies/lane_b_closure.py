"""Lane B: order >= 2 hypercollision closure (SPECTRAX #60) as the declared strong-drive regularization.

Pump frame + adaptive remap exactly as studies/h05_pump_frame.py (parent c0910a1; same H05 inputs, seeds,
omega = sqrt(1 + 1/1836), SEG = 20, rtol 1e-10, 200k steps per segment). Only the closure changes:
nu in {0.25, 0.5, 1, 2, 4}, order in {2, 3} (spectrum from spectrax._initialization.hypercollision_spectrum,
installed into Model.p["collision_matrix"] before the first trace), Nn in {64, 128}, v_q/v_te in {0.01, 0.03, 0.1}.

Criteria (fixed before running, LANES.md): t_res = first t > 50 where electron dK or W_ext differs by > 10% of the
running max between (Nn, 2Nn) at fixed (nu, order), or between nu/2 and 2nu around nu at fixed (Nn, order).
Each observable reported separately. Positivity = first time K_species or k=0 T_x of a species is negative.
nu-plateau: an observable has a plateau at time t if all nu in the scan agree within 10% of the running max.

Run one configuration:  python studies/lane_b_closure.py VQ NN NU ORDER  -> studies/lane_b/h05_vq{VQ}_Nn{NN}_nu{NU}_o{ORDER}.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import darkspectrax as ds
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_c_run import BASE as seeds, SEG, diagnostics, model  # noqa: E402  (shared H05 setup)

OUT = Path(__file__).resolve().parent / "lane_b"
NSAVE = 11


if __name__ == "__main__":
    vq, Nn, nu, order = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    T = float(sys.argv[5]) if len(sys.argv) > 5 else 1000.0
    m = model(float(vq), Nn, nu=nu, order=order)
    y0 = ds.consistent_fields(m, ds.maxwellian(m, [1.0, 1.0], seeds))
    tic = time.perf_counter()
    out = ds.run_adaptive(m, y0, T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000,
                          stop_on_negative=False)  # the recorded sweep ran to the step budget
    wall = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    K, _, Tx, _ = diagnostics(m, out)
    neg = np.any(K <= 0, axis=1) | np.any(Tx <= 0, axis=1)
    scale = max(np.abs(out["W"][:, 2]).max(), 1e-300)
    meta = {"vq": vq, "Nn": Nn, "nu": nu, "order": order, "status": str(out["status"]),
            "failure_reason": str(out["failure_reason"]), "t_reached": float(out["t"][-1]), "wall": wall,
            "events": len(out["events"]), "max_ledger_over_W_ext": float(np.abs(out["ledger_defect"][:, good]).max() / scale),
            "t_first_negative": float(out["t"][np.argmax(neg)]) if neg.any() else None,
            **{k: out[k] for k in ("num_steps", "num_rejected", "compile_time", "run_time")},
            "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT}
    print(json.dumps(meta), flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"h05_vq{vq}_Nn{Nn}_nu{nu:g}_o{order}.npz", t=out["t"], dK=K - K[0], K=K, Tx=Tx,
                        Wext=out["W"][:, 2], B=out["B"], meta=json.dumps(meta))
