"""r1nl follow-up (declared after the r1nl stage-2 results, before its runs): what sets the small W_ext deficit?

OBSERVATION (stage 2, studies/r1nl/stage2_summary.json + stage-1 records): W_ext / W_lin - 1 of the coherent drive is the same at
v_q/v_te = 0.03, 0.05, 0.07 to three digits, D(t) = -1.0e-8 t^2 (D(100) = -1.0e-4, D(200) = -4.1e-4, D(400) = -1.66e-3 at all three),
until the OTSI stage (t > ~600). A drive nonlinearity would grow with v_q; a deficit independent of v_q must be set by something
else: the seeds (5e-4, 5e-5j), the Hermite truncation of the k = 0 response, or the closure.
HYPOTHESES (declared; the runs decide):
 H-seed:    the deficit is produced by the seeds through the quiver-induced coupling <dn dE> to the mean field, so D scales as the
            square of the seed amplitude.
 H-trunc:   the deficit is a property of the homogeneous k = 0 Hermite system (no seeds): D(400) of the unseeded control is the same.
Runs (all v_q/v_te = 0.07, Nx 32, Nn 64, c 64 Lawson, T = 400, otherwise as r1nl): seed scale s in {0 (unseeded control), 0.1, 1 (the stage-1 B run)}.
GATES:
 F1 (H-seed supported)  : D(400; s=0.1) / D(400; s=1) within [0.005, 0.02] (0.01 expected for amplitude^2) AND |D(400; s=0)| < 0.1 |D(400; s=0.1)|.
 F2 (H-trunc supported) : |D(400; s=0)| > 0.5 |D(400; s=1)|.
 Both can fail (then the origin is neither, and that is recorded; no gate is amended). D = W_ext / prescribed_linear - 1 on common saves.
 F3 ledger <= 1e-6 for each run.
Run: python studies/r1nl_followup.py summary  (after the three runs of studies/r1nl_followup.txt).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1nl_analysis as A  # noqa: E402

D = Path(__file__).resolve().parent / "r1nl"
V = 0.07


def summary():
    E0 = V * A.c1.vte
    cases = {"s0": f"pre_vq{V:g}_Nx32_Nn64_c64_seed0", "s0.1": f"pre_vq{V:g}_Nx32_Nn64_c64_seed0.1", "s1": A.name("pre", V, "B")}
    H = {k: A.load(c) for k, c in cases.items()}
    t = H["s1"]["t"]
    n = min(h["t"].size for h in H.values())
    t = t[:n]
    _, Wl = A.lin_fields(t, E0)
    out = {"cases": cases, "D": {}, "ledger": {k: h["ledger"] for k, h in H.items()}, "t_reached": {k: h["t_reached"] for k, h in H.items()}}
    for k, h in H.items():
        d = h["Wext"][:n] / np.where(Wl != 0, Wl, np.nan) - 1
        out["D"][k] = {str(tt): float(d[int(np.argmin(abs(t - tt)))]) for tt in (100, 200, 300, 400)}
    d0, d1, d10 = (out["D"][k]["400"] for k in ("s0", "s0.1", "s1"))
    out["F1"] = {"ratio_s0.1_over_s1": d1 / d10, "unseeded_over_s0.1": abs(d0) / abs(d1) if d1 else None,
                 "pass": bool(0.005 <= d1 / d10 <= 0.02 and abs(d0) < 0.1 * abs(d1))}
    out["F2"] = {"unseeded_over_s1": abs(d0) / abs(d10), "pass": bool(abs(d0) > 0.5 * abs(d10))}
    out["F3"] = {"pass": bool(max(out["ledger"].values()) <= 1e-6)}
    (D / "followup_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    summary()
