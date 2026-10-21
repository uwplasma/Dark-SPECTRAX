"""R2 follow-up F, declared after the main analysis (studies/r2/summary.json) and BEFORE its runs.

Observation to explain: at nu_ei = 1e-3 the detuned cases (omega = 0.9, 1.1) fail K2 (R_late = 1.09-1.14, the same at v_q/v_te = 0.01, 0.03, 0.05)
while the unseeded homogeneous controls reproduce the linear law to 3e-8.  Hypothesis H: the departure is the back-reaction of the imposed finite-k
seed modulation (as for the 0.7% deficit of r1nl, which scaled as the seed amplitude squared), not a drive nonlinearity and not a velocity-space
effect of the collisional operator.
Runs (EM, Nx 32, Nn 64, c 64, T = 1000, chunk 200, v_q/v_te = 0.03, nu_ei = 1e-3): omega = 0.9 and omega = res, seeds multiplied by s = 0.1
(`--seed-scale 0.1`); the s = 1 runs are the main-study B members.  D_s = R_cum,s - 1 and E_s = R_late,s - 1 as defined in r2_gates.py (same tb and Delta
as the main analysis of that case).
 F1  for each of the two cases: D_0.1 / D_1 in [0.005, 0.02] and E_0.1 / E_1 in [0.005, 0.02] (scaling s^2 = 0.01 within a factor 2); PASS iff both hold in both cases.
 F2  ledger of the two runs <= 1e-6 (L1).
Hypothesis H is supported only if F1 holds; if F1 fails, the departure is recorded as not explained by seed amplitude.
Run: python studies/r2_followup.py run | analyze  -> studies/r2/followup_summary.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_analysis as an  # noqa: E402
import r2_run  # noqa: E402

CASES = (("0.9", 0.03, 1e-3), ("res", 0.03, 1e-3))
S = 0.1


def run():
    for tag, vq, nu in CASES:
        for _ in range(8):
            subprocess.run([sys.executable, str(HERE / "r2_run.py"), "--conv", "EM", "--vq", str(vq), "--nu", str(nu), "--omega", tag,
                            "--Nx", "32", "--Nn", "64", "--c", "64", "--T", "1000", "--chunk", "200", "--seed-scale", str(S)], check=True)
            h = an.load(r2_run.case_name("EM", vq, nu, tag, 32, 64, False, S))
            if h["t"][-1] >= 1000 - 1e-9:
                break


def analyze():
    summ = json.loads((HERE / "r2" / "summary.json").read_text())
    out, ok1, ok2 = {}, [], []
    for tag, vq, nu in CASES:
        key = f"vq{vq:g}_nu{nu:g}_w{tag}"
        main = summ["cases"][key]
        tb = main["tb"]
        h = an.load(r2_run.case_name("EM", vq, nu, tag, 32, 64, False, S))
        rr, _ = an.rates(h, vq, nu, tag, tb)
        D1, E1 = main["B"]["R_cum"] - 1, main["B"]["R_late"] - 1
        D, E = rr["R_cum"] - 1, rr["R_late"] - 1
        out[key] = {"tb": tb, "t_end": float(h["t"][-1]), "D_s1": D1, "E_s1": E1, "D_s0.1": D, "E_s0.1": E,
                    "D_ratio": D / D1, "E_ratio": E / E1, "ledger": an.ledger_ratio(h)}
        ok1.append(0.005 <= D / D1 <= 0.02 and 0.005 <= E / E1 <= 0.02 and h["t"][-1] >= tb)
        ok2.append(an.ledger_ratio(h) <= 1e-6)
    out["gates"] = {"F1": bool(all(ok1)), "F2": bool(all(ok2))}
    (HERE / "r2" / "followup_summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["gates"]))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
