"""Lane B analysis of studies/lane_b/*.npz (criteria in studies/lane_b_closure.py docstring) -> studies/lane_b/summary.json"""

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "lane_b"
NUS, ORDERS, NNS, VQS = (0.25, 0.5, 1.0, 2.0, 4.0), (2, 3), (64, 128), ("0.01", "0.03", "0.1")


def load(vq, Nn, nu, o):
    f = D / f"h05_vq{vq}_Nn{Nn}_nu{nu:g}_o{o}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {"t": z["t"], "dK": z["dK"][:, 0], "W": z["Wext"], "meta": json.loads(str(z["meta"]))}


def first_bad(t, a, b, tol=0.1):
    """First t > 50 with |a-b| > tol * running max(|a|,|b|); None if they agree on the whole common window."""
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


def compare(r1, r2):
    n = min(r1["t"].size, r2["t"].size)
    t = r1["t"][:n]
    out = {"common_end": float(t[-1])}
    for obs in ("dK", "W"):
        b = first_bad(t, r1[obs][:n], r2[obs][:n])
        out[obs] = b if b is not None else float(t[-1])
        out[obs + "_censored"] = b is None  # agreement up to the common end (a run stopped or t = 1000)
    return out


S = {}
for vq in VQS:
    for o in ORDERS:
        for Nn in NNS:
            for nu in NUS:
                r = load(vq, Nn, nu, o)
                if r is None:
                    continue
                m = r["meta"]
                row = {"t_reached": m["t_reached"], "status": m["status"], "t_first_negative": m["t_first_negative"],
                       "wall": m["wall"], "compile": m["compile_time"], "steps": m["num_steps"],
                       "rejected": m["num_rejected"], "events": m["events"], "ledger_over_W": m["max_ledger_over_W_ext"]}
                rN = load(vq, 2 * Nn if Nn == 64 else Nn // 2, nu, o)
                if rN is not None:
                    row["Nn_pair"] = compare(r, rN)
                lo, hi = load(vq, Nn, nu / 2, o), load(vq, Nn, nu * 2, o)
                if lo is not None and hi is not None:
                    row["nu_pair"] = compare(lo, hi)
                if "Nn_pair" in row and "nu_pair" in row:
                    row["t_res"] = {obs: min(row["Nn_pair"][obs], row["nu_pair"][obs]) for obs in ("dK", "W")}
                S[f"vq{vq}_o{o}_Nn{Nn}_nu{nu:g}"] = row
            # nu plateau: all nu agree within 10% of running max (common window)
            rs = [load(vq, Nn, nu, o) for nu in NUS]
            if all(x is not None for x in rs):
                n = min(x["t"].size for x in rs)
                t = rs[0]["t"][:n]
                P = {"common_end": float(t[-1])}
                for obs in ("dK", "W"):
                    A = np.array([x[obs][:n] for x in rs])
                    scale = np.maximum(np.maximum.accumulate(np.abs(A).max(axis=0)), 1e-12)
                    bad = ((A.max(axis=0) - A.min(axis=0)) > 0.1 * scale) & (t > 50)
                    P[obs] = float(t[np.argmax(bad)]) if bad.any() else None
                    # adjacent plateau: the longest-agreeing neighbouring pair (nu, 2nu)
                    P[obs + "_adjacent"] = {f"{NUS[i]:g}-{NUS[i + 1]:g}": first_bad(t, A[i], A[i + 1]) for i in range(4)}
                    P[obs + "_final_spread_over_mean_at_t"] = {
                        str(tt): float((A[:, i].max() - A[:, i].min()) / max(np.abs(A[:, i]).mean(), 1e-300))
                        for tt in (200, 400, 600) for i in [int(np.argmin(np.abs(t - tt)))] if t[-1] >= tt}
                S[f"plateau_vq{vq}_o{o}_Nn{Nn}"] = P
(D / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
for k, v in S.items():
    print(k, json.dumps(v))
