"""highk: LANES.md contract tables for the field-scaled closure runs (studies/highk/runs/*fnu*), plus baselines
(lane C, no closure) and the mobile-ion grid (studies/lane_c/gridm_*).
t_res(obs, pair) = first t > 50 where |a - b| > 10% of the running max of |a| (obs = dK_e, W_ext); pairs: closure
strength c halved (0.5 vs 1) and doubled (2 vs 1) at Nn 64; Nn 64 vs 128 at c = 1. t_pos = first save with
K_s <= 0 or k=0 T_x <= 0. Grid agreement: same t_res rule, Hermite (c=1, Nn64) vs grid at the same Nx.
Run: python studies/highk_analysis.py -> studies/highk/summary.json
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
R, G = D / "highk" / "runs", D / "lane_c"


def herm(case):
    p = R / f"{case}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "K_species" not in d:  # early record without per-species diagnostics
        return None
    t = d["t"]
    K = d["K_species"]
    return {"t": t, "dKe": K[:, 0] - K[0, 0], "W": d["W"][:, 2], "Tx": d["Tx"],
            "K": K, "Ek2": d["Ek2"], "rec": json.loads((R / f"{case}.json").read_text())}


def grid(vq, Nx):
    p = G / f"gridm_vq{vq:g}_Nx{Nx}_Nv4096_dt0.02.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"t": d["t"], "dKe": d["dK_e"], "W": d["W_ext"], "Ek2": d["Ek2"], "fmin": d["fmin_rel"]}


def t_res(a, b, key):
    t = np.intersect1d(np.round(a["t"], 6), np.round(b["t"], 6))
    if t.size == 0:
        return None
    ia = np.searchsorted(np.round(a["t"], 6), t)
    ib = np.searchsorted(np.round(b["t"], 6), t)
    x, y = a[key][ia], b[key][ib]
    scale = np.maximum.accumulate(np.abs(x))
    bad = np.nonzero((t > 50) & (np.abs(x - y) > 0.1 * scale))[0]
    return float(t[bad[0]]) if bad.size else f">={t[-1]:g}"


def t_pos(h):
    bad = np.nonzero((h["K"] <= 0).any(1) | (h["Tx"] <= 0).any(1))[0]
    return float(h["t"][bad[0]]) if bad.size else None


def first(vq, Nx):
    """c = 1, Nn 64 record: plain PID (Nx 8, 16) or with the noise floor (Nx 32, after the rebase on main)."""
    for tag in ("_v2", "", "_nf1e-14"):
        h = herm(f"vq{vq:g}_Nx{Nx}_Nn64_fnu1{tag}")
        if h is not None:
            return h
    return None


out = {}
for vq, T in ((0.1, 700), (0.03, 1000)):
    for Nx in (8, 16, 32):
        base = first(vq, Nx)
        if base is None:
            continue
        row = {"t_reached": base["rec"]["t_reached"], "status": base["rec"]["status"], "t_pos": t_pos(base),
               "steps": base["rec"]["steps"], "rejected": base["rec"]["rejected"],
               "compile_s": round(base["rec"]["compile_time"], 1), "run_s": round(base["rec"]["run_time"], 1)}
        for tag, case in (("c0.5", f"vq{vq:g}_Nx{Nx}_Nn64_fnu0.5"), ("c2", f"vq{vq:g}_Nx{Nx}_Nn64_fnu2"),
                          ("Nn128", f"vq{vq:g}_Nx{Nx}_Nn128_fnu1")):
            o = herm(case)
            if o is not None:
                row[tag] = {"t_res_dKe": t_res(base, o, "dKe"), "t_res_W": t_res(base, o, "W"),
                            "t_reached": o["rec"]["t_reached"], "t_pos": t_pos(o)}
        g = grid(vq, Nx)
        if g is not None:
            row["grid"] = {"t_res_dKe": t_res(base, g, "dKe"), "t_res_W": t_res(base, g, "W"),
                           "grid_fmin_lt_-1e-3": float(g["t"][np.argmax(g["fmin"] < -1e-3)]) if (g["fmin"] < -1e-3).any() else None}
        out[f"vq{vq:g}_Nx{Nx}"] = row
    # x-convergence of the closure runs: Nx 8 vs 16, 16 vs 32
    hs = {Nx: first(vq, Nx) for Nx in (8, 16, 32)}
    a, b = herm(f"vq{vq:g}_Nx16_Nn64_fnu1_v2"), herm(f"vq{vq:g}_Nx16_Nn64_fnu1_nf1e-14")
    if a is not None and b is not None:
        out[f"vq{vq:g}_Nx16_noise_floor_check"] = {"t_res_dKe": t_res(a, b, "dKe"), "t_res_W": t_res(a, b, "W")}
    for A, B in ((8, 16), (16, 32)):
        if hs[A] is not None and hs[B] is not None:
            out[f"vq{vq:g}_Nx{A}_vs_Nx{B}"] = {"t_res_dKe": t_res(hs[A], hs[B], "dKe"), "t_res_W": t_res(hs[A], hs[B], "W")}
    gs = {Nx: grid(vq, Nx) for Nx in (8, 16, 32)}
    for A, B in ((8, 16), (16, 32)):
        if gs[A] is not None and gs[B] is not None:
            out[f"grid_vq{vq:g}_Nx{A}_vs_Nx{B}"] = {"t_res_dKe": t_res(gs[A], gs[B], "dKe"), "t_res_W": t_res(gs[A], gs[B], "W")}
(D / "highk" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
