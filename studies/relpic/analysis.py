"""relpic analysis: evaluates the gates declared in gates.py (thresholds are read from there, never edited here).

    DJIC=... JIC=... python studies/relpic/analysis.py      -> studies/relpic/summary.json, figure.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gates as G  # noqa: E402

REC = HERE / "records"
WIN = 200.0


def load(tag):
    f = REC / tag / "history.npz"
    if not f.exists():
        return None
    h = dict(np.load(f))
    h["record"] = json.loads((REC / tag / "record.json").read_text())
    return h


def trailing(t, y, at):
    """Mean of y over (at - 200, at]."""
    m = (t > at - WIN) & (t <= at + 1e-9)
    return float(np.mean(y[m])) if m.any() else float("nan")


def dW(h, at):
    return trailing(h["t"], h["spread"][:, 0] - h["spread"][0, 0], at)


def series(tags, fn, at):
    runs = [load(t) for t in tags]
    return np.array([fn(r, at) for r in runs if r is not None])


def hermite():
    """Newtonian Hermite record (Dark-SPECTRAX nxconv, v_q/v_te 0.03, Nx 32, Nn 64, c 4): 1D electron thermal energy Tx/2."""
    t, tx = [], []
    for f in sorted((ROOT / "studies/nxconv/runs").glob("vq0.03_Nx32_Nn64_c4_nf1e-14.part*.npz"),
                    key=lambda p: int(p.stem.split("part")[-1])):
        d = np.load(f)
        t.append(d["t"]); tx.append(d["Tx"][:, 0] / 2)
    t, tx = np.concatenate(t), np.concatenate(tx)
    _, i = np.unique(t, return_index=True)
    return t[i], tx[i]


def main():
    out = dict(gates_declared_in="studies/relpic/gates.py")
    tags = lambda p, n: [f"{p}_s{i}" for i in range(n)]
    R, N, Q = tags("R", 3), tags("N", 3), tags("Q", 2)
    # --- single-particle ---------------------------------------------------------------------------------
    sp = json.loads((HERE / "single_particle.json").read_text()) if (HERE / "single_particle.json").exists() else None
    out["single_particle"] = sp
    # --- per-run ledgers (G1, G2) ----------------------------------------------------------------------------
    ledger = {}
    for tag in R + N + Q + ["Z", "D", "P4", "X2"]:
        h = load(tag)
        if h is None:
            continue
        r = h["record"]["results"]
        ledger[tag] = dict(
            T=float(h["t"][-1]),
            energy_defect=r["max_energy_work_defect_over_initial_thermal"],
            gauss=r["max_ordinary_gauss_over_en_eps0"], continuity=r["max_continuity_over_enwp"],
            momentum=r["max_momentum_defect_over_nmecL"], max_speed=float(h["max_speed"].max()),
            wall_s=h["record"]["wall_s"], ok_G1=r["max_energy_work_defect_over_initial_thermal"] <= G.GATES["G1_energy_defect"],
            ok_G2=(r["max_ordinary_gauss_over_en_eps0"] <= G.GATES["G2_gauss"]
                   and r["max_continuity_over_enwp"] <= G.GATES["G2_continuity"]
                   and abs(r["max_momentum_defect_over_nmecL"]) <= G.GATES["G2_momentum"]))
    out["ledger"] = ledger
    # --- G3 early response vs the homogeneous kinetic oracle (t <= 40) ---------------------------------------
    sys.path.insert(0, os.environ["DJIC"]); sys.path.insert(0, os.environ["JIC"])
    import jax
    jax.config.update("jax_enable_x64", True)
    import importlib
    m = importlib.import_module("examples.dark_reservoir")
    g3 = {}
    for tag in R:
        h = load(tag)
        if h is None:
            continue
        t = h["t"]; sel = t <= 40
        oracle = m.homogeneous(t, 0.03 * np.sqrt(1e-3), eta=None)   # parent's kinetic homogeneous orbit oracle
        ref = np.asarray(oracle["mean_E"])[sel]
        g3[tag] = float(np.max(np.abs(h["mean_E"][sel] - ref)) / np.max(np.abs(ref)))
    out["G3_early"] = dict(values=g3, gate=G.GATES["G3_early"],
                           ok=all(v is not None and v <= G.GATES["G3_early"] for v in g3.values()) if g3 else None)
    # --- the observables --------------------------------------------------------------------------------------
    times = np.arange(200.0, 2001.0, 200.0)
    obs = {}
    for name, tg in (("R", R), ("N", N), ("Q", Q), ("P4", ["P4"]), ("X2", ["X2"]), ("D", ["D"]), ("Z", ["Z"])):
        runs = [load(t) for t in tg]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        row = {}
        for key, fn in (("dW_th", lambda h, at: dW(h, at)),
                        ("W_E0", lambda h, at: trailing(h["t"], h["mean_E"] ** 2 / 2, at)),
                        ("W_Ek", lambda h, at: trailing(h["t"], h["nonzero_electric"], at)),
                        ("rms_e", lambda h, at: trailing(h["t"], h["rms"][:, 0], at))):
            vals = []
            for at in times:
                v = np.array([fn(r, at) for r in runs if r["t"][-1] >= at - 1e-9])
                vals.append((float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None, len(v)) if len(v) else None)
            row[key] = vals
        obs[name] = row
    out["times"] = times.tolist()
    out["observables"] = obs
    # --- G4-G7 refinements ------------------------------------------------------------------------------------
    g = G.GATES
    Rr = [load(t) for t in R]; Rr = [r for r in Rr if r is not None]
    ref = {}
    if len(Rr) >= 2:
        for name, tg in (("G4_ppc412", "P4"), ("G5_dx2000", "X2"), ("G6_dt0.01", "D")) + tuple(
                ("G7_quintic_s%d" % i, "Q_s%d" % i) for i in range(2)):
            h = load(tg)
            if h is None:
                continue
            rows = []
            for at in g["G456_times"]:
                mr = np.array([dW(r, at) for r in Rr])
                mean, std = mr.mean(), mr.std(ddof=1)
                v = dW(h, at)
                tol = max(g["G456_rel"] * mean, g["G456_sigma"] * std)
                rows.append(dict(t=at, refined=v, mean_R=float(mean), std_R=float(std), diff=float(v - mean),
                                 tol=float(tol), ok=bool(abs(v - mean) <= tol)))
            ref[name] = dict(rows=rows, ok=all(r["ok"] for r in rows))
    out["refinement"] = ref
    # --- G8 relativistic detuning ------------------------------------------------------------------------------
    Nr = [load(t) for t in N]; Nr = [r for r in Nr if r is not None]
    g8 = []
    if len(Rr) >= 2 and len(Nr) >= 2:
        for at in g["G8_times"]:
            a = np.array([dW(r, at) for r in Rr if r["t"][-1] >= at - 1e-9])
            b_ = np.array([dW(r, at) for r in Nr if r["t"][-1] >= at - 1e-9])
            if len(a) < 2 or len(b_) < 2:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))
            g8.append(dict(t=at, R=float(a.mean()), N=float(b_.mean()), diff=float(a.mean() - b_.mean()),
                           ratio_R_over_N=float(a.mean() / b_.mean()), stderr=float(se),
                           sigmas=float(abs(a.mean() - b_.mean()) / se), resolved=bool(abs(a.mean() - b_.mean()) > g["G8_sigma"] * se)))
    out["G8_relativistic_vs_newtonian"] = g8
    # --- G9 noise floor -----------------------------------------------------------------------------------------
    z = load("Z")
    if z is not None and Rr:
        zv, rv = dW(z, 1000.0), float(np.mean([dW(r, 1000.0) for r in Rr]))
        out["G9_noise_floor"] = dict(no_drive=zv, driven=rv, ratio=zv / rv, ok=bool(zv <= g["G9_noise_fraction"] * rv))
    # --- G10 Dark-SPECTRAX Hermite (Newtonian) vs Newtonian PIC -------------------------------------------------------
    th, tx = hermite()
    g10 = []
    if len(Nr) >= 2:
        for at in (400.0, 600.0, 800.0, 1000.0):
            hv = trailing(th, tx - tx[0], at)
            pv = float(np.mean([dW(r, at) for r in Nr]))
            g10.append(dict(t=at, hermite=hv, pic_N=pv, rel_diff=float((pv - hv) / hv),
                            ok=bool(abs(pv - hv) <= g["G10_rel"] * abs(hv))))
    out["G10_hermite_vs_newtonian_pic"] = dict(rows=g10, ok=all(r["ok"] for r in g10) if g10 else None)
    # also the relativistic PIC against the Hermite record (the quantity the nonrelativistic model lacks)
    out["hermite_dW_th"] = {str(int(at)): trailing(th, tx - tx[0], at) for at in times if at <= 1000}
    # --- HHS Fig. 2 (digitized curves, approximate) ---------------------------------------------------------------
    f = Path(os.environ["DJIC"]) / "docs/_static/figures/paper_replay/data.npz"
    if f.exists():
        d = np.load(f)
        ts, rms = d["strong_electron_rms_t"], d["strong_electron_rms"]
        rows = []
        for at in times:
            sel = (ts > at - WIN) & (ts <= at)
            if sel.sum() < 2 or not Rr:
                continue
            hhs = float(np.mean(rms[sel] ** 2 / 2) - 5e-4)
            mine = float(np.mean([dW(r, at) for r in Rr if r["t"][-1] >= at - 1e-9]))
            rows.append(dict(t=float(at), hhs_digitized_dW_th=hhs, relpic_R=mine, ratio=mine / hhs if hhs else None))
        out["HHS_fig2_digitized"] = rows
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("observables",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
