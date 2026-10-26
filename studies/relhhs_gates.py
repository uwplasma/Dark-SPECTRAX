"""relhhs gates, DECLARED BEFORE ANY PRODUCTION RUN (only a 40-omega_pe^-1 smoke run of relhhs_run.py, Nx 32, Nn 64, was done to measure cost and check
that the records are finite: ledger 1e-14, f_e min/max -6e-42; nothing was gated or tuned). Thresholds are not amended after results are seen; a failed gate
is recorded as a failure and any follow-up is declared in a new commit before it is run.

Question: does the relativistic momentum-space Hermite model (SPECTRAX #88 + pump-frame F0 support, Dark-SPECTRAX inv_c2 = 1: velocities in c, T_e = T_i =
1e-3 m_e c^2, i.e. u_te = v_te/c = sqrt(1e-3): the (v_te/c)^-2 scaling is carried by alpha_s = sqrt(2) sqrt(1e-3) with inv_c2 = 1/c^2 = 1 in c = 1 units) reproduce the relativistic collapse of the HHS resonance
(relgrid: mean-field energy 10% off the linear law at t ~ 367-392) and the relativistic electron thermal gain (relpic PIC 21.5 E_th at t = 1000, relgrid
invalid beyond t ~ 350), and how far in t is it certified?

Set-up (all production runs): v_q/v_te = 0.03, m_i/m_e = 1836, T_i = T_e, L = 40 c/omega_pe, omega = sqrt(1 + 1/1836), deterministic nxconv seeds (lane_c_run
r0), pump frame with remap check every 20, relativistic RHS, field-scaled closure of strength c, exact in the Lawson factor (streaming explicit), rtol 1e-10,
atol 1e-14 with noise floor 1e-14, T = 1000, saves every 0.5. Full factorial Nx in {32, 64} x Nn in {64, 128} x c in {16, 64}: 8 runs.

G1  relativistic homogeneous limit (unseeded control, Nx 8, c = 64, v_q = 0.03, T = 500, Nn 64 and 128) against the exact kinetic ODE
    studies/relgrid_ref.homogeneous(rel=True): max |Ebar - Ebar_ref| / max |Ebar_ref| <= 1e-6; |W_ext(T) - W_ref(T)| / |W_ref(T)| <= 1e-6; pump phase of Ebar
    on [450, 500] (relgrid_ref.demod_phase) within 1e-6 rad of the ODE.  Discrimination: the nonrelativistic exact ODE (rel=False) differs from the relativistic one
    by >= 0.1 in W_ext(T) (so that the test can fail).
G2  refinement pairs (12: Nx 32 vs 64, Nn 64 vs 128, c 16 vs 64, each with the other two settings at each of their 4 values). Observables dK_e (electron
    (gamma - 1) m c^2 change) and W_ext. A pair agrees up to the first t > 50 where |a - b| > 0.1 * running max |b| (b = Nx 64 / Nn 128 / c 64; nxconv rule).
    PASS iff all 12 pairs agree on both observables up to t = 400 (this covers the relgrid collapse time 367-392).
G3  positivity: no run loses positivity (run_adaptive stop: K_s <= 0 or k = 0 T_x <= 0) nor has min f_e / max f_e < -1e-3 at any save with t <= 400.
G4  ledger: |K_e + K_i - K(0) - (W_em + W_D + W_ext)| / max |W_ext| <= 1e-4 for t <= 400 in every run (K with (gamma - 1) m c^2; the closure's dissipation is
    part of this defect, as in the nonrelativistic Hermite records).
Certified window of a run: t_cert = min(T, first f_e < -1e-3 max, positivity stop, ledger > 1e-4, first disagreement in any pair it belongs to on dK_e / W_ext).
Measurements (no thresholds): collapse time = first t > 50 with |W_ext - W_lin| > 0.1 |W_lin|, W_lin the nonrelativistic homogeneous ODE (as relgrid_analysis);
E1D_e = (1/2) m (<v^2> - <v>^2/n) (v = u/gamma) change over E_th = 5e-4, trailing 200 omega_pe^-1 average (HHS measure), and dK_e / E_th, against relpic and relgrid.

Run: python studies/relhhs_gates.py g1 | prod | f1 | f2

FOLLOW-UP, declared after G1-G4 (all PASSED, every run certified to T = 1000; the deterministic runs show collapse at t = 392.5 and dE1D_e = 0.056 E_th at
1000, the relgrid values) and BEFORE any run below. No threshold of G1-G4 changes. The deterministic runs have no noise, so they cannot reproduce the PIC
dW_th (the PIC value is noise-seeded); and the certified window never ended, so the horizon is open.
F1  horizon: continue the base runs (Nx 32, Nn 64, c 64) and (Nx 64, Nn 128, c 64) to T = 2000 (same command with --T 2000; resumes the committed parts).
    Certified horizon of a run = min(T, first f_e/f_max < -1e-3, positivity stop, ledger > 1e-4 of max W_ext, first 10% disagreement of dK_e / W_ext between the
    two runs). PASS iff both reach T = 2000 with the horizon at 2000.
F2  shot-noise seeds (relgrid recipe, independent e and i, seed 0, modes 1..(Nx-1)//3 taken from a shared 21-mode draw), T = 1000, c = 64, for Np = 1000 npc with
    npc = 103 (the relpic ppc) and 2.05e5 (the relgrid load), Hermite (Nx 32, Nn 64) and (Nx 64, Nn 128). Per run the certified horizon is defined as in F1
    (the pair is the two Hermite resolutions of the same npc). PASS iff all four runs have horizon >= 400 (G3/G4 window). The measurement is dE1D_e (200-average)
    / E_th at the horizon and at 1000 where certified, against relpic R (7.3 / 16.1 / 20.0 / 21.5 at t = 400 / 600 / 800 / 1000) and the relgrid shot runs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relgrid_ref as X  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "relhhs" / "runs"
OUTD = HERE / "relhhs"
E_TH = 5e-4
G = {"G1_err": 1e-6, "G1_phase": 1e-6, "G1_discrim": 0.1, "G1_T": 500.0, "G2_rule": 0.1, "G2_t": 400.0, "G3_fneg": -1e-3, "G3_t": 400.0,
     "G4_ledger": 1e-4, "G4_t": 400.0}
NX, NN, CC = (32, 64), (64, 128), (16, 64)


def case(nx, nn, c):
    return f"rel_vq0.03_Nx{nx}_Nn{nn}_c{c}"


def sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()


def load(cs, d=RUNS):
    parts = sorted(d.glob(f"{cs}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    if not parts:
        return None
    cat, recs = {}, []
    for i, p in enumerate(parts):
        z = np.load(p)
        recs.append(json.loads(Path(str(p)[:-4] + ".json").read_text()))
        for k in ("t", "K_species", "W", "Ebar", "Ek2", "Uk", "E1D", "B", "fmin", "fmax", "top_frac"):
            cat.setdefault(k, []).append(z[k] if i == 0 else z[k][1:])  # drop the repeated start of each resumed part
    r = {k: np.concatenate(v) for k, v in cat.items()}
    r["recs"] = recs
    r["status"] = recs[-1]["status"]
    r["failure_reason"] = recs[-1]["failure_reason"]
    r["dK_e"] = r["K_species"][:, 0] - r["K_species"][0, 0]
    r["W_ext"] = r["W"][:, 2]
    r["ledger"] = (r["K_species"].sum(1) - r["K_species"].sum(1)[0]) - r["W"].sum(1)
    r["dE1D_e"] = r["E1D"][:, 0] - r["E1D"][0, 0]
    return r


def first_t(t, bad, tmin=-1.0):
    i = np.nonzero(bad & (t > tmin))[0]
    return float(t[i[0]]) if i.size else None


def agree_t(a, b, obs):
    """first t > 50 where |a - b| > 0.1 running max |b|; None if they agree on the common times."""
    ta, tb = np.round(a["t"], 6), np.round(b["t"], 6)
    t = np.intersect1d(ta, tb)
    x, y = a[obs][np.searchsorted(ta, t)], b[obs][np.searchsorted(tb, t)]
    bad = (t > 50) & (np.abs(x - y) > G["G2_rule"] * np.maximum.accumulate(np.abs(y)))
    return (float(t[np.argmax(bad)]) if bad.any() else None), float(t[-1])


def g1():
    out = {"repository_commit": sha(), "thresholds": G, "runs": {}}
    T = G["G1_T"]
    ref = X.homogeneous(0.03, T, rel=True)
    non = X.homogeneous(0.03, T, rel=False)
    for nn in NN:
        cs = f"rel_vq0.03_Nx8_Nn{nn}_c64_hom"
        r = load(cs)
        if r is None:
            continue
        t = r["t"]
        yr = ref.sol(t)
        eE = float(np.abs(r["Ebar"] - yr[2]).max() / np.abs(yr[2]).max())
        eW = float(abs(r["W_ext"][-1] - yr[3][-1]) / abs(yr[3][-1]))
        ph = X.demod_phase(t, r["Ebar"], X.OMEGA, T - 50, T)[0]
        ph_ref = X.demod_phase(t, yr[2], X.OMEGA, T - 50, T)[0]
        out["runs"][cs] = {"T": float(t[-1]), "Ebar_rel_err": eE, "W_ext_rel_err": eW, "phase_err_rad": float(abs(ph - ph_ref)),
                           "status": r["status"], "pass": bool(t[-1] >= T - 1e-9 and eE <= G["G1_err"] and eW <= G["G1_err"] and abs(ph - ph_ref) <= G["G1_phase"])}
    d = abs(non.sol(T)[3] - ref.sol(T)[3]) / abs(ref.sol(T)[3])
    out["nonrel_vs_rel_W_ext"] = float(d)
    out["discrimination_pass"] = bool(d >= G["G1_discrim"])
    out["pass"] = bool(out["runs"] and all(v["pass"] for v in out["runs"].values()) and len(out["runs"]) == len(NN) and out["discrimination_pass"])
    (OUTD / "gates_g1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def prod():
    out = {"repository_commit": sha(), "thresholds": G, "runs": {}, "pairs": {}}
    R = {(nx, nn, c): load(case(nx, nn, c)) for nx in NX for nn in NN for c in CC}
    for key, r in R.items():
        if r is None:
            continue
        t = r["t"]
        ratio = r["fmin"] / r["fmax"]
        led = np.abs(r["ledger"]) / max(np.abs(r["W_ext"]).max(), 1e-300)
        d = {"T": float(t[-1]), "status": r["status"], "failure_reason": r["failure_reason"],
             "t_fneg": first_t(t, ratio < G["G3_fneg"]), "t_ledger": first_t(t, led > G["G4_ledger"]),
             "max_ledger_t400": float(led[t <= G["G4_t"]].max()) if (t <= G["G4_t"]).any() else None}
        d["t_posstop"] = float(t[-1]) if r["status"] == "failure" and "positivity" in str(r["failure_reason"]) else None
        d["t_fail"] = float(t[-1]) if r["status"] == "failure" else None
        out["runs"][str(key)] = d
    for ax, vals in (("Nx", NX), ("Nn", NN), ("c", CC)):
        for key in R:
            idx = {"Nx": 0, "Nn": 1, "c": 2}[ax]
            if key[idx] != vals[0]:
                continue
            kb = list(key)
            kb[idx] = vals[1]
            kb = tuple(kb)
            if R[key] is None or R[kb] is None:
                continue
            res = {obs: agree_t(R[key], R[kb], obs) for obs in ("dK_e", "W_ext")}
            out["pairs"][f"{ax}:{key}->{kb}"] = {obs: {"t_disagree": v[0], "t_common": v[1]} for obs, v in res.items()}
    runs = out["runs"]
    n_have = len(runs)
    out["n_runs"] = n_have

    def ok_before(v, tlim):
        return v is None or v > tlim
    out["G2_pass"] = bool(len(out["pairs"]) == 12 and all(ok_before(o["t_disagree"], G["G2_t"]) and o["t_common"] >= G["G2_t"] for p in out["pairs"].values() for o in p.values()))
    out["G3_pass"] = bool(n_have == 8 and all(ok_before(v["t_fneg"], G["G3_t"]) and ok_before(v["t_fail"], G["G3_t"]) for v in runs.values()))
    out["G4_pass"] = bool(n_have == 8 and all(ok_before(v["t_ledger"], G["G4_t"]) for v in runs.values()))
    # certified window per run
    for key, r in R.items():
        if r is None:
            continue
        cands = [out["runs"][str(key)]["T"]] + [v for v in (out["runs"][str(key)]["t_fneg"], out["runs"][str(key)]["t_ledger"],
                                                            out["runs"][str(key)]["t_fail"]) if v is not None]
        for name, p in out["pairs"].items():
            if str(key) in name:
                cands += [o["t_disagree"] for o in p.values() if o["t_disagree"] is not None]
        out["runs"][str(key)]["t_cert"] = float(min(cands))
    (OUTD / "gates_prod.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for k, v in out["pairs"].items():
        print(k, v)


def horizon(r, other=None):
    """other may be None (no partner record): the pair terms are then absent."""
    t = r["t"]
    led = np.abs(r["ledger"]) / max(np.abs(r["W_ext"]).max(), 1e-300)
    c = {"T": float(t[-1]), "t_fneg": first_t(t, r["fmin"] / r["fmax"] < G["G3_fneg"]), "t_ledger": first_t(t, led > G["G4_ledger"]),
         "t_fail": float(t[-1]) if r["status"] == "failure" else None}
    if other is not None:
        for obs in ("dK_e", "W_ext"):
            c["t_pair_" + obs] = agree_t(r, other, obs)[0]
    c["horizon"] = float(min(v for v in c.values() if v is not None))
    return c


def f1():
    out = {"repository_commit": sha(), "runs": {}}
    A, B = load(case(32, 64, 64)), load(case(64, 128, 64))
    for nm, r, o in (("(32, 64, 64)", A, B), ("(64, 128, 64)", B, A)):
        out["runs"][nm] = horizon(r, o)
    out["pass"] = bool(all(v["horizon"] >= 2000.0 - 1e-9 for v in out["runs"].values()))
    (OUTD / "gates_f1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def f2():
    out = {"repository_commit": sha(), "runs": {}}
    for npc in ("103", "205000"):
        tag = f"_shot{npc}s0"
        A, B = load(case(32, 64, 64) + tag), load(case(64, 128, 64) + tag)
        if B is None:  # the Nx 64 run never completed a chunk (killed after 70 min in the first one): record it, no pair
            out["runs"][f"npc{npc}_Nx64_Nn128"] = {"horizon": None, "note": "no record: killed in the first 100-unit chunk"}
        for nm, r, o in (("Nx32_Nn64", A, B), ("Nx64_Nn128", B, A)):
            if r is None:
                continue
            h = horizon(r, o)
            t = r["t"]
            import relgrid_analysis as RA
            dE = RA.mavg(t, r["dE1D_e"]) / E_TH
            h["dE1D_over_Eth_at_horizon"] = float(dE[int(np.argmin(np.abs(t - h["horizon"])))])
            h["dE1D_over_Eth"] = {str(q): float(dE[int(np.argmin(np.abs(t - q)))]) for q in (200, 400, 600, 800, 1000) if q <= t[-1]}
            out["runs"][f"npc{npc}_{nm}"] = h
    out["pass"] = bool(len(out["runs"]) == 4 and all(v["horizon"] is not None and v["horizon"] >= 400.0 for v in out["runs"].values()))
    (OUTD / "gates_f2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"g1": g1, "prod": prod, "f1": f1, "f2": f2}[sys.argv[1]]()
