"""R1(b): finite temporal coherence of a prescribed drive (Lorentzian line by phase diffusion). Gates: studies/r1_common.py.

python studies/r1_coherence.py list | run KEY | all | mc | summary
Records: studies/r1/coh_<key>.json + .npz; linear Monte Carlo and derived expectation: studies/r1/coh_linear.npz;
summary -> studies/r1/coherence_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import darkspectrax as ds
import r1_common as c

V2 = "v2" in sys.argv[1:2]  # python studies/r1_coherence.py v2 CMD ...: v2 definition (c.V2_NOTE), files coh2_*
if V2:
    sys.argv.pop(1)
FAC, PFX = (2.0, "coh2_") if V2 else (1.0, "coh_")
OUT = Path(__file__).resolve().parent / "r1"
TAUS, NREAL, NMC, VQ = (30.0, 100.0, 300.0), 6, 400, 0.01
E0 = VQ * c.vte


def cases():
    L = [dict(key="tauinf_r0_Nn32_Nx8", tau=np.inf, r=0, Nn=32, Nx=8)]
    L += [dict(key=f"tau{int(t)}_r{r}_Nn32_Nx8", tau=t, r=r, Nn=32, Nx=8) for t in TAUS for r in range(NREAL)]
    L += [dict(key=f"tau{int(t)}_r0_Nn64_Nx8", tau=t, r=0, Nn=64, Nx=8) for t in TAUS] + [dict(key="tauinf_r0_Nn64_Nx8", tau=np.inf, r=0, Nn=64, Nx=8)]
    L += [dict(key="tau100_r0_Nn32_Nx16", tau=100.0, r=0, Nn=32, Nx=16), dict(key="tauinf_r0_Nn32_Nx16", tau=np.inf, r=0, Nn=32, Nx=16)]
    return L


def run_case(cs):
    z = c.noise_knots(cs["tau"], cs["r"], fac=FAC)
    m = c.mobile_model(cs["Nx"], cs["Nn"], mode="prescribed_drive", E0=E0, cls=c.PhasedModel, z=tuple(z))
    out = c.go(m, c.initial(m), c.T_END)
    rec = {"key": cs["key"], "tau_c": cs["tau"], "realization": cs["r"], "Nn": cs["Nn"], "Nx": cs["Nx"], "E0": E0, "definition": "v2" if V2 else "v1", "phase_var_factor": FAC,
           "status": out["status"], "failure_reason": out["failure_reason"], "t_reached": float(out["t"][-1]),
           "events": len(out["events"]), "ledger_over_Wscale": c.ledger_ok(out), "first_nonpositive": c.first_nonpositive(out),
           "wall_time": out["wall"], "parent_commit": ds.PARENT_COMMIT,
           "repository_commit": ds._simulation._git_sha(), "command": f"python studies/r1_coherence.py run {cs['key']}"}
    if cs["Nn"] == 32 and cs["Nx"] == 8:
        ti, R, S = c.identity_residual(m, out)
        rec["identity_max_rel"] = float(np.abs(R).max() / (c.w_L ** 2 * S.max()))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{PFX}{cs['key']}.npz", t=out["t"], K_species=out["K_species"], W=out["W"], Tx=out["Tx"])
    (OUT / f"{PFX}{cs['key']}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(rec, flush=True)


def mc():
    """Linear mean-field Monte Carlo (NMC realizations, seeds 0..NMC-1) and the derived expectation."""
    tg = np.arange(0.0, c.T_END + 0.5, 2.0)
    D = {"t": tg}
    for tau in (np.inf,) + TAUS:
        nm = f"tau{'inf' if not np.isfinite(tau) else int(tau)}"
        n = 1 if not np.isfinite(tau) else NMC
        D[f"{nm}_W"] = np.array([c.linear_work_realization(tg, c.noise_knots(tau, s, fac=FAC), E0) for s in range(n)])
        te, We, P = c.expected_work(tau, E0, out_every=20, fac=FAC)  # dt 0.1 -> every 2
        D[f"{nm}_expected_W"], D[f"{nm}_expected_P"] = We, P[::20]
        print(nm, "done", flush=True)
    np.savez_compressed(OUT / f"{PFX}linear.npz", **D)


def load(key):
    return json.loads((OUT / f"{PFX}{key}.json").read_text()), np.load(OUT / f"{PFX}{key}.npz")


def late_slope(t, W, lo):
    sel = t >= lo
    return float(np.polyfit(t[sel], W[sel], 1)[0])


def summary():
    L = np.load(OUT / f"{PFX}linear.npz")
    tl = L["t"]
    S = {"E0": E0, "definition": "v2" if V2 else "v1", "gates": {}, "tau": {}}
    ledg, pos, ident = [], [], []
    nx_caps = []
    for key in ("tau100_r0", "tauinf_r0"):
        a, da = load(f"{key}_Nn32_Nx8")
        b, db = load(f"{key}_Nn32_Nx16")
        n = min(da["t"].size, db["t"].size)
        pr = {"dK_e": c.disagreement(da["t"][:n], da["K_species"][:n, 0], db["K_species"][:n, 0]),
              "W_ext": c.disagreement(da["t"][:n], da["W"][:n, 2], db["W"][:n, 2])}
        S[f"Nx8_vs_Nx16_{key}"] = pr
        nx_caps += [v if v is not None else float(da["t"][n - 1]) for v in pr.values()] + [a["t_reached"], b["t_reached"]]
        ledg.append(b["ledger_over_Wscale"]), pos.append(b["first_nonpositive"])
    ex_inf = c.prescribed_linear(tl, E0)
    S["gates"]["G8_quadrature_rel_err"] = float(np.abs(L["tauinf_expected_W"][1:] - ex_inf[1:]).max() / ex_inf[-1])
    S["gates"]["G8_pass"] = bool(S["gates"]["G8_quadrature_rel_err"] <= 2e-3)
    g9 = []
    for tau in (None,) + TAUS:
        nm = "tauinf" if tau is None else f"tau{int(tau)}"
        tcap = c.CERTIFIED[VQ]
        r0, d0 = load(f"{nm}_r0_Nn32_Nx8")
        rn, dn = load(f"{nm}_r0_Nn64_Nx8")
        t = d0["t"]
        n = min(t.size, dn["t"].size)
        pr = {"dK_e": c.disagreement(t[:n], d0["K_species"][:n, 0], dn["K_species"][:n, 0]),
              "W_ext": c.disagreement(t[:n], d0["W"][:n, 2], dn["W"][:n, 2])}
        reals = [0] if tau is None else range(NREAL)
        recs = [load(f"{nm}_r{r}_Nn32_Nx8") for r in reals]
        caps = [v if v is not None else float(t[n - 1]) for v in pr.values()] + nx_caps + [tcap] + [r["t_reached"] for r, _ in recs] + [rn["t_reached"]]
        caps += [r["first_nonpositive"] for r, _ in recs if r["first_nonpositive"] is not None]
        window = min(caps)
        Wk = np.array([np.interp(tl, d["t"], d["W"][:, 2]) for _, d in recs])
        Wl = L[f"{nm}_W"][:Wk.shape[0]] if tau is not None else L["tauinf_W"]
        sel = tl <= window
        per = []
        for i in range(Wk.shape[0]):
            per.append({"t_dev_10pct": c.disagreement(tl[sel], Wk[i][sel], Wl[i][sel]),
                        "ratio_late": float(Wk[i][(tl >= 0.8 * window) & sel][-1] / Wl[i][(tl >= 0.8 * window) & sel][-1])})
        lo = 0.8 * window
        e = {"claim_window": window, "pairs_Nn32_vs_Nn64": pr, "n_realizations_kinetic": Wk.shape[0],
             "mean_W_kin_final": float(Wk.mean(0)[sel][-1]), "mean_W_lin_same_real_final": float(Wl[:Wk.shape[0]].mean(0)[sel][-1]),
             "expected_W_lin_final": float(np.interp(window, tl, L[f"{nm}_expected_W"])),
             "late_slope_kin": late_slope(tl[sel], Wk.mean(0)[sel], lo), "late_slope_lin_same_real": late_slope(tl[sel], Wl[:Wk.shape[0]].mean(0)[sel], lo),
             "late_slope_expected": late_slope(tl[sel], L[f"{nm}_expected_W"][sel], lo),
             "late_slope_MC400": late_slope(tl[sel], Wl.mean(0)[sel], lo) if tau is not None else None,
             "late_slope_asymptote_E0sq_tau_wL2_over_4": None if tau is None else float(E0 ** 2 * tau * c.w_L ** 2 / 4),
             "per_realization": per,
             "ratio_late_mean": float(np.mean([p["ratio_late"] for p in per])),
             "ratio_late_se": float(np.std([p["ratio_late"] for p in per], ddof=1) / np.sqrt(len(per))) if len(per) > 1 else None,
             "G10_pass": all(p["t_dev_10pct"] is None for p in per),
             "ledger": [r["ledger_over_Wscale"] for r, _ in recs] + [rn["ledger_over_Wscale"]],
             "identity_max_rel": max(r["identity_max_rel"] for r, _ in recs)}
        if tau is not None:
            mc_ = L[f"{nm}_W"]
            mean, se = mc_.mean(0), mc_.std(0, ddof=1) / np.sqrt(mc_.shape[0])
            idx = [int(np.argmin(np.abs(tl - tt))) for tt in np.linspace(c.T_END / 10, c.T_END, 10)]
            z = np.abs(mean[idx] - L[f"{nm}_expected_W"][idx]) / se[idx]
            e["G9_max_z"], e["G9_pass"] = float(z.max()), bool(z.max() <= 4)
            g9.append(e["G9_pass"])
            e["MC_mean_W_final"], e["MC_se_final"] = float(mean[-1]), float(se[-1])
            e["realized_phase_var_rate_x_tau_c"] = c.phase_variance_rate(tau, FAC) * tau
            if Wk.shape[0] > 1:
                at = int(np.argmin(np.abs(tl - window)))
                kse = Wk[:, at].std(ddof=1) / np.sqrt(Wk.shape[0])
                e["G12_z_kin_mean_vs_expected"] = float((Wk[:, at].mean() - L[f"{nm}_expected_W"][at]) / kse)
        ledg += e["ledger"]
        pos += [r["first_nonpositive"] for r, _ in recs] + [rn["first_nonpositive"]]
        ident.append(e["identity_max_rel"])
        S["tau"][nm] = e
        print(nm, {k: e[k] for k in ("claim_window", "ratio_late_mean", "G10_pass")}, flush=True)
    S["gates"].update({"G1_ledger_max": max(ledg), "G1_pass": bool(max(ledg) <= c.LEDGER_TOL),
                       "G2_first_nonpositive": [x for x in pos if x is not None],
                       "G4_identity_max": max(ident), "G4_pass": bool(max(ident) <= c.ID_TOL), "G9_pass": all(g9)})
    if V2:
        S["gates"]["G11_rate_x_tau_c"] = {nm: v["realized_phase_var_rate_x_tau_c"] for nm, v in S["tau"].items() if "realized_phase_var_rate_x_tau_c" in v}
        S["gates"]["G11_pass"] = all(abs(x - 2.0) <= 0.04 for x in S["gates"]["G11_rate_x_tau_c"].values())
    (OUT / ("coherence2_summary.json" if V2 else "coherence_summary.json")).write_text(json.dumps(S, indent=1, default=float) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        print("\n".join(cs["key"] for cs in cases()))
    elif cmd == "run":
        run_case(next(cs for cs in cases() if cs["key"] == sys.argv[2]))
    elif cmd == "all":
        for cs in cases():
            if not (OUT / f"{PFX}{cs['key']}.json").exists():
                run_case(cs)
    elif cmd == "mc":
        OUT.mkdir(exist_ok=True)
        mc()
    elif cmd == "summary":
        summary()
