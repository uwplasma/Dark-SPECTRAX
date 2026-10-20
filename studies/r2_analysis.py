"""R2 analysis: implements exactly the rules of studies/r2_gates.py (no rule changes).  Run: python studies/r2_analysis.py [runs_dir]
-> studies/r2/summary.json.  Implementation details that are not rule changes: the end of the rate interval is tb = the largest save
<= Wn - pi/omega (so that the cycle-averaging window lies inside the certified window); cycle averages are means over |t - tq| <= pi/omega."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h08  # noqa: E402
import r1_common as c1  # noqa: E402
import r2_gates as g  # noqa: E402
import r2_run  # noqa: E402
import r2_theory as th  # noqa: E402

RUNS = HERE / "r2" / "runs"


def load(case):
    h08.RUNS = RUNS
    return h08.load(case)


def om(tag):
    return r2_run.omega_of(tag)


def cmean(t, y, tq, w):
    sel = np.abs(t - tq) <= np.pi / w
    return float(np.mean(y[sel]))


def first_bad(h):
    bad = (h["K_species"] <= 0).any(axis=1) | (h["Tx"] <= 0).any(axis=1)
    return float(h["t"][np.argmax(bad)]) if bad.any() else None


def t_pair(a, b, key):
    ta, tb = np.round(a["t"], 6), np.round(b["t"], 6)
    t = np.intersect1d(ta, tb)
    x, y = a[key][np.searchsorted(ta, t)], b[key][np.searchsorted(tb, t)]
    d = c1.disagreement(t, x, y)
    return d if d is not None else float(t[-1])


def ledger_ratio(h):
    return float(np.abs(h["ledger"]).max() / max(np.abs(h["Wx"]).max(), 1e-300))


def delta_of(tag):
    if tag == "res":
        return g.DELTA
    nb = int(np.floor(g.DELTA * abs(om(tag) - th.wL) / (2 * np.pi)))
    return nb * 2 * np.pi / abs(om(tag) - th.wL)


def lin_curve(vq, nu, tag, t):
    return th.exact_linear(vq, nu, om(tag), t)[1]


def rates(h, vq, nu, tag, tb, Wlin=None):
    w = om(tag)
    D = delta_of(tag)
    Wl = lin_curve(vq, nu, tag, h["t"]) if Wlin is None else Wlin
    c = lambda y, tq: cmean(h["t"], y, tq, w)  # noqa: E731
    Pk = (c(h["Wx"], tb) - c(h["Wx"], tb - D)) / D
    Pl = (c(Wl, tb) - c(Wl, tb - D)) / D
    pss = th.p_ss(vq * th.vte, nu, w)
    return {"P_kin": Pk, "P_lin": Pl, "p_ss": pss, "R_late": Pk / Pl, "R_cum": c(h["Wx"], tb) / c(Wl, tb), "f_ss": Pl / pss,
            "P_kin_over_pss": Pk / pss, "Delta": D, "Wext_tb": c(h["Wx"], tb), "Wlin_tb": c(Wl, tb)}, Wl


def lsq(X, y):
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), int(len(y))


def main(argv):
    global RUNS
    if argv:
        RUNS = Path(argv[0])
    out = {"gates": {}, "T0": {}, "controls": {}, "runs": {}, "cases": {}, "fits": {}}
    # T0 theory self-check
    t0ok = []
    for tag in g.OMEGA:
        t = np.linspace(0, 400, 40001)
        W = th.exact_linear(0.03, g.T0_NU, om(tag), t)[1]
        n_per = 80 * 2 * np.pi / om(tag)
        sel = t >= t[-1] - n_per
        # mean slope over the last 80 periods by endpoint cycle averages
        a, b = t[-1] - n_per, t[-1] - np.pi / om(tag)
        sl = (cmean(t, W, b, om(tag)) - cmean(t, W, a + np.pi / om(tag), om(tag))) / (b - a - np.pi / om(tag))
        ref = th.p_ss(0.03 * th.vte, g.T0_NU, om(tag))
        out["T0"][tag] = {"slope": sl, "p_ss": ref, "rel": sl / ref - 1}
        t0ok.append(abs(sl / ref - 1) <= g.T0_TOL)
    out["gates"]["T0"] = all(t0ok)  # as declared: fails (window longer than record, NaN); see T0b
    t0b = []
    for tag in g.OMEGA:
        t = np.linspace(0, g.T0B_T, int(g.T0B_T * 100) + 1)
        W = th.exact_linear(0.03, g.T0_NU, om(tag), t)[1]
        n_per = g.T0B_PERIODS * 2 * np.pi / om(tag)
        a, b = t[-1] - n_per, t[-1] - np.pi / om(tag)
        sl = (cmean(t, W, b, om(tag)) - cmean(t, W, a + np.pi / om(tag), om(tag))) / (b - a - np.pi / om(tag))
        ref = th.p_ss(0.03 * th.vte, g.T0_NU, om(tag))
        out["T0b"] = out.get("T0b", {})
        out["T0b"][tag] = {"slope": sl, "p_ss": ref, "rel": sl / ref - 1}
        t0b.append(abs(sl / ref - 1) <= g.T0_TOL)
    out["gates"]["T0b"] = all(t0b)
    # controls
    c0 = []
    for conv, vq, nu, tag, _, _ in [c for c in g.cases() if c[5]]:
        h = load(r2_run.case_name(conv, vq, nu, tag, 8, 64, True))
        if h is None:
            c0.append(False)
            continue
        Eb, W = th.exact_linear(vq, nu, om(tag), h["t"])
        r = {"t_end": float(h["t"][-1]), "Ebar_err": float(np.abs(h["Ebar"] - Eb).max() / np.abs(Eb).max()),
             "Wext_err": float(np.abs(h["Wx"] - W).max() / np.abs(W).max())}
        out["controls"][r2_run.case_name(conv, vq, nu, tag, 8, 64, True)] = r
        c0.append(r["Ebar_err"] <= g.C0_TOL and r["Wext_err"] <= g.C0_TOL and r["t_end"] >= 1000 - 1e-9)
    out["gates"]["C0"] = bool(c0) and all(c0)
    # runs: L1-L3
    hs, l1, l2, l3 = {}, [], [], []
    for conv, vq, nu, tag, mem, ctl in g.cases():
        if ctl:
            continue
        Nx, Nn = g.MEMBERS[mem]
        name = r2_run.case_name(conv, vq, nu, tag, Nx, Nn, False)
        h = load(name)
        if h is None:
            continue
        hs[name] = h
        E0 = vq * th.vte
        r = {"t_end": float(h["t"][-1]), "status": h["recs"][-1]["status"], "failure_reason": h["recs"][-1]["failure_reason"],
             "steps": sum(x["steps"] for x in h["recs"]), "wall_time": sum(x["wall_time"] for x in h["recs"]),
             "max_ledger_over_Wext": ledger_ratio(h), "max_R_over_wL2E0": float(np.abs(h["R"]).max() / (th.wL ** 2 * E0)),
             "max_Psum_over_Pe": float(np.abs(h["P"].sum(1)).max() / max(np.abs(h["P"][:, 0]).max(), 1e-300)),
             "t_nonpositive": first_bad(h)}
        out["runs"][name] = r
        l1.append(r["max_ledger_over_Wext"] <= g.L1_TOL)
        l2.append(r["max_Psum_over_Pe"] <= g.L2_TOL)
        l3.append(r["max_R_over_wL2E0"] <= g.L3_TOL)
    out["gates"].update(L1=bool(l1) and all(l1), L2=bool(l2) and all(l2), L3=bool(l3) and all(l3))
    out["gate_failures"] = {"L1": [k for k, v in out["runs"].items() if v["max_ledger_over_Wext"] > g.L1_TOL],
                            "L2": [k for k, v in out["runs"].items() if v["max_Psum_over_Pe"] > g.L2_TOL],
                            "L3": [k for k, v in out["runs"].items() if v["max_R_over_wL2E0"] > g.L3_TOL]}
    # cases
    k2, k3, cv = [], [], []
    wins = {}
    for vq in g.VQ:
        for nu in g.NU:
            for tag in g.OMEGA:
                nm = {m: r2_run.case_name("EM", vq, nu, tag, *g.MEMBERS[m], False) for m in g.MEMBERS}
                B, X, N = (hs.get(nm[m]) for m in "BXN")
                if B is None or X is None or N is None:  # incomplete case (partial data while runs are in progress)
                    continue
                key = f"vq{vq:g}_nu{nu:g}_w{tag}"
                mem = [h for h in (B, X, N) if h is not None]
                caps = {"t_reached": min(float(h["t"][-1]) for h in mem), "T": g.T_END}
                nonpos = [first_bad(h) for h in mem if first_bad(h) is not None]
                if nonpos:
                    caps["nonpositive"] = min(nonpos)
                for lab, o in (("X", X), ("N", N)):
                    if o is not None:
                        for kk in ("dKe", "Wx"):
                            caps[f"pair_B{lab}_{kk}"] = t_pair(B, o, kk)
                Wn = min(caps.values())
                Te0 = B["Te"][0]
                above = np.nonzero(B["Te"] / Te0 >= g.TE_REL)[0]
                Wp = min(Wn, float(B["t"][above[0]])) if above.size else Wn
                ledger_ok = all(ledger_ratio(h) <= g.L1_TOL for h in mem)
                wins[key] = Wn
                rec = {"Wn": Wn, "Wp": Wp, "caps": caps, "certified": bool(ledger_ok and Wn > 50), "members": len(mem)}
                w = om(tag)
                if rec["certified"] and Wn >= g.W_MIN:
                    tb = float(B["t"][np.searchsorted(B["t"], Wn - np.pi / w, side="right") - 1])
                    rec["tb"] = tb
                    rr, Wl = rates(B, vq, nu, tag, tb)
                    rec["B"] = rr
                    for lab, o in (("X", X), ("N", N)):
                        if o is not None:
                            rec[lab] = rates(o, vq, nu, tag, tb)[0]
                    ok = abs(rr["R_late"] - 1) <= g.TOL_R and abs(rr["R_cum"] - 1) <= g.TOL_R
                    spread = max(abs(rec[lab]["R_late"] - rr["R_late"]) for lab in "XN" if lab in rec)
                    rec["R_late_spread"] = spread
                    rec["K2"] = bool(ok and spread <= g.TOL_REF)
                    k2.append(rec["K2"])
                    if 0.9 <= rr["f_ss"] <= 1.1:
                        rec["K3"] = bool(abs(rr["P_kin_over_pss"] - 1) <= g.TOL_SS)
                        k3.append(rec["K3"])
                    c = lambda y, tq: cmean(B["t"], y, tq, w)  # noqa: E731
                    Wx = c(B["Wx"], tb)
                    Q = np.interp(tb, B["t"], B["Qcum"][:, 1])
                    rec["report"] = {"Te_over_Te0": c(B["Te"], tb) / Te0, "Ti_over_Ti0": c(B["Ti"], tb) / B["Ti"][0],
                                     "dKe_over_Wext": c(B["dKe"], tb) / Wx, "dKi_over_Wext": (B["K_species"][np.argmin(abs(B["t"] - tb)), 1]
                                                                                            - B["K_species"][0, 1]) / Wx,
                                     "Qi_over_Wext": float(Q / Wx), "fluct_field": float(B["Ek2"][np.argmin(abs(B["t"] - tb)), 1:].sum())}
                    # ratio along the run versus heating
                    D = rr["Delta"]
                    path = []
                    for tq in np.arange(300.0, tb + 1e-9, 100.0):
                        if tq - D < 0:
                            continue
                        r_, _ = rates(B, vq, nu, tag, float(tq), Wl)
                        path.append([float(tq), r_["R_late"], c(B["Te"], tq) / Te0])
                    rec["path_t_R_Te"] = path
                    # convention comparison
                    G = hs.get(r2_run.case_name("G", vq, nu, tag, 32, 64, False))
                    if G is not None and G["t"][-1] >= tb + np.pi / w:
                        cg = lambda y, tq: cmean(G["t"], y, tq, w)  # noqa: E731
                        gk = {"Wx": cg(G["Wx"], tb) / Wx - 1, "dKe": cg(G["dKe"], tb) / c(B["dKe"], tb) - 1,
                              "Te": cg(G["Te"], tb) / c(B["Te"], tb) - 1, "Ti": cg(G["Ti"], tb) / c(B["Ti"], tb) - 1,
                              "Qi": float(np.interp(tb, G["t"], G["Qcum"][:, 1]) / Q - 1),
                              "R_late_G": rates(G, vq, nu, tag, tb)[0]["R_late"]}
                        rec["CV"] = gk
                        rec["CV_pass"] = bool(abs(gk["Wx"]) <= g.CV_W and abs(gk["dKe"]) <= g.CV_K and abs(gk["Te"]) <= g.CV_K)
                        cv.append(rec["CV_pass"])
                out["cases"][key] = rec
    out["gates"]["K2_all_pass"] = bool(k2) and all(k2)
    out["gates"]["K3_all_pass"] = bool(k3) and all(k3)
    out["gates"]["CV_all_pass"] = bool(cv) and all(cv)
    out["counts"] = {"K2": [int(sum(k2)), len(k2)], "K3": [int(sum(k3)), len(k3)], "CV": [int(sum(cv)), len(cv)]}
    # K4 fits
    for tag in g.OMEGA:
        rows = [(float(k.split("_")[0][2:]), float(k.split("_")[1][2:]), r) for k, r in out["cases"].items()
                if k.endswith(f"_w{tag}") and "B" in r]
        if len(rows) < 4:
            continue
        X = [[1.0, np.log(v), np.log(n)] for v, n, _ in rows]
        res = {"n": len(rows)}
        for lab, key in (("kin", "P_kin"), ("lin", "P_lin")):
            y = np.log([max(r["B"][key], 1e-300) for _, _, r in rows])
            b, s, _ = lsq(X, y)
            res[lab] = {"a": b[0], "p": b[1], "q": b[2], "se_a": s[0], "se_p": s[1], "se_q": s[2]}
        lr = np.log([r["B"]["R_late"] for _, _, r in rows])
        res["kappa"] = float(np.exp(lr.mean()))
        res["kappa_se"] = float(res["kappa"] * lr.std(ddof=1) / np.sqrt(len(lr)))
        res["R_late_range"] = [float(np.exp(lr.min())), float(np.exp(lr.max()))]
        out["fits"][tag] = res
    out["repository_commit"] = h08._sim._git_sha()
    (HERE / "r2").mkdir(exist_ok=True)
    (HERE / "r2" / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"gates": out["gates"], "counts": out["counts"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
