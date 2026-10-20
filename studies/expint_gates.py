"""expint: integrating-factor (Lawson) treatment of the diagonal closure damping in the Dark Dopri8 solve.

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).
Motivation: studies/stiffdiag (docs/results.md) found rho(J) = nu_e = c sqrt(2 Nn) max|E|/a exactly (real, negative) at the
H05 v_q/v_te = 0.1 failure and a 6-9x step reduction when the closure is removed from the right-hand side. SPECTRAX
LawsonSolver (integration/dark-baseline 4090fbe, PR #81) integrates -nu_s s(n) C_s exactly with nu frozen at the step
start (order of the wrapped Dopri8 kept: spectrax/_lawson.py); fields, dark fields, work ledger and basis states have rate zero.
Streaming is NOT exponentiated (not built); Dopri8 handles it as before.

Records. Baseline = the committed stiffdiag Dopri8 records studies/nxconv/runs/vq0.1_Nx64_Nn128_c{4,64}_nf1e-14_sd.part*
(Nx 64, Nn 128, v_q/v_te 0.1, pump frame + remaps, noise floor 1e-14, rtol 1e-10, atol 1e-14, 100-unit chunks to
t = 500 (c = 4) / 700 (c = 64), then 20-unit chunks). Lawson runs: same command with `--lawson --tag _li` and the same
chunking (studies/expint_batch.sh), GPU 0 of the office machine, x64. Step counts are the records' `steps` (accepted +
rejected); wall time = compile + run time of the parts.

Chunk boundaries and window. T_cmp(c) = the last baseline chunk end of a SUCCESSFUL baseline part that is <= the Lawson
run's t_end (c = 4: the baseline's final part ends in the step-budget failure at 578; c = 64: positivity at 760).
Accuracy window: t <= 500, common save times (every 0.5 time unit).

Gates, per case c in {4, 64}:
 D2   steps(baseline, t <= T_cmp) / steps(Lawson, t <= T_cmp) >= 3  OR  the same ratio of (compile + run) wall time >= 3.
      Wall time includes the per-step overhead of the transform and the recomputed FSAL stage. The Lawson run must reach T_cmp
      with status success until then (a solver failure before T_cmp fails D2). D2 as a whole passes only if BOTH cases pass.
 A    dK_e(t) = K_e(t) - K_e(0) and W_ext(t) (ledger column 2) at t <= 500: max |Lawson - baseline| / max |baseline| <= 1e-6 for
      each. The records have no own refinement-uncertainty estimate (no rtol-refined rerun of these two cases exists), so the
      criterion is the absolute 1e-6 and no looser fallback is declared; any larger difference is reported as a failure.
 L    ledger: max_t |(K(t) - K(0)) - sum_j W_j(t)| / max_t |W_ext| <= 1e-9 over the whole Lawson run (the gpures G3 definition).
 P    positivity: the first save with K_s <= 0 or k=0 T_x <= 0 of the Lawson run is not earlier than that of the baseline
      (baseline without such a save: not earlier than the baseline's last save, t_end).
Platform note (added with the analysis code before the Lawson analysis, no gate changed): GPU 0 was busy (other users, 100%) for the
whole session, so the Lawson runs were made on the office CPU (JAX_PLATFORMS=cpu, 2 threads, nice) with the identical script and
chunking; the committed baseline is a GPU record. Wall ratios are therefore not comparable and D2 is decided by the step ratio
only (still "OR wall" as declared, but wall is reported, not used); to separate platform from integrator the plain Dopri8 was rerun
on the same CPU to t = 500 (tag `_sdcpu`, ungated controls A_ungated_*).
Ungated (reported): window reached by each Lawson run (t_end, t_pos), steps and wall ratios at 500 and at the end, the
step-to-failure trajectory, and per-step cost (studies/expint_cost.py). Expected from the design note
(docs/design/implicit-exponential-hermite.md D2): speed-up, not a longer window (the AW truncation instability is not removed).

Ordinary B10 (studies/b10.py, M_mi100_ord_j10, nu = 100 constant hypercollision, Nn 32, Nx 40, rtol 1e-10, atol 1e-12):
 B1   steps(record M_mi100_ord_j10.json) / steps(M_mi100_ord_j10_li.json) >= 3  OR  wall ratio >= 3, the wall ratio taken
      between the Lawson run and a plain rerun on the same machine in the same session (`--rerun`, record _rerun.json);
      the committed record's own wall time is from another session and is reported but not gated.
 B2   fitted sideband growth rate gamma_fit within 1e-3 relative of the committed record (and the record's window within 1%).
 B3   max ledger defect / U_field0 <= 1e-9 and Gauss residuals <= 1e-12 A_p; status success.
Run: python studies/expint_gates.py -> studies/expint/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nxconv_analysis as na  # noqa: E402
from nxconv_analysis import t_pos  # noqa: E402

D = Path(__file__).resolve().parent / "expint"
TOL_A, TOL_L = 1e-6, 1e-9
T_ACC = 500.0


SD = Path(__file__).resolve().parent / "stiffdiag" / "runs"   # the committed Dopri8 baseline records live here


def herm(case_name):
    """Records of ``case_name``: the baseline (`_sd`) from studies/stiffdiag/runs, the Lawson (`_li`) from studies/nxconv/runs."""
    old = na.R
    na.R = SD if case_name.endswith("_sd") else old
    try:
        return na.herm(case_name)
    finally:
        na.R = old


def case(c, tag):
    return f"vq0.1_Nx64_Nn128_c{c}_nf1e-14{tag}"


def ledger(h):
    return float(np.abs(h["K_species"].sum(1) - h["K_species"][0].sum() - h["W"].sum(1)).max()
                 / max(np.abs(h["Wx"]).max(), 1e-300))


def cumulative(h):
    """(chunk end times, cumulative steps, cumulative wall) of the successful and failed parts, in order."""
    r = h["recs"]
    return (np.array([x["t_end"] for x in r]), np.cumsum([x["steps"] for x in r]),
            np.cumsum([x["compile_time"] + x["run_time"] for x in r]), [x["status"] for x in r])


def at(h, tc):
    ends, steps, wall, _ = cumulative(h)
    i = np.nonzero(np.abs(ends - tc) < 1e-6)[0]
    return (float(steps[i[0]]), float(wall[i[0]])) if i.size else (None, None)


def compare(base, li):
    t = np.round(base["t"], 6)
    tl = np.round(li["t"], 6)
    common = np.intersect1d(t[t <= T_ACC], tl[tl <= T_ACC])
    out = {"n_common_saves": int(common.size), "t_common_max": float(common.max()) if common.size else None}
    ib, il = np.searchsorted(t, common), np.searchsorted(tl, common)
    for k in ("dKe", "Wx"):
        b, a = base[k][ib], li[k][il]
        out[k] = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    return out


def main():
    D.mkdir(exist_ok=True)
    S = {"cases": {}}
    ok_d2 = []
    for c in (4, 64):
        b, li = herm(case(c, "_sd")), herm(case(c, "_li"))
        if b is None or li is None:
            S["cases"][c] = {"missing": "baseline" if b is None else "lawson"}
            ok_d2.append(False)
            continue
        eb, _, _, stb = cumulative(b)
        el, _, _, stl = cumulative(li)
        good = [e for e, s in zip(eb, stb) if s == "success" and e <= el[-1] + 1e-6]
        Tc = float(max(good)) if good else None
        sb, wb = at(b, Tc) if Tc else (None, None)
        sl, wl = at(li, Tc) if Tc else (None, None)
        fail_before = any(s == "failure" and e < Tc - 1e-6 for e, s in zip(el, stl)) if Tc else True
        r = {"T_cmp": Tc, "steps_base": sb, "steps_lawson": sl, "wall_base": wb, "wall_lawson": wl}
        if sb and sl:
            r["steps_ratio"], r["wall_ratio"] = sb / sl, wb / wl
            r["D2"] = bool((r["steps_ratio"] >= 3 or r["wall_ratio"] >= 3) and not fail_before)
        else:
            r["D2"] = False
        s5b, w5b = at(b, T_ACC)
        s5l, w5l = at(li, T_ACC)
        if s5b and s5l:
            r["steps_ratio_500"], r["wall_ratio_500"] = s5b / s5l, w5b / w5l
        ok_d2.append(r["D2"])
        r["A"] = compare(b, li)
        r["A_pass"] = bool(r["A"]["dKe"] <= TOL_A and r["A"]["Wx"] <= TOL_A)
        cp = herm(case(c, "_sdcpu"))  # ungated platform control: plain Dopri8 on the same CPU as the Lawson runs, t <= 500
        if cp is not None:
            r["A_ungated_cpu_plain_vs_gpu_record"] = compare(b, cp)   # platform effect alone
            r["A_ungated_lawson_vs_cpu_plain"] = compare(cp, li)       # integrator effect alone
            r["steps_cpu_plain_to_500"] = at(cp, T_ACC)[0]
        r["L"] = ledger(li)
        r["L_pass"] = bool(r["L"] <= TOL_L)
        r["L_baseline"] = ledger(b)
        tpb, tpl = t_pos(b), t_pos(li)
        r["t_pos_base"], r["t_pos_lawson"] = tpb, tpl
        ref = tpb if tpb is not None else float(b["t"][-1])
        r["P_pass"] = bool(tpl is None or tpl >= ref)
        r["t_end_base"], r["t_end_lawson"] = float(b["t"][-1]), float(li["t"][-1])
        r["status_lawson"] = stl[-1]
        r["steps_total_base"], r["steps_total_lawson"] = float(cumulative(b)[1][-1]), float(cumulative(li)[1][-1])
        S["cases"][c] = r
    S["D2_all"] = bool(all(ok_d2))
    # B10
    B = Path(__file__).resolve().parent / "b10"
    rec, li = B / "M_mi100_ord_j10.json", B / "M_mi100_ord_j10_li.json"
    rr = B / "M_mi100_ord_j10_rerun.json"
    if rec.exists() and li.exists():
        a, b = json.loads(rec.read_text()), json.loads(li.read_text())
        bb = {"steps_record": a["steps"], "steps_lawson": b["steps"], "steps_ratio": a["steps"] / b["steps"],
              "wall_record_other_session": a["wall_time"], "wall_lawson": b["wall_time"]}
        if rr.exists():
            c = json.loads(rr.read_text())
            bb.update(steps_rerun=c["steps"], wall_rerun=c["wall_time"], wall_ratio=c["wall_time"] / b["wall_time"],
                      rerun_gamma_fit=c["results"][0]["gamma_fit"])
        ga, gb = a["results"][0], b["results"][0]
        bb["gamma_fit_record"], bb["gamma_fit_lawson"] = ga["gamma_fit"], gb["gamma_fit"]
        if ga["gamma_fit"] and gb["gamma_fit"]:
            bb["gamma_rel_diff"] = abs(gb["gamma_fit"] - ga["gamma_fit"]) / abs(ga["gamma_fit"])
            wa, wb_ = np.array(ga["window"]), np.array(gb["window"])
            bb["window_rel_diff"] = float(np.abs(wb_ - wa).max() / wa.max())
        bb["ledger_over_U0"] = b["max_ledger_over_U0"]
        bb["gauss_over_Ap"] = b["max_gauss_over_Ap"]
        bb["B1"] = bool(bb["steps_ratio"] >= 3 or bb.get("wall_ratio", 0) >= 3)
        bb["B2"] = bool(bb.get("gamma_rel_diff", 1) <= 1e-3 and bb.get("window_rel_diff", 1) <= 1e-2)
        bb["B3"] = bool(b["max_ledger_over_U0"] <= 1e-9 and max(b["max_gauss_over_Ap"]) <= 1e-12 and b["status"] == "success")
        S["b10"] = bb
    (D / "summary.json").write_text(json.dumps(S, indent=2, default=float) + "\n")
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
