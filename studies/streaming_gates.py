"""stream: exact Hermite free-streaming integrating factor (SPECTRAX #83) on top of the closure Lawson wrapper (#82).

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).
Question. SPECTRAX #82/#83: the Lawson wrapper with the field-scaled closure rate inside the exponent removed the closure
stiffness at H05 v_q/v_te 0.1 (studies/expint: 5.0x fewer steps at c = 64, none at c = 4). #83 additionally exponentiates
the streaming operator exactly per Fourier mode (one real eigh of the Hermite tridiagonal, Gauss-Hermite nodal transform
shared by all k and species; widths/drifts from the state basis frozen at the step start) and composes it with the diagonal
closure rate by a Strang product Phi = A(h/2) B(h) A(h/2) (exact identity; leftover O((h|G|)^2 |r|), see
spectrax/_lawson_streaming.py). Per Runge-Kutta stage it costs two dense Nn x Nn transforms (measured per-step cost
ratio vs closure-only Lawson: ungated, studies/stream_cost.py). The stiffdiag study found the closure (not streaming) to be
the stiff mode (rho(J) = nu_e, real), so the expected outcome is NO step reduction beyond the closure-only Lawson and a
~3-4x higher cost per step; the gates below are written so that this outcome is a recorded failure of the streaming option
at this regime, not a pass.

Cases (same as studies/expint_gates.py): H05, v_q/v_te = 0.1, Nx 64, Nn 128, noise floor 1e-14, rtol 1e-10, atol 1e-14,
pump frame + remaps, closure c in {4, 64}, 100-unit chunks to t = 500 (CPU, 2 threads, nice, JAX_PLATFORMS=cpu).
Runs: streaming `_ls` (studies/stream_batch.sh C); same-session closure-only control `_li2` (TAG=_li2 LAW=--lawson
studies/stream_batch.sh C); the committed `_li` (closure-only, studies/expint) and `_sdcpu` (plain Dopri8 on CPU, ungated
platform control) records exist already.

Gates, per case c in {4, 64}, window t <= 500 (common saves every 0.5):
 S1  steps(_li2, t<=500)/steps(_ls, t<=500) >= 1.5  OR  wall(_li2, t<=500)/wall(_ls, t<=500) >= 1.0 (compile + run; both
     in the same session on the same machine). S1 as a whole passes only if BOTH cases pass.
 S2  accuracy: dK_e(t) and W_ext(t) of `_ls` versus the plain CPU record `_sdcpu`: max |ls - plain| / max |plain| <= 1e-6.
     (Same absolute criterion as expint A; versus the closure-only `_li2` is reported, ungated.)
 S3  ledger: max_t |(K(t) - K(0)) - sum_j W_j(t)| / max_t |W_ext| <= 1e-9 over the whole `_ls` run; status success at t = 500.
 S4  verdict: "streaming earns its place in this regime" iff S1 (both), S2 and S3 (both) hold; anything else is reported
     as not beneficial here, with the numbers.
Ungated: cumulative steps and wall to 500, step trajectory, per-step cost (studies/stream_cost.py), steps/h versus |G| h
(the largest h k a sqrt(2 Nn) reached).

Ordinary B10 (studies/b10.py, M_mi100_ord_j10, nu = 100 constant hypercollision, Nn 32, Nx 40, rtol 1e-10, atol 1e-12):
 Runs: `--streaming` (record _ls: streaming + constant hypercollision in the factor) and a same-session closure-only
 `--lawson` rerun (record _li2 via file copy before the streaming run; see stream_b10.sh).
 B1s steps(_li2)/steps(_ls) >= 1.5 OR wall(_ls) <= wall(_li2).
 B2s gamma_fit within 1e-3 relative of the committed plain record M_mi100_ord_j10.json and its window within 1%.
 B3s max ledger defect / U_field0 <= 1e-9 and Gauss residuals <= 1e-12 A_p, status success (the same B3 that the closure-only
     Lawson failed at 8.4e-9; kept unamended); reported with the closure-only value.
Run: python studies/streaming_gates.py -> studies/stream/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expint_gates import at, case, compare, cumulative, herm, ledger  # noqa: E402

D = Path(__file__).resolve().parent / "stream"
TOL_A, TOL_L = 1e-6, 1e-9
T_ACC = 500.0


def main():
    D.mkdir(exist_ok=True)
    S = {"cases": {}}
    s1, s2, s3 = [], [], []
    for c in (4, 64):
        ls, li2, cp = herm(case(c, "_ls")), herm(case(c, "_li2")), herm(case(c, "_sdcpu"))
        li = herm(case(c, "_li"))
        if ls is None or li2 is None or cp is None:
            S["cases"][c] = {"missing": [n for n, h in (("ls", ls), ("li2", li2), ("sdcpu", cp)) if h is None]}
            s1.append(False), s2.append(False), s3.append(False)
            continue
        r = {}
        sl, wl = at(ls, T_ACC)
        s2_, w2 = at(li2, T_ACC)
        r["steps_ls"], r["wall_ls"], r["steps_li2"], r["wall_li2"] = sl, wl, s2_, w2
        if sl and s2_:
            r["steps_ratio_li2_over_ls"], r["wall_ratio_li2_over_ls"] = s2_ / sl, w2 / wl
            r["S1"] = bool(r["steps_ratio_li2_over_ls"] >= 1.5 or r["wall_ratio_li2_over_ls"] >= 1.0)
        else:
            r["S1"] = False
        if li is not None:
            sli, wli = at(li, T_ACC)
            r["steps_li_committed"], r["wall_li_committed"] = sli, wli
        sp, wp = at(cp, T_ACC)
        r["steps_plain_cpu"], r["wall_plain_cpu"] = sp, wp
        r["A_vs_plain_cpu"] = compare(cp, ls)
        r["S2"] = bool(r["A_vs_plain_cpu"]["dKe"] <= TOL_A and r["A_vs_plain_cpu"]["Wx"] <= TOL_A)
        r["A_ungated_vs_li2"] = compare(li2, ls)
        r["ledger_ls"] = ledger(ls)
        r["ledger_li2"] = ledger(li2)
        r["status_ls"] = cumulative(ls)[3][-1]
        r["t_end_ls"] = float(ls["t"][-1])
        r["S3"] = bool(r["ledger_ls"] <= TOL_L and r["status_ls"] == "success" and r["t_end_ls"] >= T_ACC - 1e-6)
        s1.append(r["S1"]), s2.append(r["S2"]), s3.append(r["S3"])
        S["cases"][c] = r
    S["S1_all"], S["S2_all"], S["S3_all"] = bool(all(s1)), bool(all(s2)), bool(all(s3))
    S["S4_streaming_earns_its_place"] = bool(all(s1) and all(s2) and all(s3))
    B = Path(__file__).resolve().parent / "b10"
    rec, ls, li2 = B / "M_mi100_ord_j10.json", B / "M_mi100_ord_j10_ls.json", B / "M_mi100_ord_j10_li2.json"
    if rec.exists() and ls.exists() and li2.exists():
        a, b, c2 = (json.loads(p.read_text()) for p in (rec, ls, li2))
        bb = {"steps_ls": b["steps"], "steps_li2": c2["steps"], "steps_ratio_li2_over_ls": c2["steps"] / b["steps"],
              "wall_ls": b["wall_time"], "wall_li2": c2["wall_time"]}
        ga, gb = a["results"][0], b["results"][0]
        bb["gamma_fit_record"], bb["gamma_fit_ls"] = ga["gamma_fit"], gb["gamma_fit"]
        if ga["gamma_fit"] and gb["gamma_fit"]:
            bb["gamma_rel_diff"] = abs(gb["gamma_fit"] - ga["gamma_fit"]) / abs(ga["gamma_fit"])
            wa, wb_ = np.array(ga["window"]), np.array(gb["window"])
            bb["window_rel_diff"] = float(np.abs(wb_ - wa).max() / wa.max())
        bb["ledger_over_U0_ls"], bb["ledger_over_U0_li2"] = b["max_ledger_over_U0"], c2["max_ledger_over_U0"]
        bb["gauss_over_Ap"] = b["max_gauss_over_Ap"]
        bb["B1s"] = bool(bb["steps_ratio_li2_over_ls"] >= 1.5 or bb["wall_ls"] <= bb["wall_li2"])
        bb["B2s"] = bool(bb.get("gamma_rel_diff", 1) <= 1e-3 and bb.get("window_rel_diff", 1) <= 1e-2)
        bb["B3s"] = bool(b["max_ledger_over_U0"] <= 1e-9 and max(b["max_gauss_over_Ap"]) <= 1e-12 and b["status"] == "success")
        S["b10"] = bb
    (D / "summary.json").write_text(json.dumps(S, indent=2, default=float) + "\n")
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
