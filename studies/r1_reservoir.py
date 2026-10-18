"""R1(a): finite dark reservoir with mobile ions at the H05 seeded setup. Gates and set-up: studies/r1_common.py.

python studies/r1_reservoir.py list | run KEY | all | summary
Records: studies/r1/res_<key>.json + .npz (one per run, resumable); summary -> studies/r1/reservoir_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import darkspectrax as ds
import r1_common as c

OUT = Path(__file__).resolve().parent / "r1"
VQS, US = (0.01, 0.03), (0.1, 1.0, 10.0)


def eta_of(vq, u):
    return vq / np.sqrt(2 * u)


def cases():
    L = []
    for vq in VQS:
        L += [dict(key=f"pre_vq{vq}_Nn{Nn}_Nx8", kind="pre", vq=vq, Nn=Nn, Nx=8) for Nn in (32, 64)]
        for u in US:
            L += [dict(key=f"sc_vq{vq}_u{u}_Nn{Nn}_Nx{Nx}", kind="sc", vq=vq, u=u, Nn=Nn, Nx=Nx)
                  for Nn, Nx in ((32, 8), (64, 8)) + (((32, 16),) if u == 1.0 else ())]
    return L


def run_case(cs):
    vq, E0 = cs["vq"], cs["vq"] * c.vte
    if cs["kind"] == "pre":
        m = c.mobile_model(cs["Nx"], cs["Nn"], mode="prescribed_drive", E0=E0)
        y0 = c.initial(m)
        eta = 0.0
    else:
        eta = eta_of(vq, cs["u"])
        m = c.mobile_model(cs["Nx"], cs["Nn"], mode="self_consistent", eta=eta, Omega_D=c.w_L)
        y0 = c.initial(m, E_D0=E0 / eta)
    out = c.go(m, y0, c.T_END)
    rec = {"key": cs["key"], **cs, "eta": eta, "E0": E0, "U_D0": float(out["U_D"][0]), "status": out["status"],
           "failure_reason": out["failure_reason"], "t_reached": float(out["t"][-1]), "events": len(out["events"]),
           "ledger_over_Wscale": c.ledger_ok(out), "first_nonpositive": c.first_nonpositive(out),
           "min_K_species": out["K_species"].min(axis=0).tolist(), "min_Tx_species": out["Tx"].min(axis=0).tolist(),
           "wall_time": out["wall"], "parent_commit": ds.PARENT_COMMIT,
           "repository_commit": ds._simulation._git_sha(), "command": f"python studies/r1_reservoir.py run {cs['key']}"}
    if cs["Nn"] == 32 and cs["Nx"] == 8:
        ti, R, S = c.identity_residual(m, out)
        rec["identity_max_rel"] = float(np.abs(R).max() / (c.w_L ** 2 * S.max()))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"res_{cs['key']}.npz", t=out["t"], K_species=out["K_species"], W=out["W"],
                        U_D=out["U_D"], U_gamma=out["U_gamma"], Ebar=out["Fk"][:, 0, 0, 0, 0].real,
                        EDbar=out["Dk"][:, 0, 0, 0, 0].real, ADbar=out["Dk"][:, 6, 0, 0, 0].real, Tx=out["Tx"])
    (OUT / f"res_{cs['key']}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(rec, flush=True)


def load(key):
    return json.loads((OUT / f"res_{key}.json").read_text()), np.load(OUT / f"res_{key}.npz")


def first_below(t, f, thr):
    i = np.flatnonzero(f < thr)
    return float(t[i[0]]) if i.size else None


def summary():
    S = {"gates": {}, "cases": {}}
    ids, ledg, pos = [], [], []
    for vq in VQS:
        E0 = vq * c.vte
        rp32, dp32 = load(f"pre_vq{vq}_Nn32_Nx8")
        rp64, dp64 = load(f"pre_vq{vq}_Nn64_Nx8")
        tp = dp32["t"]
        lin_pre = c.prescribed_linear(tp, E0)
        for u in US:
            key = f"sc_vq{vq}_u{u}"
            r, d = load(f"{key}_Nn32_Nx8")
            r64, d64 = load(f"{key}_Nn64_Nx8")
            eta, U0 = r["eta"], r["U_D0"]
            t = d["t"]
            n = min(t.size, d64["t"].size)
            dK, dK64 = d["K_species"][:n, 0] - d["K_species"][0, 0], d64["K_species"][:n, 0] - d64["K_species"][0, 0]
            pairs = {"Nn32_vs_Nn64": {"dK_e": c.disagreement(t[:n], dK, dK64), "W_D": c.disagreement(t[:n], d["W"][:n, 1], d64["W"][:n, 1]),
                                      "U_D": c.disagreement(t[:n], d["U_D"][:n], d64["U_D"][:n])}}
            ends = [r["t_reached"], r64["t_reached"]]
            if u == 1.0:
                r16, d16 = load(f"{key}_Nn32_Nx16")
                m = min(t.size, d16["t"].size)
                dK16 = d16["K_species"][:m, 0] - d16["K_species"][0, 0]
                pairs["Nx8_vs_Nx16"] = {"dK_e": c.disagreement(t[:m], dK[:m], dK16), "W_D": c.disagreement(t[:m], d["W"][:m, 1], d16["W"][:m, 1]),
                                        "U_D": c.disagreement(t[:m], d["U_D"][:m], d16["U_D"][:m])}
                ends += [r16["t_reached"]]
                ids.append(r16["ledger_over_Wscale"]), pos.append(r16["first_nonpositive"])
            caps = [v if v is not None else float(t[n - 1]) for p in pairs.values() for v in p.values()]
            caps += [c.CERTIFIED[vq]] + ends + [x for x in (r["first_nonpositive"], r64["first_nonpositive"]) if x is not None]
            window = min(caps)
            # linear coupled-oscillator prediction
            X, UDlin = c.coupled_solution(t, eta, E0 / eta)
            Xi = c.coupled_solution_ivp(t, eta, E0 / eta)
            g6 = float(np.abs(X - Xi).max() / np.abs(X).max())
            sel = t <= window
            dev = np.abs(d["U_D"] - UDlin) / U0
            t_dev = first_below(t, -dev, -0.1) if (dev > 0.1).any() else None
            WDlin = U0 - UDlin
            # matched prescribed drive
            ts = np.array([x for x in tp if x <= window])
            Wp = dp32["W"][: ts.size, 2]
            WDk = np.interp(ts, t, d["W"][:, 1])
            bad = (np.abs(WDk - Wp) > 0.1 * np.abs(Wp)) & (ts > 50)
            t_split = float(ts[np.argmax(bad)]) if bad.any() else None
            Wlin_p = lin_pre[: ts.size]
            WDlin_i = np.interp(ts, t, WDlin)
            badl = (np.abs(WDlin_i - Wlin_p) > 0.1 * np.abs(Wlin_p)) & (ts > 50)
            t_split_lin = float(ts[np.argmax(badl)]) if badl.any() else None
            # beat: first minimum of U_D in window
            iw = np.flatnonzero(sel)
            imin = iw[np.argmin(d["U_D"][iw])]
            ilin = np.argmin(UDlin[iw])
            t_pred_min = float(np.pi / eta)
            S["cases"][key] = {
                "vq": vq, "u": u, "eta": eta, "U_D0_over_nTe": U0 / c.vte ** 2, "claim_window": window,
                "caps": {"certified": c.CERTIFIED[vq], "pairs": pairs, "t_reached": ends,
                         "first_nonpositive": [r["first_nonpositive"], r64["first_nonpositive"]]},
                "beat_period_2pi_over_eta": 2 * np.pi / eta, "t_first_min_U_D_pred_pi_over_eta": t_pred_min,
                "U_D_min_over_U_D0_kin_in_window": float(d["U_D"][imin] / U0), "t_min_kin": float(t[imin]),
                "U_D_min_over_U_D0_lin_in_window": float(UDlin[iw[ilin]] / U0), "t_min_lin": float(t[iw[ilin]]),
                "max_dev_U_D_over_U_D0_in_window": float(dev[sel].max()), "t_dev_10pct": t_dev, "G5_pass": bool(dev[sel].max() <= 0.1),
                "W_D_final_over_U_D0": float(np.interp(window, t, d["W"][:, 1]) / U0),
                "W_D_lin_at_window_over_U_D0": float(np.interp(window, t, WDlin) / U0),
                "W_D_over_nTe_at_window": float(np.interp(window, t, d["W"][:, 1]) / c.vte ** 2),
                "W_ext_prescribed_over_nTe_at_window": float(np.interp(window, tp, dp32["W"][:, 2]) / c.vte ** 2),
                "t_split_kin_10pct": t_split, "t_split_lin_10pct": t_split_lin,
                "G7_pass": (t_split is None and t_split_lin is None) or (t_split is not None and t_split_lin is not None and 1 / 1.5 <= t_split / t_split_lin <= 1.5),
                "G6_matrix_vs_ivp": g6, "identity_max_rel": r["identity_max_rel"],
                "ledger": [r["ledger_over_Wscale"], r64["ledger_over_Wscale"]],
                "electron_dK_over_nTe_at_window": float((np.interp(window, t, d["K_species"][:, 0]) - d["K_species"][0, 0]) / c.vte ** 2),
                "ion_dK_over_nTe_at_window": float((np.interp(window, t, d["K_species"][:, 1]) - d["K_species"][0, 1]) / c.vte ** 2),
                "Ebar_max_ratio_to_prescribed": float(np.abs(d["Ebar"][sel]).max() / max(np.abs(dp32["Ebar"][dp32["t"] <= window]).max(), 1e-300)),
            }
            ledg += [r["ledger_over_Wscale"], r64["ledger_over_Wscale"]]
            pos += [r["first_nonpositive"], r64["first_nonpositive"]]
            print(key, {k: S["cases"][key][k] for k in ("claim_window", "G5_pass", "t_dev_10pct", "t_min_kin", "t_min_lin", "G7_pass")}, flush=True)
        # prescribed pair for W_ext
        t = dp32["t"]
        n = min(t.size, dp64["t"].size)
        S["cases"][f"pre_vq{vq}"] = {"W_ext_Nn32_vs_Nn64": c.disagreement(t[:n], dp32["W"][:n, 2], dp64["W"][:n, 2]),
                                     "dK_e_Nn32_vs_Nn64": c.disagreement(t[:n], dp32["K_species"][:n, 0], dp64["K_species"][:n, 0]),
                                     "t_reached": [rp32["t_reached"], rp64["t_reached"]], "ledger": [rp32["ledger_over_Wscale"], rp64["ledger_over_Wscale"]],
                                     "identity_max_rel": rp32["identity_max_rel"], "W_ext_final_over_nTe": float(dp32["W"][-1, 2] / c.vte ** 2),
                                     "W_ext_linear_final_over_nTe": float(lin_pre[-1] / c.vte ** 2)}
        ledg += [rp32["ledger_over_Wscale"], rp64["ledger_over_Wscale"]]
        pos += [rp32["first_nonpositive"], rp64["first_nonpositive"]]
    ident = [v["identity_max_rel"] for v in S["cases"].values() if "identity_max_rel" in v]
    S["gates"] = {"G1_ledger_max": max(ledg), "G1_pass": bool(max(ledg) <= c.LEDGER_TOL),
                  "G2_first_nonpositive_per_run": [x for x in pos if x is not None],
                  "G4_identity_max": max(ident), "G4_pass": bool(max(ident) <= c.ID_TOL),
                  "G5_pass_cases": [k for k, v in S["cases"].items() if v.get("G5_pass")],
                  "G5_fail_cases": [k for k, v in S["cases"].items() if v.get("G5_pass") is False],
                  "G6_max": max(v["G6_matrix_vs_ivp"] for v in S["cases"].values() if "G6_matrix_vs_ivp" in v),
                  "G7_fail_cases": [k for k, v in S["cases"].items() if v.get("G7_pass") is False]}
    (OUT / "reservoir_summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        print("\n".join(cs["key"] for cs in cases()))
    elif cmd == "run":
        run_case(next(cs for cs in cases() if cs["key"] == sys.argv[2]))
    elif cmd == "all":
        for cs in cases():
            if not (OUT / f"res_{cs['key']}.json").exists():
                run_case(cs)
    elif cmd == "summary":
        summary()
