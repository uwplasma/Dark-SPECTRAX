"""R3 analysis: applies exactly the rules declared in r3_gates.py to the records in studies/r3/ and writes
studies/r3/summary.json. No rule is changed here; extra reported (ungated) columns are labelled as such.
Run: python studies/r3_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_gates as g  # noqa: E402

D = Path(__file__).resolve().parent / "r3"
CH = ("refl_dark", "refl_photon", "trans_dark", "trans_photon", "stored", "beyond")


def load(pattern):
    return {p.name: json.loads(p.read_text()) for p in sorted(D.glob(pattern))}


def ref_name(eta, w, theta, pol, N=4096, nt=0):
    b = "B0" if theta is None else f"th{theta}"
    return f"ref_eta{eta:g}_w{w:g}_{b}_{pol}" + (f"_trunc{nt}" if nt else "") + f"_N{N}.json"


def kin_name(eta, w, theta, pol, Nx=512, Nn=4, extra=""):
    b = "B0" if theta is None else f"th{theta}"
    return f"kin_eta{eta:g}_w{w:g}_{b}_{pol}_Nx{Nx}_Nn{Nn}{extra}.json"


def wrapfix(ch):
    """Declared channels plus the wrap-corrected reflected totals (reported, not gated)."""
    return {"refl_dark_wrapcorr": ch["refl_dark"] + ch["wrap_refl_dark"],
            "refl_photon_wrapcorr": ch["refl_photon"] + ch["wrap_refl_photon"]}


def main():
    ref, kin = load("ref_*.json"), load("kin_*.json")
    S = {"gates": g.GATE}
    # ---------------------------------------------------------------- reference
    main_ref = {k: v for k, v in ref.items() if v["N"] == 4096 and not v["case"]["ntrunc"] and v["case"]["eta"] in g.ETAS}
    S["R1"] = {"max_energy_drift": max(v["energy_drift"] for v in main_ref.values()), "n_cases": len(main_ref)}
    S["R1"]["pass"] = S["R1"]["max_energy_drift"] <= g.GATE["R1"]
    # R2 refinement
    r2 = []
    for (th, pol) in ((None, "y"), (None, "x"), (45, "y")):
        a, b = ref.get(ref_name(0.3, 4.0, th, pol)), ref.get(ref_name(0.3, 4.0, th, pol, N=6144))
        if a and b:
            d = max(abs(a["channels"][k] - b["channels"][k]) for k in a["channels"] if k != "closure")
            r2.append({"case": a["case"]["name"], "max_channel_change": d})
    S["R2"] = {"cases": r2, "pass": bool(r2) and all(x["max_channel_change"] <= g.GATE["R2"] for x in r2)}
    # R3 flux budget
    res = {k: abs(v["budget"]["residual"]) for k, v in main_ref.items()}
    om = [(k, abs(v["budget"]["residual_without_phiA"])) for k, v in main_ref.items()
          if v["case"]["theta"] is None and v["case"]["pol"] == "x"]
    S["R3"] = {"max_residual_with_phiA": max(res.values()), "omission_residuals_B0_x": dict(om),
               "pass_with": max(res.values()) <= g.GATE["R3"],
               "pass_omission": bool(om) and min(x[1] for x in om) > g.GATE["R3_omit"]}
    # R4 Landau-Zener
    r4 = []
    for k, v in sorted(main_ref.items()):
        lz = v["analytic"].get("lz")
        if not lz:
            continue
        ch = v["channels"]
        wc = wrapfix(ch)
        dev = {"trans_dark": ch["trans_dark"] - lz["trans_dark"], "refl_photon": ch["refl_photon"] - lz["refl_photon"],
               "refl_dark": ch["refl_dark"] - lz["refl_dark"]}
        dev_wc = {"refl_photon": wc["refl_photon_wrapcorr"] - lz["refl_photon"], "refl_dark": wc["refl_dark_wrapcorr"] - lz["refl_dark"]}
        p0 = lz["parts"][0]
        r4.append({"case": v["case"]["name"], "eta": v["case"]["eta"], "w": v["case"]["w"], "theta": v["case"]["theta"],
                   "valid": lz["valid"], "Lc": p0["Lc"], "kbar": p0["kbar"], "kbarLc": p0["Lc"] * p0["kbar"],
                   "eta_fnc_over_kbar2": v["case"]["eta"] * p0["f"] * p0["n_c"] / p0["kbar"] ** 2,
                   "lz": {kk: lz[kk] for kk in ("trans_dark", "refl_photon", "refl_dark")},
                   "ref": {kk: ch[kk] for kk in ("trans_dark", "refl_photon", "refl_dark")}, "ref_wrapcorr": wc,
                   "dev": dev, "dev_wrapcorr": dev_wc, "max_abs_dev": max(abs(x) for x in dev.values()),
                   "pass": max(abs(x) for x in dev.values()) <= g.GATE["R4"]})
    S["R4"] = {"rows": r4, "n_valid": sum(x["valid"] for x in r4),
               "pass_in_V": all(x["pass"] for x in r4 if x["valid"]) if any(x["valid"] for x in r4) else None}
    # R5 longitudinal
    r5 = []
    for k, v in sorted(main_ref.items()):
        wk = v["analytic"].get("wkb")
        if not wk:
            continue
        ch = v["channels"]
        wc = wrapfix(ch)
        row = {"case": v["case"]["name"], "eta": v["case"]["eta"], "w": v["case"]["w"], "T_wkb": wk["T_wkb"],
               "kref_Lc": wk["kref_Lc"], "stored": ch["stored"], "refl_dark": ch["refl_dark"],
               "refl_dark_wrapcorr": wc["refl_dark_wrapcorr"], "trans_dark": ch["trans_dark"], "trans_photon": ch["trans_photon"],
               "refl_photon": ch["refl_photon"], "beyond": ch["beyond"]}
        row["gated"] = wk["kref_Lc"] >= 3
        row["dev_stored"] = ch["stored"] - wk["T_wkb"]
        row["dev_refl_dark"] = ch["refl_dark"] - (1 - wk["T_wkb"])
        row["pass"] = abs(row["dev_stored"]) <= g.GATE["R5"] and abs(row["dev_refl_dark"]) <= g.GATE["R5"]
        r5.append(row)
    S["R5"] = {"rows": r5, "pass_gated": all(x["pass"] for x in r5 if x["gated"]) if any(x["gated"] for x in r5) else None}
    # R6 B0 effects (reported)
    r6 = []
    for k, v in sorted(main_ref.items()):
        c = v["case"]
        if c["theta"] is None or c["eta"] != 0.3:
            continue
        base = main_ref.get(ref_name(0.3, c["w"], None, "y" if c["pol"] in ("y", "z") else "x"))
        if not base:
            continue
        r6.append({"case": c["name"], "theta": c["theta"], "pol": c["pol"], "w": c["w"],
                   "delta_vs_B0": {kk: v["channels"][kk] - base["channels"][kk] for kk in CH},
                   "channels": {kk: v["channels"][kk] for kk in CH}, "wrap": wrapfix(v["channels"])})
    S["R6"] = r6
    # ref table (all main cases)
    S["ref_table"] = [{"case": v["case"]["name"], "eta": v["case"]["eta"], "w": v["case"]["w"], "theta": v["case"]["theta"],
                       "pol": v["case"]["pol"], **{kk: v["channels"][kk] for kk in CH}, **wrapfix(v["channels"]),
                       "static": v["channels"]["static_left"] + v["channels"]["static_mid"]} for v in main_ref.values()]
    # ---------------------------------------------------------------- kinetic
    k1 = kin.get("kin_eta0.3_w4_B0_y_Nx512_Nn4_nopacket.json")
    S["K1"] = None if not k1 else {"max_dn_over_nhi": k1["max_dn_over_nhi"], "max_field_final": k1["max_abs_E_dark_final"],
                                   "E0_max": k1["background"]["E0_max"], "dark_near_Ex_max": k1["background"]["dark_near_Ex_max"],
                                   "balance_residual_max": k1["background"]["balance_residual_max"],
                                   "pass": k1["max_dn_over_nhi"] <= g.GATE["K1_n"] and k1["max_abs_E_dark_final"] <= g.GATE["K1_field"] * g.AMP}
    runs = {k: v for k, v in kin.items() if "channels" in v and v["amp_scale"] == 1 and not v["nopacket"] and v["vte"] == g.VTE
            and v["Nx"] in (512, 768)}
    S["K2"] = {"all_success": all(v["status"] == "success" for v in runs.values()),
               "max_ledger_rel_E_in": max(max(v["ledger_rel_E_in"]) for v in runs.values()), "n_runs": len(runs)}
    S["K2"]["pass"] = S["K2"]["all_success"] and S["K2"]["max_ledger_rel_E_in"] <= g.GATE["K2"]
    rows, k5 = [], []
    for k, v in sorted(runs.items()):
        c = v["case"]
        Nx, Nn = v["Nx"], v["Nn"]
        twin = ref.get(ref_name(c["eta"], c["w"], c["theta"], c["pol"], nt=Nx))
        ch = v["channels"]
        row = {"run": k, "Nx": Nx, "Nn": Nn, "eta": c["eta"], "w": c["w"], "theta": c["theta"], "pol": c["pol"],
               "closure": ch["closure"], "flux_residual": v["budget"]["residual"],
               "flux_residual_without_phiA": v["budget"]["residual_without_phiA"], "absorbed": ch["absorbed"],
               **{kk: ch[kk] for kk in CH}, **wrapfix(ch), "static": ch["static_left"] + ch["static_mid"]}
        if twin:
            tc = twin["channels"]
            diffs = {kk: ch[kk] - tc[kk] for kk in CH}
            diffs["stored_plus_absorbed"] = ch["stored"] + ch["absorbed"] - tc["stored"]
            row["vs_cold_twin"] = diffs
            lim = g.GATE["K4_L"] if c["pol"] == "x" and c["theta"] is None else (g.GATE["K4_T"] if c["theta"] is None else g.GATE["K4_B"])
            keys = ("refl_dark", "stored_plus_absorbed") if lim == g.GATE["K4_L"] else CH
            row["K4_max"] = max(abs(diffs[kk]) for kk in keys)
            row["K4_limit"] = lim
            row["K4_pass"] = row["K4_max"] <= lim
        rows.append(row)
    S["kin_rows"] = rows
    for r in rows:
        if r["Nx"] == 768:
            base = next((x for x in rows if x["Nx"] == 512 and (x["eta"], x["w"], x["theta"], x["pol"]) == (r["eta"], r["w"], r["theta"], r["pol"])), None)
            if base:
                d = max(abs(r[kk] - base[kk]) for kk in CH)
                k5.append({"case": f"eta{r['eta']:g}_w{r['w']:g}_{r['theta']}_{r['pol']}", "max_channel_change": d,
                           "pass": d <= g.GATE["K5"]})
    S["K5"] = k5
    S["K3"] = {"max_closure": max((r["closure"] for r in rows), default=None),
               "max_flux_residual_512": max((abs(r["flux_residual"]) for r in rows if r["Nx"] == 512), default=None),
               "max_flux_residual_768": max((abs(r["flux_residual"]) for r in rows if r["Nx"] == 768), default=None)}
    k6a, k6b = kin.get(kin_name(0.3, 4.0, None, "y")), kin.get(kin_name(0.3, 4.0, None, "y", extra="_amp3"))
    if k6a and k6b:
        d = max(abs(k6a["channels"][kk] - k6b["channels"][kk]) for kk in CH)
        S["K6"] = {"max_channel_change": d, "pass": d <= g.GATE["K6"]}
    (D / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({k: S[k] for k in ("R1", "R2", "R3", "K1", "K2", "K3", "K5", "K6") if k in S}, indent=1))
    print("R4 valid:", S["R4"]["n_valid"], "pass_in_V:", S["R4"]["pass_in_V"], " R5 pass:", S["R5"]["pass_gated"])


if __name__ == "__main__":
    main()
