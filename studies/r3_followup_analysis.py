"""R3 follow-up analysis: applies the rules of r3_followup_gates.py (FU1-FU4) to the records. Run: python studies/r3_followup_analysis.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_gates as g  # noqa: E402
import r3_ref as ref  # noqa: E402
import r3_followup_gates as fg  # noqa: E402
from r3_analysis import kin_name, ref_name  # noqa: E402

D = Path(__file__).resolve().parent / "r3"


def load(name):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else None


def remnant(ch):
    return 1 - (ch["refl_dark"] + ch["wrap_refl_dark"]) - (ch["refl_photon"] + ch["wrap_refl_photon"]) - ch["trans_dark"] \
        - ch["trans_photon"] - ch["absorbed"]


def vals(ch):
    return {"refl_dark": ch["refl_dark"] + ch["wrap_refl_dark"], "refl_photon": ch["refl_photon"] + ch["wrap_refl_photon"],
            "trans_dark": ch["trans_dark"], "remnant": remnant(ch)}


def main():
    S = {"gates": fg.GATE, "ladder": {}}
    for label, (th, pol) in (("longitudinal_B0", (None, "x")), ("th45_y", (45, "y"))):
        rows = []
        for nx in fg.NX:
            k = load(kin_name(0.3, 4.0, th, pol, Nx=nx, Nn=4))
            c = load(ref_name(0.3, 4.0, th, pol, nt=nx))
            if not k or not c:
                continue
            kv, cv = vals(k["channels"]), vals(c["channels"])
            rows.append({"Nx": nx, "status": k["status"], "kin": kv, "cold": cv, "diff": {q: kv[q] - cv[q] for q in kv},
                         "stored_kin": k["channels"]["stored"], "stored_cold": c["channels"]["stored"],
                         "left_back_in_mid_kin": k["channels"]["left_back_in_mid"], "wrap_right_kin": k["channels"]["wrap_right"],
                         "absorbed": k["channels"]["absorbed"], "flux_residual": k["budget"]["residual"],
                         "run_time": k["run_time"], "closure": k["channels"]["closure"]})
        S["ladder"][label] = {"rows": rows}
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            ch = {q: abs(a["kin"][q] - b["kin"][q]) for q in a["kin"]}
            S["ladder"][label]["FU1"] = {"between": [a["Nx"], b["Nx"]], "changes": ch, "max": max(ch.values()),
                                         "converged": max(ch.values()) <= fg.GATE["FU1"]}
            d = b["diff"]
            S["ladder"][label]["FU2"] = {"Nx": b["Nx"], "diff": d,
                                         "pass": all(abs(d[q]) <= fg.GATE["FU2"] for q in ("refl_dark", "refl_photon", "trans_dark"))
                                         and abs(d["remnant"]) <= fg.GATE["FU2_remnant"]}
        # FU3 thermal
        th_rows = {}
        for v in (0.005, 0.02):
            pat = f"kin_eta0.3_w4_{'B0' if th is None else 'th%d' % th}_{pol}_Nx3072_Nn4_vte{v:g}.json"
            k = load(pat)
            if k:
                th_rows[v] = vals(k["channels"])
        k = load(kin_name(0.3, 4.0, th, pol, Nx=3072, Nn=4))
        if k:
            th_rows[0.01] = vals(k["channels"])
        S["ladder"][label]["FU3"] = {str(v): r for v, r in th_rows.items()}
    # FU4 refined law
    rows = []
    for p in sorted(D.glob("ref_*N4096.json")):
        r = json.loads(p.read_text())
        c = r["case"]
        if c["ntrunc"] or c["pol"] not in ("y",) or not (c["theta"] is None or c["theta"] == 0):
            continue
        case = ref.Case(c["eta"], c["w"], c["theta"], c["pol"])
        lz = ref.lz_prediction(case)
        rf = ref.refined_prediction(case)
        ch = r["channels"]
        wc = {"trans_dark": ch["trans_dark"], "refl_photon": ch["refl_photon"] + ch["wrap_refl_photon"], "refl_dark": ch["refl_dark"] + ch["wrap_refl_dark"]}
        p0 = lz["parts"][0]
        rows.append({"case": c["name"], "eta": c["eta"], "w": c["w"], "theta": c["theta"], "kbarLc": p0["kbar"] * p0["Lc"],
                     "eta_fnc_over_kbar2": c["eta"] * p0["f"] * p0["n_c"] / p0["kbar"] ** 2,
                     "ref": wc, "simple": {q: lz[q] for q in wc}, "refined": rf,
                     "dev_simple": {q: wc[q] - lz[q] for q in wc}, "dev_refined": {q: wc[q] - rf[q] for q in wc},
                     "exponent_ratio": float(np.log(ch["trans_dark"]) / np.log(lz["trans_dark"])),
                     "gated": p0["kbar"] * p0["Lc"] >= 3})
    for r in rows:
        r["max_dev_refined"] = max(abs(x) for x in r["dev_refined"].values())
        r["max_dev_simple"] = max(abs(x) for x in r["dev_simple"].values())
    S["FU4"] = {"rows": rows, "pass": all(r["max_dev_refined"] <= 0.02 for r in rows if r["gated"]) if rows else None,
                "max_dev_refined_gated": max((r["max_dev_refined"] for r in rows if r["gated"]), default=None),
                "max_dev_refined_all": max((r["max_dev_refined"] for r in rows), default=None)}
    (D / "followup_summary.json").write_text(json.dumps(S, indent=1) + "\n")
    for label, v in S["ladder"].items():
        print(label)
        for r in v["rows"]:
            print("  Nx", r["Nx"], {q: round(x, 4) for q, x in r["kin"].items()}, "cold", {q: round(x, 4) for q, x in r["cold"].items()}, "t", round(r["run_time"]))
        print("  FU1", v.get("FU1"), "\n  FU2", v.get("FU2"), "\n  FU3", v.get("FU3"))
    print("FU4 pass:", S["FU4"]["pass"], "max dev refined (gated/all):", S["FU4"]["max_dev_refined_gated"], S["FU4"]["max_dev_refined_all"])
    for r in rows:
        print(f"  {r['case']:22s} g={r['gated']!s:5} kLc={r['kbarLc']:.1f} s={r['eta_fnc_over_kbar2']:.2f} simple dev {r['max_dev_simple']:.3f} refined dev {r['max_dev_refined']:.3f} ratio {r['exponent_ratio']:.3f}")


if __name__ == "__main__":
    main()
