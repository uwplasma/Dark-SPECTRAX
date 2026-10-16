"""Gates for the SW Hermite prototype (declared before the runs, see docs/design/l2-stable-basis.md):
  G3: H05 v_q 0.1, nu = 0, Nx 16/32: dK_e and W_ext agree with the masked grid reference within 10% (running-max rule
      of studies/nxconv_analysis.py, t > 50) to t >= 700, and the Nn and Nx refinement pairs agree to t >= 700.
  G4: energy-ledger drift |dK_e + dK_i + U_k + U_0 - W_ext| <= 1e-3 x running max W_ext for 50 < t <= 600.
Rejection: G3 failing before t = 650 or G4 failing.  Grid reference = g_vq0.1_Nx64_Nv16384_dt0.02_osc_v64_spectral_xmask23
(certified for dK_e, W_ext to t ~ 750, spectrum to ~640); the Nx 32 masked wide-box grid is also listed.  AW cost from
studies/nxconv/runs (c = 4 closure runs).
Run: python studies/sw/sw_analysis.py -> studies/sw/summary.json (+ markdown tables on stdout)
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent))
from nxconv_analysis import herm, spec_res, t_res  # noqa: E402

G = D.parent / "nxconv" / "grid"
GRIDS = ["g_vq0.1_Nx64_Nv16384_dt0.02_osc_v64_spectral_xmask23", "g_vq0.1_Nx32_Nv16384_dt0.02_osc_v64_spectral_xmask23"]


def load(p):
    d = dict(np.load(p))
    d["Wx"], d["dKe"] = d["W_ext"], d["dK_e"]
    return d


def num(s):
    return float(str(s).lstrip(">=").split()[0]) if s is not None else -1.0


def main():
    sw = {p.stem: load(p) for p in sorted((D / "runs").glob("sw_*_T1000.npz"))}
    grids = {g: load(G / f"{g}.npz") for g in GRIDS}
    out = {"vs_grid": [], "pairs": [], "ledger": [], "cost": []}
    print("| SW run | vs grid | dK_e | W_ext | field spectrum |\n|---|---|---|---|---|")
    for name, d in sw.items():
        for g, gd in grids.items():
            nx = int(name.split("_Nx")[1].split("_")[0])
            row = {"run": name, "grid": g, "dK_e": t_res(gd, d, "dKe"), "W_ext": t_res(gd, d, "Wx"),
                   "spectrum": spec_res(gd, d, (nx - 1) // 3)}
            out["vs_grid"].append(row)
            print(f"| {name} | {g.split('_dt')[0]} | {row['dK_e']} | {row['W_ext']} | {row['spectrum']} |")
    pairs = [("Nx32_Nn128", "Nx32_Nn64"), ("Nx32_Nn256", "Nx32_Nn128"), ("Nx16_Nn128", "Nx16_Nn64"),
             ("Nx32_Nn64", "Nx16_Nn64"), ("Nx32_Nn128", "Nx16_Nn128")]
    print("\n| SW pair (reference first) | dK_e | W_ext | field spectrum |\n|---|---|---|---|")
    for a, b in pairs:
        na, nb = f"sw_vq0.1_{a}_dt0.02_T1000", f"sw_vq0.1_{b}_dt0.02_T1000"
        if na in sw and nb in sw:
            nx = min(int(a.split("Nx")[1].split("_")[0]), int(b.split("Nx")[1].split("_")[0]))
            row = {"a": na, "b": nb, "dK_e": t_res(sw[na], sw[nb], "dKe"), "W_ext": t_res(sw[na], sw[nb], "Wx"),
                   "spectrum": spec_res(sw[na], sw[nb], (nx - 1) // 3)}
            out["pairs"].append(row)
            print(f"| {a} vs {b} | {row['dK_e']} | {row['W_ext']} | {row['spectrum']} |")
    print("\n| SW run | max ledger/W (50<t<=600) | first ledger/W > 1e-3 | max mass drift e | max momentum drift (t<=600) "
          "| max L2 drift | max top-quarter frac | wall s | RHS evals |\n|---|---|---|---|---|---|---|---|---|")
    for name in sw:
        r = json.loads((D / "runs" / f"{name}.json").read_text())
        out["ledger"].append(r)
        print(f"| {name} | {r['max_energy_defect_over_runmaxW_t<=600']:.1e} | {r['first_energy_defect_over_W_gt_1e-3']} "
              f"| {r['max_mass_drift_e']:.1e} | {r['max_momentum_drift_t<=600']:.1e} | {r['max_L2_drift_e']:.1e} "
              f"| {r['max_top_frac']:.1e} | {r['wall_time']:.0f} | {r['rhs_evals']} |")
    for case in ("vq0.1_Nx32_Nn64_c4_nf1e-14", "vq0.1_Nx64_Nn64_c4_nf1e-14"):
        h = herm(case)
        if h is not None:
            out["cost"].append({"case": case, "T": float(h["t"][-1]),
                                "steps": sum(r["steps"] for r in h["recs"]),
                                "rejected": sum(r["rejected"] for r in h["recs"]),
                                "run_time": sum(r["run_time"] for r in h["recs"]),
                                "wall_time": sum(r["wall_time"] for r in h["recs"])})
    print("\nAW c=4 cost:", json.dumps(out["cost"]))
    # gates on the primary grid reference
    ref = GRIDS[0]
    g3 = {}
    for name in sw:
        row = next(r for r in out["vs_grid"] if r["run"] == name and r["grid"] == ref)
        g3[name] = min(num(row["dK_e"]), num(row["W_ext"]))
    pair_min = min([min(num(p["dK_e"]), num(p["W_ext"])) for p in out["pairs"]] or [-1])
    g4 = {r["case"]: r["max_energy_defect_over_runmaxW_t<=600"] for r in out["ledger"]}
    out["gates"] = {"G3_t_vs_grid": g3, "G3_pairs_min": pair_min,
                    "G3_pass": bool(g3) and min(g3.values()) >= 700 and pair_min >= 700,
                    "G3_reject_before_650": bool(g3) and min(g3.values()) < 650,
                    "G4_max": g4, "G4_pass": bool(g4) and max(g4.values()) <= 1e-3}
    print(json.dumps(out["gates"], indent=1))
    (D / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
