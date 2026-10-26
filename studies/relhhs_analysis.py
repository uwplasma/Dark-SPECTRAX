"""relhhs analysis: measurements of the relativistic Hermite runs (studies/relhhs/runs) with the gates of relhhs_gates.py -> studies/relhhs/summary.json.

Per run: collapse time (first t > 50 with |W_ext - W_lin| > 0.1 |W_lin|, W_lin the nonrelativistic homogeneous ODE, as relgrid_analysis), mean-field energy over
the linear law, dE1D_e (HHS measure, trailing 200 average) and dK_e over E_th = 5e-4 at fixed times, certified window t_cert (relhhs_gates), end state.
References are read from the committed records: relgrid (rel det vq 0.03 and shot-noise variants), relpic (relativistic and Newtonian PIC, digitized HHS Fig. 2).
Run: python studies/relhhs_analysis.py (after python studies/relhhs_gates.py prod)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relgrid_analysis as RA  # noqa: E402
import relgrid_ref as X  # noqa: E402
import relhhs_gates as G  # noqa: E402

HERE = Path(__file__).resolve().parent
TQ = (200, 300, 400, 500, 600, 800, 1000)


def at(t, y, tq):
    i = int(np.argmin(np.abs(t - tq)))
    return float(y[i]) if abs(t[i] - tq) < 1.0 else None


def run_row(r, prod):
    t = r["t"]
    lin = X.homogeneous(0.03, float(t[-1]) + 1, rel=False)
    yl = lin.sol(t)
    dE = RA.mavg(t, r["dE1D_e"]) / G.E_TH
    dK = RA.mavg(t, r["dK_e"]) / G.E_TH
    fld = RA.mavg(t, 0.5 * r["Ebar"] ** 2)
    fl = RA.mavg(t, 0.5 * yl[2] ** 2)
    d10 = np.nonzero((t > 50) & (np.abs(r["W_ext"] - yl[3]) > 0.1 * np.abs(yl[3])))[0]
    return {"T": float(t[-1]), "status": r["status"], "t_collapse_W_10pc": float(t[d10[0]]) if d10.size else None,
            "dE1D_e_over_Eth": {str(q): at(t, dE, q) for q in TQ}, "dK_e_over_Eth_avg": {str(q): at(t, dK, q) for q in TQ},
            "Ebar_energy_over_linear": {str(q): (at(t, fld, q) / at(t, fl, q) if at(t, fl, q) else None) for q in TQ},
            "max_dE1D_over_Eth": float(dE.max()), "dE1D_at_end": float(dE[-1]),
            "max_top_frac_e": float(r["top_frac"][:, 0].max()), "t_cert": prod["runs"].get(str(prod["key"]), {}).get("t_cert")}


def main():
    prod = json.loads((HERE / "relhhs" / "gates_prod.json").read_text())
    out = {"repository_commit": G.sha(), "E_th": G.E_TH, "runs": {}, "references": {}}
    for nx in G.NX:
        for nn in G.NN:
            for c in G.CC:
                r = G.load(G.case(nx, nn, c))
                if r is None:
                    continue
                prod["key"] = (nx, nn, c)
                out["runs"][str((nx, nn, c))] = run_row(r, prod)
    rg = json.loads((HERE / "relgrid" / "summary.json").read_text())["cases"]
    out["references"]["relgrid"] = {k: {"t_W_10pc_from_linear": rg[k]["t_W_10pc_from_linear"], "dE1D_over_Eth": rg[k]["dE1D_over_Eth"],
                                        "t_fneg": rg[k]["t_fneg_1e-3"], "t_edge": rg[k]["t_edge_1e-6"], "t_ledger": rg[k]["t_ledger_1e-4"]}
                                    for k in rg if "vq0.03" in k and "0.001" not in k}
    rp = json.loads((HERE / "relpic" / "summary.json").read_text())
    obs = rp["observables"]
    out["references"]["relpic_dW_th_over_Eth"] = {k: {str(int(t)): v[0] / G.E_TH for t, v in zip(rp["times"], obs[k]["dW_th"]) if v is not None} for k in obs if "dW_th" in obs[k]}
    out["references"]["hhs_digitized_dW_th_over_Eth"] = {str(int(x["t"])): x["hhs_digitized_dW_th"] / G.E_TH for x in rp["HHS_fig2_digitized"]}
    (HERE / "relhhs" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["runs"].items():
        print(k, "T", v["T"], v["status"], "collapse", v["t_collapse_W_10pc"], "t_cert", v["t_cert"], "dE1D", v["dE1D_e_over_Eth"], "Ebar/lin", v["Ebar_energy_over_linear"])
    print(json.dumps(out["references"], indent=1)[:3000])


if __name__ == "__main__":
    main()
