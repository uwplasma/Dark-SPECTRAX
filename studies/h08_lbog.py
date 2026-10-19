"""H08 with the Francisquez LBO-G cross-species form (collision_model = "dougherty_g", beta = 0; parent #73).

New study, gates declared before running. Same setup as studies/h08.py (it is imported; only the collision model differs):
nu_dougherty = [[nu_ei, nu_ei], [0, nu_ei sqrt(m_e/m_i)]], H05 nxconv setup, Nx 32, Nn 64, restart chunk 240 (nxconv's, so the
nu = 0 record needs no re-attribution), T = 1000, v_q/v_te in {0.01, 0.03}, nu_ei in {1e-3, 1e-2}, plus unseeded homogeneous
controls (Nx 8). LBO-G with beta = 0 matches both Morse rates (momentum friction and temperature exchange); the H08 records
(studies/h08/runs, LBO-EM) have the same momentum friction but 2x Morse temperature exchange, scaled by m_e/m_i.
Prediction (not a gate): electron-ion temperature exchange halves; the effect on W_ext, dK_e, T_e, T_i is small.

Gates (declared before running; failures recorded, never amended):
G0 homogeneous controls vs the exact uniform driven two-fluid oscillator with drag (same ODE as H08 G0: the uniform
   momentum drag is identical in LBO-G and LBO-EM): max error <= 1e-6 of max |exact| for Ebar and W_ext, t <= 1000.
G2 every run: max |P_e + P_i| <= 1e-9 max |P_e| and max |ledger defect| <= 1e-6 max |W_ext|.
G3 every run: mean-field identity residual with the collisional current term max |R| <= 1e-8 omega_L^2 E0.
G4 small effect vs the LBO-EM H08 records of the same case (Nx 32, Nn 64), cycle averages at t in {250, 500, 750, 950} (and
   1000 at v_q 0.01): |relative change| <= 2e-2 in each of W_ext, dK_e, T_e, T_i.
Reported, not gated: the same relative changes in dK_e/W_ext and the electron-ion exchange Q_e = int P_e dt (and Q_i),
and the maximum relative change over the window. Claim window = the H08 window for the same case (1000 at v_q 0.01,
950 at 0.03); no Nx/Nn refinement pairs are run here (H08 pairs for EM apply to the same grid; LBO-G is not re-certified).
The EM records were made with restart chunk 200, these with 240: a 1e-5-level trajectory difference is expected from that
alone (studies/h08_g1_cause.py), far below G4's 2e-2.

Run: python studies/h08_lbog.py run  (all cases, resumable)    python studies/h08_lbog.py analyze -> studies/h08_lbog/summary.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h08  # noqa: E402
import darkspectrax as ds  # noqa: E402
from darkspectrax import _simulation as _sim  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "h08_lbog"
EM_RUNS = HERE / "h08" / "runs"
CHUNK = 240
CASES = [(vq, nu) for vq in (0.01, 0.03) for nu in (1e-3, 1e-2)]
TQ = (250.0, 500.0, 750.0, 950.0, 1000.0)


def setup():
    h08.COLL, h08.RUNS = "dougherty_g", D / "runs"


def run_all():
    for vq, nu in CASES:
        for extra in ([["--Nx", "8", "--control"], ["--Nx", "32"]]):
            for _ in range(8):  # one process per chunk
                cmd = ["--vq", str(vq), "--nu", str(nu), "--chunk", str(CHUNK)] + extra
                subprocess.run([sys.executable, "-c", "import sys, h08; from pathlib import Path; h08.COLL = 'dougherty_g'; "
                                "h08.RUNS = Path(sys.argv[1]); h08.main(['run'] + sys.argv[2:])", str(D / "runs")] + cmd,
                               cwd=HERE, check=True)
                nx = 8 if "--control" in extra else 32
                c = h08.case_name(vq, nu, nx, 64, "--control" in extra)
                parts = sorted((D / "runs").glob(c + ".part*.json"), key=lambda p: int(p.suffixes[-2][5:]))
                if json.loads(parts[-1].read_text())["t_end"] >= 1000 - 1e-9:
                    break


def analyze():
    out = {"gates": {}, "controls": {}, "runs": {}, "vs_em": {}}
    g0, g2, g3, g4 = [], [], [], []
    for vq, nu in CASES:
        setup()
        hc = h08.load(h08.case_name(vq, nu, 8, 64, True))
        Eb, W = h08.exact_control(vq, nu, hc["t"])
        r = {"t_end": float(hc["t"][-1]), "Ebar_err": float(np.abs(hc["Ebar"] - Eb).max() / np.abs(Eb).max()),
             "Wext_err": float(np.abs(hc["Wx"] - W).max() / np.abs(W).max())}
        out["controls"][f"vq{vq:g}_nu{nu:g}"] = r
        g0.append(r["Ebar_err"] <= 1e-6 and r["Wext_err"] <= 1e-6 and r["t_end"] >= 1000 - 1e-9)
        c = h08.case_name(vq, nu, 32, 64, False)
        h = h08.load(c)
        h["dKi"] = h["K_species"][:, 1] - h["K_species"][0, 1]
        h["Qi"] = h["Qcum"][:, 1]
        h08.RUNS = EM_RUNS
        e = h08.load(c)
        e["dKi"] = e["K_species"][:, 1] - e["K_species"][0, 1]
        e["Qi"] = e["Qcum"][:, 1]
        E0 = vq * h08.vte
        rr = {"t_end": float(h["t"][-1]), "status": h["recs"][-1]["status"], "steps": sum(x["steps"] for x in h["recs"]),
              "max_R_over_wL2E0": float(np.abs(h["R"]).max() / ((1 + 1 / h08.MI) * E0)),
              "max_ledger_over_Wext": float(np.abs(h["ledger"]).max() / np.abs(h["Wx"]).max()),
              "max_Psum_over_Pe": float(np.abs(h["P"].sum(1)).max() / np.abs(h["P"][:, 0]).max())}
        out["runs"][c] = rr
        g2.append(rr["max_Psum_over_Pe"] <= 1e-9 and rr["max_ledger_over_Wext"] <= 1e-6)
        g3.append(rr["max_R_over_wL2E0"] <= 1e-8)
        win = h08.CERT[vq]
        v = {}
        for t in TQ:
            if t > win + 1e-9:
                continue
            a = {k: h08.at(h, k, t) for k in ("Wx", "dKe", "Te", "Ti", "dKi", "Qe", "Qi")}
            b = {k: h08.at(e, k, t) for k in a}
            rel = {k: a[k] / b[k] - 1 for k in a}
            rel["dKe_over_Wext"] = (a["dKe"] / a["Wx"]) / (b["dKe"] / b["Wx"]) - 1
            v[f"{t:g}"] = {"lbog": a, "em": b, "rel_change": rel}
            g4.append(all(abs(rel[k]) <= 2e-2 for k in ("Wx", "dKe", "Te", "Ti")))
        out["vs_em"][c] = v
        # full-window maximum over saves, EM vs LBO-G on the common grid
        tc = np.intersect1d(np.round(h["t"], 6), np.round(e["t"], 6))
        tc = tc[(tc > 50) & (tc <= win)]
        ih, ie = np.searchsorted(np.round(h["t"], 6), tc), np.searchsorted(np.round(e["t"], 6), tc)
        out["runs"][c]["max_rel_change_over_window"] = {
            k: float(np.max(np.abs(h[k][ih] - e[k][ie]) / np.maximum(np.abs(e[k][ie]), 1e-300)))
            for k in ("Wx", "dKe", "Te", "Ti")}
    out["gates"] = {"G0": all(g0), "G2": all(g2), "G3": all(g3), "G4": all(g4)}
    out["passed"] = all(out["gates"].values())
    out["repository_commit"], out["parent_commit"] = _sim._git_sha(), ds.PARENT_COMMIT
    (D / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["gates"]))


if __name__ == "__main__":
    D.mkdir(exist_ok=True)
    {"run": run_all, "analyze": analyze}[sys.argv[1]]()
