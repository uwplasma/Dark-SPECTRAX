"""relgrid analysis (H04): HHS quantities, relativistic detuning, comparison with the nonrelativistic records, refinement windows (gate E), ledger (gate D).

Reads studies/relgrid/runs/*.npz (+ gates_*.json, homog.json), the nonrelativistic nxconv grid records and writes studies/relgrid/summary.json.
HHS quantities (PRL Fig. 2, SM Eqs. S35-S36, footnote 8): dE1D_e = (1/2) m <(v - <v>)^2> change (1-D velocity variance, box-mean flow removed),
units n m c^2, E_th = 5e-4; mean-field energy (1/2) Ebar^2 (HHS: <E^2>/2); all with a trailing 200/omega_pe average. Detuning: pump phase of Ebar
(windowed demodulation, half width 100) and its drift; linear law = exact nonrelativistic homogeneous oscillator (relgrid_ref.homogeneous(rel=False)).
Run: python studies/relgrid_analysis.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relgrid_ref as X  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "relgrid" / "runs"
E_TH = 0.5e-3
TQ = (200, 500, 1000, 1500, 2000, 3000, 4000, 5000)


def mavg(t, y, w=200.0):
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty_like(y)
    dt = t[1] - t[0]
    n = int(round(w / dt))
    for i in range(y.size):
        j = max(0, i - n + 1)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


def at(t, y, tq):
    return float(y[int(np.argmin(np.abs(t - tq)))])


def load(tag):
    p = RUNS / f"{tag}.npz"
    return {k: v for k, v in np.load(p).items()} if p.exists() else None


def window(a, b, ta, tb, obs, tol=0.1):
    """last time two records agree within tol of the running max of the reference (nxconv rule); '>=' if they agree to the end."""
    n = min(a[obs].size, b[obs].size)
    x, y = a[obs][:n], b[obs][:n]
    bad = np.abs(x - y) > tol * np.maximum.accumulate(np.abs(y)) + 1e-300
    bad &= np.arange(n) > 4
    return float(ta[np.argmax(bad) - 1]) if bad.any() else float(ta[n - 1])


def quantities(r, vq, w=X.OMEGA):
    t = r["t"]
    out = {"T": float(t[-1])}
    dE = mavg(t, r["dE1D_e"])
    fld = mavg(t, 0.5 * r["Ebar"] ** 2)
    lin = X.homogeneous(vq, float(t[-1]) + 1, rel=False)
    yl = lin.sol(t)
    fl = mavg(t, 0.5 * yl[2] ** 2)
    Uk = mavg(t, r["U_k"])
    out["dE1D_over_Eth"] = {str(q): at(t, dE, q) / E_TH for q in TQ if q <= t[-1]}
    out["max_dE1D_over_Eth"] = float(dE.max() / E_TH)
    out["min_dE1D_over_Eth"] = float(dE.min() / E_TH)
    out["Ebar_energy"] = {str(q): at(t, fld, q) for q in TQ if q <= t[-1]}
    out["Ebar_energy_linear"] = {str(q): at(t, fl, q) for q in TQ if q <= t[-1]}
    out["Ebar_energy_over_linear"] = {str(q): at(t, fld, q) / at(t, fl, q) for q in TQ if q <= t[-1]}
    out["U_k_avg"] = {str(q): at(t, Uk, q) for q in TQ if q <= t[-1]}
    out["W_over_linear"] = {str(q): at(t, r["W_ext"], q) / at(t, yl[3], q) for q in TQ if q <= t[-1]}
    out["gm1_e_max"] = float(r["gm1_e"].max())
    out["dK_e_over_Eth_at"] = {str(q): at(t, r["dK_e"], q) / E_TH for q in TQ if q <= t[-1]}
    # detuning: pump phase of Ebar in windows [c - 100, c + 100]
    cs = np.arange(100.0, t[-1] - 99, 100.0)
    ph = np.unwrap([X.demod_phase(t, r["Ebar"], w, c - 100, c + 100)[0] for c in cs]) if cs.size > 1 else np.array([])
    pl = np.unwrap([X.demod_phase(t, yl[2], w, c - 100, c + 100)[0] for c in cs]) if cs.size > 1 else np.array([])
    amp = np.array([X.demod_phase(t, r["Ebar"], w, c - 100, c + 100)[1] for c in cs])
    out["pump_phase"] = {"t": cs.tolist(), "rel": ph.tolist(), "linear": pl.tolist(), "amp": amp.tolist()}
    out["energy_ledger"] = float(np.abs(r["energy_defect"]).max() / max(np.abs(r["W_ext"]).max(), 1e-300))
    out["mass_drift"] = float(max(np.abs(r["mass_e"]).max(), np.abs(r["mass_i"]).max()))
    out["edge"] = float(r["edge"].max())
    neg = np.nonzero(np.minimum(r["fmin_v"], r["fmin_x"]) < -1e-3)[0]
    out["t_fneg_1e-3"] = float(t[neg[0]]) if neg.size else None
    # time at which the mean-field amplitude (demodulated) peaks and first departure from the linear law by 10 percent
    d10 = np.nonzero((t > 50) & (np.abs(r["W_ext"] - yl[3]) > 0.1 * np.abs(yl[3])))[0]
    out["t_W_10pc_from_linear"] = float(t[d10[0]]) if d10.size else None
    return out


CASES = {  # label -> (tag stem suffixes)
    "rel_det_vq0.03": ("r_vq0.03_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_det_xmask23", 0.03),
    "rel_shot_vq0.03": ("r_vq0.03_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_shot1_xmask23", 0.03),
    "nonrel_shot_vq0.03": ("r_vq0.03_Nx32_Nu4096_dt0.02_osc_u32_nr_taylor_shot1_xmask23", 0.03),
    "rel_det_vq0.001": ("r_vq0.001_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_det_xmask23", 1e-3),
    "rel_shot_vq0.001": ("r_vq0.001_Nx32_Nu4096_dt0.02_osc_u32_rel_taylor_shot1_xmask23", 1e-3),
    "nonrel_det_vq0.001": ("r_vq0.001_Nx32_Nu4096_dt0.02_osc_u32_nr_taylor_det_xmask23", 1e-3),
    "nonrel_shot_vq0.001": ("r_vq0.001_Nx32_Nu4096_dt0.02_osc_u32_nr_taylor_shot1_xmask23", 1e-3),
}
REFINE = {  # gate E (v_q = 0.03 relativistic det, T = 1000)
    "Nx64": "r_vq0.03_Nx64_Nu4096_dt0.02_osc_u32_rel_taylor_det_xmask23",
    "Nu8192": "r_vq0.03_Nx32_Nu8192_dt0.02_osc_u32_rel_taylor_det_xmask23",
    "dt0.01": "r_vq0.03_Nx32_Nu4096_dt0.01_osc_u32_rel_taylor_det_xmask23",
    "umax48": "r_vq0.03_Nx32_Nu6144_dt0.02_osc_u48_rel_taylor_det_xmask23",
}

if __name__ == "__main__":
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"repository_commit": sha, "E_th": E_TH, "cases": {}, "refine": {}}
    runs = {}
    for lab, (tag, vq) in CASES.items():
        r = load(tag)
        if r is not None:
            runs[lab] = r
            out["cases"][lab] = quantities(r, vq)
            out["cases"][lab]["tag"] = tag
    # nonrelativistic deterministic records of the nxconv grid (T = 1000, same setup): HHS quantities need E1D, not stored -> dK_e, W_ext, U_k only
    nx = HERE / "nxconv/grid/g_vq0.03_Nx32_Nv4096_dt0.02_osc_v32_spectral_xmask23.npz"
    if nx.exists():
        d = np.load(nx)
        out["nxconv_nonrel_det_vq0.03"] = {"dK_e_over_Eth": {str(q): at(d["t"], d["dK_e"], q) / E_TH for q in (200, 500, 1000)},
                                          "W_ext": {str(q): at(d["t"], d["W_ext"], q) for q in (200, 500, 1000)}}
    base = load(CASES["rel_det_vq0.03"][0])
    if base is not None:
        t = base["t"]
        for k, tag in REFINE.items():
            o = load(tag)
            if o is None:
                continue
            out["refine"][k] = {obs: window(o, base, o["t"], t, obs) for obs in ("dK_e", "dE1D_e", "W_ext", "U_k")}
            out["refine"][k]["T"] = float(o["t"][-1])
        if out["refine"]:
            out["refine_window_min"] = min(v for d_ in out["refine"].values() for kk, v in d_.items() if kk != "T")
    # gate D over every run found
    led = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json")}
    out["gate_D"] = {k: {"energy_defect_over_W": v["max_energy_defect_over_W"], "mass": v["max_mass_drift"], "edge": v["max_edge"],
                         "pass": bool(v["max_energy_defect_over_W"] <= 1e-4 and v["max_mass_drift"] <= 1e-10 and v["max_edge"] <= 1e-6)}
                     for k, v in led.items()}
    for g in ("a", "b", "c"):
        p = HERE / "relgrid" / f"gates_{g}.json"
        if p.exists():
            out[f"gates_{g}"] = json.loads(p.read_text())
    p = HERE / "relgrid" / "homog.json"
    if p.exists():
        h = json.loads(p.read_text())
        out["homog"] = {k: {a: b for a, b in v.items() if a in ("W_rel_over_nonrel_at", "t_W_10pc", "Ue_rel_at", "max_abs_Ue_rel", "Ebar_amp_ratio_at")}
                        for k, v in h.items() if k.startswith("vq")}
    (HERE / "relgrid" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k in ("refine", "refine_window_min", "gate_D")}, indent=1))
    for lab, v in out["cases"].items():
        print(lab, "T", v["T"], "dE1D/Eth", v["dE1D_over_Eth"], "Ebar/lin", v["Ebar_energy_over_linear"], "t10", v["t_W_10pc_from_linear"], "ledger", v["energy_ledger"])
