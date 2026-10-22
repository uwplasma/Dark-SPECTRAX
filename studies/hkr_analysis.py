"""hkr analysis with the gates of hkr_gates.py -> studies/hkr/summary.json (compact)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hkr_gates as G  # noqa: E402

D = Path(__file__).resolve().parent
VTE = np.sqrt(1e-3)
E0 = 0.03 * VTE


def load_grid(nx, seed):
    f = D / f"hkr/grid/h_vq0.03_Nx{nx}_Nv2048_v16_dt0.02_{seed}"
    f = Path(str(f) + ("1" if seed == "noise" else ""))
    if not f.with_suffix(".npz").exists() and not Path(str(f) + ".npz").exists():
        return None
    d = np.load(str(f) + ".npz")
    j = json.loads(Path(str(f) + ".json").read_text())
    t = d["t"]
    # energy defect relative to its initial value (K + U - initial total - W_ext), per the declared gate; the in-script
    # json t_defect omits the initial field energy and is not used
    de = np.abs(d["energy_defect"] - d["energy_defect"][0]) / np.maximum(np.maximum.accumulate(np.abs(d["W_ext"])), 1e-300)
    idd = np.nonzero(de > G.GATES["C_defect"])[0]
    j["t_defect_1e-6"] = float(t[idd[0]]) if idd.size else None
    cand = [j["T"]] + [x for x in (j["t_fneg_1e-3"], j["t_edge_1e-6"], j["t_defect_1e-6"]) if x is not None]
    tcert = min(cand)
    dE = d["dE_e"] / G.E_TH
    return dict(t=t, dE=dE, avg=G.trailing(t, dE), Ek2=d["Ek2"], Uk=d["U_k"], W=d["W_ext"], dK=d["dK_e"], tcert=tcert, rec=j,
                case=j["case"])


def at(r, t0):
    i = int(np.argmin(np.abs(r["t"] - t0)))
    return float(r["avg"][i])


def run_summary(r, nx):
    t = r["t"]
    m = t <= r["tcert"]
    kmask = (nx - 1) // 3
    S = r["Ek2"][m, : kmask + 1]
    smax = S.max(0)
    ref = smax[1]
    pop = np.nonzero(smax[1:] >= G.SPEC_FLOOR * ref)[0] + 1
    nhi = int(pop.max()) if pop.size else 0
    kl = 2 * np.pi / 40 * VTE ** 0 * nhi * VTE  # k lambda_De of the highest populated mode
    on = np.nonzero(2 * r["Uk"] >= E0 ** 2)[0]
    return {"case": r["case"], "T": r["rec"]["T"], "t_cert": r["tcert"], "t_fneg": r["rec"]["t_fneg_1e-3"],
            "t_edge": r["rec"]["t_edge_1e-6"], "t_defect": r["rec"]["t_defect_1e-6"],
            "kl_max_retained": float(2 * np.pi / 40 * kmask * VTE), "n_hi": nhi, "n_mask": kmask, "kl_hi": float(kl),
            "dE_avg_over_Eth": {str(t0): at(r, t0) for t0 in (100, 200, 300, 500, 700, 1000) if t0 <= r["rec"]["T"] + 1e-9 and t0 <= r["tcert"] + 1e-9},
            "dE_avg_at_tcert": float(r["avg"][int(np.argmin(np.abs(t - r["tcert"])))]),
            "dE_inst_max_over_Eth": float(r["dE"][m].max()),
            "t_otsi": float(t[on[0]]) if on.size else None,
            "max_Uk_over_E0sq": float(r["Uk"][m].max() / E0 ** 2), "wall_s": r["rec"]["wall_time"], "commit": r["rec"]["repository_commit"]}


def pair(a, b, tmax):
    m = (a["t"] <= tmax) & (a["t"] <= b["t"][-1])
    n = min(int(m.sum()), len(b["t"]))
    t = a["t"][:n]
    ta, ok = G.agree_time(t, a["avg"][:n], b["avg"][:n], G.FLOOR_E)
    return {"t_agree_dE": ta, "to_end": ok}


def main():
    out = {"gates": {k: v for k, v in G.GATES.items()}, "runs": {}, "pairs": {}, "q1": {}}
    runs = {}
    for seed in ("det", "noise"):
        for nx in (64, 256, 640, 1024):
            r = load_grid(nx, seed)
            if r is not None:
                runs[(nx, seed)] = r
                out["runs"][f"Nx{nx}_{seed}"] = run_summary(r, nx)
    for seed in ("det", "noise"):
        nxs = sorted(nx for (nx, s) in runs if s == seed)
        for lo, hi in zip(nxs[:-1], nxs[1:]):
            a, b = runs[(lo, seed)], runs[(hi, seed)]
            out["pairs"][f"{seed}_Nx{lo}_vs_{hi}"] = pair(a, b, min(a["tcert"], b["tcert"]))
            out["pairs"][f"{seed}_Nx{lo}_vs_{hi}"]["common_tcert"] = min(a["tcert"], b["tcert"])
    for nx in (64, 256, 640, 1024):  # seed effect at equal Nx
        if (nx, "det") in runs and (nx, "noise") in runs:
            a, b = runs[(nx, "det")], runs[(nx, "noise")]
            out["pairs"][f"Nx{nx}_det_vs_noise"] = pair(a, b, min(a["tcert"], b["tcert"]))
    # Q1 at t = Q1_t within the common certified window
    for seed in ("det", "noise"):
        ref = runs.get((64, seed))
        if ref is None:
            continue
        for (nx, s), r in runs.items():
            if s != seed or nx == 64:
                continue
            tt = min(G.GATES["Q1_t"], r["tcert"], ref["tcert"])
            ra, rb = at(r, tt), at(ref, tt)
            out["q1"][f"{seed}_Nx{nx}_over_Nx64"] = {"t": tt, "dE_hi": ra, "dE_64": rb, "ratio": ra / rb if rb else None,
                                                     "cutoff_matters": bool(rb and ra / rb >= G.GATES["Q1_ratio"]),
                                                     "hhs_reached": bool(ra >= G.GATES["Q1_hhs"])}
    (D / "hkr/summary_grid.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
