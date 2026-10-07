"""nxconv: LANES.md tables for the Nx-convergence runs (studies/nxconv/runs) and the improved grid (studies/nxconv/grid).

t_res(obs, a, b) = first t > 50 where |a - b| > 10% of the running max of |a| (a = the reference member of the pair:
the c = 1, Nn 64 run, or the finer Nx).  Observables: dK_e, W_ext (Hermite ledger W[:, 2]); field spectra: |E_k|^2
for every k strictly below the top retained mode of the coarser run (k <= Nx/3 - 1), each compared with the same
10% rule on its own running max, reported as the first such t over those k (and which k).
t_pos = first save with K_s <= 0 or k=0 T_x <= 0. Cost = summed steps, rejected, compile + run time over parts.
Grid: t_fneg = first save with min f_e < -1e-3 max f_e (after either sub-step); edge, energy defect, mass drift.
Run: python studies/nxconv_analysis.py -> studies/nxconv/summary.json (+ printed tables)
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
R, G = D / "nxconv" / "runs", D / "nxconv" / "grid"


def herm(case):
    parts = sorted(R.glob(f"{case}.part*.npz"), key=lambda p: int(p.suffixes[-2][5:]))
    if not parts:
        return None
    cat = {}
    recs = [json.loads(Path(str(p)[:-4] + ".json").read_text()) for p in parts]
    for i, p in enumerate(parts):
        d = np.load(p)
        for k in ("t", "K_species", "Tx", "W", "Ek2", "herm_n", "top_frac"):
            x = d[k] if i == 0 else d[k][1:]  # drop the repeated start of each resumed part
            cat.setdefault(k, []).append(x)
    h = {k: np.concatenate(v) for k, v in cat.items()}
    h["dKe"] = h["K_species"][:, 0] - h["K_species"][0, 0]
    h["Wx"] = h["W"][:, 2]
    h["recs"] = recs
    return h


def t_res(a, b, ka, kb=None):
    kb = kb or ka
    ta, tb = np.round(a["t"], 6), np.round(b["t"], 6)
    t = np.intersect1d(ta, tb)
    if t.size == 0:
        return None
    x, y = np.asarray(ka(a) if callable(ka) else a[ka])[np.searchsorted(ta, t)], \
        np.asarray(kb(b) if callable(kb) else b[kb])[np.searchsorted(tb, t)]
    scale = np.maximum.accumulate(np.abs(x))
    bad = np.nonzero((t > 50) & (np.abs(x - y) > 0.1 * scale))[0]
    return float(t[bad[0]]) if bad.size else f">={t[-1]:g}"


def spec_res(a, b, kmax):
    """First 10% disagreement over k = 1 .. kmax - 1 of |E_k|^2 (per-k running max)."""
    best = None
    for k in range(1, kmax):
        r = t_res(a, b, lambda h, k=k: h["Ek2"][:, k])
        if isinstance(r, float) and (best is None or r < best[0]):
            best = (r, k)
    if best is None:
        tt = np.intersect1d(np.round(a["t"], 6), np.round(b["t"], 6))
        return f">={tt[-1]:g}"
    return f"{best[0]:g} (k{best[1]})"


def t_pos(h):
    bad = np.nonzero((h["K_species"] <= 0).any(1) | (h["Tx"] <= 0).any(1))[0]
    return float(h["t"][bad[0]]) if bad.size else None


def cost(h):
    r = h["recs"]
    return {"t_reached": float(h["t"][-1]), "status": r[-1]["status"], "failure": r[-1]["failure_reason"],
            "parts": len(r), "steps": sum(x["steps"] for x in r), "rejected": sum(x["rejected"] for x in r),
            "compile_s": round(sum(x["compile_time"] for x in r), 1), "run_s": round(sum(x["run_time"] for x in r), 1),
            "wall_s": round(sum(x["wall_time"] for x in r), 1), "commits": sorted({x["repository_commit"][:7] for x in r}),
            "parent": r[0]["parent_commit"][:7]}


def kmax(nx):
    return (nx - 1) // 3


def grid(tag):
    p = G / f"{tag}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p))
    d["dKe"], d["Wx"] = d["dK_e"], d["W_ext"]
    d["rec"] = json.loads((G / f"{tag}.json").read_text())
    return d


def main():
    S = {"hermite": {}, "pairs": {}, "grid": {}, "grid_pairs": {}, "hermite_vs_grid": {}}
    H = {}
    for vq in (0.1, 0.03):
        for nx in (16, 32, 64, 128):
            for nn in (64, 128):
                for c in (0.5, 1, 2, 4, 8):
                    case = f"vq{vq:g}_Nx{nx}_Nn{nn}_c{c:g}_nf1e-14"
                    h = herm(case)
                    if h is not None:
                        H[case] = h
                        S["hermite"][case] = {**cost(h), "t_pos": t_pos(h)}
        for nx in (32, 64):
          for c in (1, 4):
            a, b = H.get(f"vq{vq:g}_Nx{2 * nx}_Nn64_c{c}_nf1e-14"), H.get(f"vq{vq:g}_Nx{nx}_Nn64_c{c}_nf1e-14")
            if a and b:
                S["pairs"][f"vq{vq:g} c{c} Nx{nx} vs {2 * nx}"] = {"dK_e": t_res(a, b, "dKe"), "W_ext": t_res(a, b, "Wx"),
                                                              f"Ek2 k<{kmax(nx)}": spec_res(a, b, kmax(nx))}
        ref = H.get(f"vq{vq:g}_Nx64_Nn64_c4_nf1e-14")
        for tag, lab in (("Nn128_c4", "c4 Nn 64 vs 128"), ("Nn64_c1", "c 4 vs 1"), ("Nn64_c2", "c 4 vs 2"),
                         ("Nn64_c8", "c 4 vs 8")):
            b = H.get(f"vq{vq:g}_Nx64_{tag}_nf1e-14")
            if ref and b:
                S["pairs"][f"vq{vq:g} Nx64 {lab}"] = {"dK_e": t_res(ref, b, "dKe"), "W_ext": t_res(ref, b, "Wx"),
                                                      f"Ek2 k<{kmax(64)}": spec_res(ref, b, kmax(64))}
    for p in sorted(G.glob("g_vq*.json")):
        tag = p.stem
        if "landau" in tag or "unseeded" in tag or "_T" in tag:
            continue
        d = grid(tag)
        if d is None:
            continue
        r = d["rec"]
        S["grid"][tag] = {k: r[k] for k in ("T", "finite", "wall_time", "t_fneg_1e-3", "min_f_rel", "max_negfrac",
                                            "max_edge", "max_energy_defect_over_W", "max_mass_drift")}
    gd = {k: grid(k) for k in S["grid"]}
    # grid pairs: v-resolution, scheme, dt, frame (all against the osc-pfc Nv 8192 dt 0.02 run when present)
    ref = "g_vq0.1_Nx16_Nv8192_dt0.02_osc_v32_pfc"
    if ref in gd:
        for k, d in gd.items():
            if k != ref and k.startswith("g_vq0.1_Nx16_"):
                S["grid_pairs"][f"{ref} vs {k}"] = {"dK_e": t_res(gd[ref], d, "dKe"), "W_ext": t_res(gd[ref], d, "Wx"),
                                                    "Ek2 k<5": spec_res(gd[ref], d, 5)}
    for k, d in gd.items():
        nx = int(k.split("_Nx")[1].split("_")[0])
        vq = float(k.split("_vq")[1].split("_")[0])
        h = H.get(f"vq{vq:g}_Nx{nx}_Nn64_c4_nf1e-14")
        if h is not None and "xmask23" in k:
            S["hermite_vs_grid"][f"{k}"] = {"dK_e": t_res(h, d, "dKe"), "W_ext": t_res(h, d, "Wx"),
                                            f"Ek2 k<{kmax(nx)}": spec_res(h, d, kmax(nx))}
    for vq in (0.1, 0.03):
        for nx in (32, 64):
            a, b = gd.get(f"g_vq{vq:g}_Nx{2*nx}_Nv4096_dt0.02_osc_v32_spectral_xmask23"), \
                gd.get(f"g_vq{vq:g}_Nx{nx}_Nv4096_dt0.02_osc_v32_spectral_xmask23")
            if a is not None and b is not None:
                S["grid_pairs"][f"mask23 vq{vq:g} Nx{nx} vs {2*nx}"] = {"dK_e": t_res(a, b, "dKe"), "W_ext": t_res(a, b, "Wx"),
                                                                    f"Ek2 k<{kmax(nx)}": spec_res(a, b, kmax(nx))}
    (D / "nxconv" / "summary.json").write_text(json.dumps(S, indent=1, default=str) + "\n")
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
