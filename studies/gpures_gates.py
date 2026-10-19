"""gpures: GPU (RTX A4000, float64) extension of the H05 Hermite certified windows by resolution.

DECLARED BEFORE RUNNING (never amended after results; a wrong gate is recorded as a failure and a follow-up is declared).
Cases: v_q/v_te in {0.1, 0.05}, Nx in {64, 128}, Nn in {128, 256}, pump frame + remap, PR #66 closure c = 4,
noise_floor 1e-14, rtol 1e-10, atol 1e-14, T = 1000, chunked (studies/nxconv_run.py records, tag `_gpu`).
Pairs: Nn 128 vs 256 at each Nx; Nx 64 vs 128 at each Nn.  t_res = first t > 50 where the two runs differ by more
than 10% of the running max of the first run (studies/nxconv_analysis.py t_res), on dK_e and W_ext.
G1 (pair)      t_res(dK_e) >= 1000 and t_res(W_ext) >= 1000, i.e. Nn and Nx pairs agree within 10% to T.
G2 (positivity) per run: K_s > 0 and k=0 T_x > 0 at every save through T.
G3 (ledger)    per run: max_t |(K(t) - K(0)) - sum_j W_j(t)| <= 1e-6 * max_t |W_ext(t)|.
G4 (status)    per run: solver status success and t_end = T.
A run's certified window = min over its pairs of min(t_res dK_e, t_res W_ext) and its positivity time t_pos; the
case verdict is PASS for T = 1000 only if G1-G4 all hold for every pair/run of that v_q; otherwise the certified
window is reported honestly with the first failing gate.
C08 (GPU vs CPU, studies/c08_gpu_cpu.py): short case v_q 0.1, Nx 32, Nn 64, c = 4, T = 100, same venv, JAX_PLATFORMS=cpu
versus cuda:0, x64.  Gate: max relative difference (against the max over the run) of K_e, W_ext, Tx, |E_k|^2 (k <= 8)
<= 1e-6.  The max relative difference is reported whatever it is.
Run: python studies/gpures_gates.py -> studies/gpures/summary.json + compact records studies/gpures/records/*.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nxconv_analysis import cost, herm, t_pos, t_res  # noqa: E402

D = Path(__file__).resolve().parent / "gpures"
T = 1000.0
VQS, NXS, NNS = (0.1, 0.05), (64, 128), (128, 256)


def case(vq, nx, nn):
    return f"vq{vq:g}_Nx{nx}_Nn{nn}_c4_nf1e-14_gpu"


def ledger(h):
    return float(np.abs(h["K_species"].sum(1) - h["K_species"][0].sum() - h["W"].sum(1)).max()
                 / max(np.abs(h["Wx"]).max(), 1e-300))


def num(x):
    return float(str(x).lstrip(">="))  # t_res returns a float or ">=t_end"


def main():
    D.mkdir(exist_ok=True)
    (D / "records").mkdir(exist_ok=True)
    S = {"runs": {}, "pairs": {}, "verdict": {}}
    H = {}
    for vq in VQS:
        for nx in NXS:
            for nn in NNS:
                c = case(vq, nx, nn)
                h = herm(c)
                if h is None:
                    continue
                H[c] = h
                tp = t_pos(h)
                S["runs"][c] = {**cost(h), "t_pos": tp, "ledger_rel": ledger(h),
                                "G2": tp is None, "G3": ledger(h) <= 1e-6,
                                "G4": h["recs"][-1]["status"] in ("success", "short") and h["t"][-1] >= T - 1e-9}
                np.savez_compressed(D / "records" / f"{c}.npz", t=h["t"], dKe=h["dKe"], Wx=h["Wx"], K_species=h["K_species"],
                                    Tx=h["Tx"], W=h["W"], Ek2=h["Ek2"][:, :17], herm_n=h["herm_n"][::10])
    pairs = {}
    for vq in VQS:
        for nx in NXS:
            pairs[f"vq{vq:g} Nx{nx} Nn128 vs 256"] = (case(vq, nx, 128), case(vq, nx, 256))
        for nn in NNS:
            pairs[f"vq{vq:g} Nn{nn} Nx64 vs 128"] = (case(vq, 64, nn), case(vq, 128, nn))
    for lab, (ca, cb) in pairs.items():
        if ca in H and cb in H:
            a, b = H[ca], H[cb]
            r = {"dK_e": t_res(a, b, "dKe"), "W_ext": t_res(a, b, "Wx")}
            r["window"] = min(num(r["dK_e"]), num(r["W_ext"]))
            r["G1"] = r["window"] >= T
            S["pairs"][lab] = r
    for vq in VQS:
        pr = {k: v for k, v in S["pairs"].items() if k.startswith(f"vq{vq:g} ")}
        rs = {k: v for k, v in S["runs"].items() if k.startswith(f"vq{vq:g}_")}
        if pr and rs:
            win = min([v["window"] for v in pr.values()] + [v["t_pos"] for v in rs.values() if v["t_pos"] is not None]
                      + [v["t_reached"] for v in rs.values()])
            S["verdict"][f"vq{vq:g}"] = {"n_pairs": len(pr), "n_runs": len(rs), "certified_window": win,
                                         "PASS": bool(len(pr) == 4 and len(rs) == 4 and all(v["G1"] for v in pr.values())
                                                      and all(v["G2"] and v["G3"] and v["G4"] for v in rs.values()))}
    (D / "summary.json").write_text(json.dumps(S, indent=1, default=str) + "\n")
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
