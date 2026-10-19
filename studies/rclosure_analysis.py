"""rclosure stage F analysis (gates declared in studies/rclosure_gates.py): c in {16, 64, 256} vs the c = 4 record.
Run: python studies/rclosure_analysis.py -> studies/rclosure/stagef.json"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from nxconv_analysis import herm, t_pos  # noqa: E402


def ledger(h):
    return float(np.abs(h["K_species"].sum(1) - h["K_species"][0].sum() - h["W"].sum(1)).max() / max(np.abs(h["Wx"]).max(), 1e-300))


def fidelity(h, ref, tmax=500.0):
    """max over t <= tmax of |x - y| / running max |y| for dK_e and W_ext (y = c = 4 record); None if the run ends earlier."""
    out = {}
    for lab, k in (("dK_e", "dKe"), ("W_ext", "Wx")):
        ta, tb = np.round(h["t"], 6), np.round(ref["t"], 6)
        t = np.intersect1d(ta, tb)
        t = t[t <= tmax]
        x, y = h[k][np.searchsorted(ta, t)], ref[k][np.searchsorted(tb, t)]
        out[lab] = float((np.abs(x - y) / np.maximum.accumulate(np.maximum(np.abs(y), 1e-300))).max()) if t.size else None
    out["t_compared"] = float(t[-1]) if t.size else None
    return out


def main():
    ref_npz = np.load(D / "gpures" / "records" / "vq0.1_Nx64_Nn128_c4_nf1e-14_gpu.npz")
    ref = {"t": ref_npz["t"], "dKe": ref_npz["dKe"], "Wx": ref_npz["Wx"]}
    S = {"c4_reference": {"t_end": float(ref["t"][-1])}, "runs": {}}
    for c in (16, 64, 256):
        h = herm(f"vq0.1_Nx64_Nn128_c{c}_nf1e-14_rcf")
        if h is None:
            continue
        tp = t_pos(h)
        reached = float(h["t"][-1])
        S["runs"][f"c{c}"] = {"t_reached": reached, "t_pos": tp, "window": min([reached] + ([tp] if tp is not None else [])),
                              "status": h["recs"][-1]["status"], "ledger_rel": ledger(h), "fidelity_t500": fidelity(h, ref)}
    w256 = S["runs"].get("c256", {}).get("window")
    S["F1"] = bool(w256 is not None and w256 >= 744.0)
    S["F2"] = {k: bool(all(v is not None and v <= 1e-3 for kk, v in r["fidelity_t500"].items() if kk != "t_compared"))
               for k, r in S["runs"].items()}
    S["F3"] = {k: bool(r["ledger_rel"] <= 1e-6 and r["status"] in ("success", "short")) for k, r in S["runs"].items()}
    (D / "rclosure" / "stagef.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
