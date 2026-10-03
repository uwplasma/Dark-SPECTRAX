"""Post-process studies/hhs_scan records: energies at the last time the work ledger still closes.

Later samples of a failed run (step budget exhausted) are not physical and are not reported.
Run: python studies/hhs_summary.py (rewrites studies/hhs_scan/run.json in place).
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "hhs_scan"


K0 = 0.75 * 2e-3  # basis-matched Maxwellian, a^2 = 2 v_t^2 = 2e-3: K_e0 = K_i0 = (3/4) m a^2 n


def first_bad(t, x):
    return float(t[np.argmax(x)]) if np.any(x) else None


def agreement_time(arrays, grp, k1, k2, tol=0.1):
    """First time the electron kinetic-energy change of two runs differs by > tol of the larger magnitude."""
    t1, d1 = arrays[f"{grp}_{k1}_tK"], arrays[f"{grp}_{k1}_dK"][:, 0]
    t2, d2 = arrays[f"{grp}_{k2}_tK"], arrays[f"{grp}_{k2}_dK"][:, 0]
    n = min(t1.size, t2.size)
    t1, d1, d2 = t1[:n], d1[:n], d2[:n]
    scale = np.maximum(np.maximum(np.abs(d1), np.abs(d2)), 1e-12)
    bad = np.abs(d1 - d2) > tol * scale
    bad[t1 < 50] = False  # ignore the initial transient where both changes are ~0
    return first_bad(t1, bad), float(t1[-1])


def summarize(rec, arrays):
    for grp in ("H05", "H06"):
        for key, r in rec[grp].items():
            if not isinstance(r, dict) or "t_ledger_valid" not in r:
                continue
            tK, dK = arrays[f"{grp}_{key}_tK"], arrays[f"{grp}_{key}_dK"]
            t, W = arrays[f"{grp}_{key}_t"], arrays[f"{grp}_{key}_Wext"]
            i = np.nonzero(tK <= r["t_ledger_valid"])[0][-1]
            j = np.nonzero(t <= tK[i])[0][-1]
            r["at_valid"] = {"t": float(tK[i]), "W_ext": float(W[j]), "dK_electron": float(dK[i, 0]),
                             "dK_ion": float(dK[i, 1]), "ion_share": float(dK[i, 1] / (dK[i, 0] + dK[i, 1]))}
            # K_s >= 0 for any nonnegative f; a negative moment energy marks an inadmissible Hermite state
            r["t_admissible"] = first_bad(tK, (K0 + dK[:, 0] < 0) | (K0 + dK[:, 1] < 0))
            for stale in ("W_ext_final", "dK_electron_final", "dK_ion_final", "ion_share_of_dK_final",
                          "W_ext_over_n_Te_final"):
                r.pop(stale, None)
    for ratio in ("0.001", "0.003", "0.01", "0.03", "0.1"):
        g = f"vq{ratio}"
        A = {"Nn32_vs_Nn64_nu0": agreement_time(arrays, "H05", f"{g}_Nn32_nu0", f"{g}_Nn64_nu0"),
             "nu0_vs_nu1_Nn64": agreement_time(arrays, "H05", f"{g}_Nn64_nu0", f"{g}_Nn64_nu1")}
        ends = [v[0] if v[0] is not None else v[1] for v in A.values()]
        tr = min(ends)
        key = f"{g}_Nn64_nu0"
        tK, dK = arrays[f"H05_{key}_tK"], arrays[f"H05_{key}_dK"]
        t, W = arrays[f"H05_{key}_t"], arrays[f"H05_{key}_Wext"]
        i = np.nonzero(tK < tr)[0][-1] if np.any(A["Nn32_vs_Nn64_nu0"][0] or A["nu0_vs_nu1_Nn64"][0]) else tK.size - 1
        j = np.nonzero(t <= tK[i])[0][-1]
        A["resolved_until"] = float(tK[i])
        A["Nn64_nu0_at_resolved"] = {"W_ext": float(W[j]), "dK_electron": float(dK[i, 0]), "dK_ion": float(dK[i, 1]),
                                     "W_ext_over_n_Te": float(W[j] / 1e-3)}
        rec["H05"][f"{g}_agreement"] = A
    for d in ("up", "down"):
        rec["H06"][f"{d}_agreement"] = {"Nn32nu0_vs_Nn64nu1": agreement_time(arrays, "H06", f"{d}_Nn32_nu0",
                                                                              f"{d}_Nn64_nu1")}
    return rec


if __name__ == "__main__":
    data = dict(np.load(OUT / "run.npz"))
    rec = summarize(json.loads((OUT / "run.json").read_text()), data)
    (OUT / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
