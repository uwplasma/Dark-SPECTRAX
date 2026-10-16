"""B11 collisional Landau: declared follow-up for two mis-specified gates of studies/b11_landau.py.

The recorded run (studies/b11/landau.json) FAILED G0 and G3 as declared; that record stands. Both gates were wrong:
- G0 compared chi_R2(nu = 1e-6) with the collisionless wofz chi to 1e-6, but the physical collisional change of chi
  is itself O(nu) (recorded 5.7e-7 and 1.9e-6), so the gate tested physics, not the reference.
- G3b (Dougherty rows 0-2 relative to the max over rows, per state) includes t = 0, where the Dougherty rate of the
  seeded Maxwellian is round-off in every row, so the ratio is 1 by construction (recorded 1.0 everywhere).
New gates, declared before this script is run (nothing else is changed; G1, G2, G4 and the ledger part of G3 are
reported from the original record):
G0' chi_R2 -> wofz chi as nu -> 0 linearly: with d(nu) = |chi_R2(nu) - chi_wofz| at the collisionless root,
    d(1e-5)/d(1e-4) and d(1e-6)/d(1e-5) within 10% of 0.1, for k lambda_De = 0.3, 0.5.
G3b' at 5 saved states of each run, max |Dougherty dCk rows 0-2| <= 1e-12 of the max |Dougherty dCk| over all rows
     and all 5 states of that run.
Run: python studies/b11_landau_followup.py -> studies/b11/landau_followup.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b11_landau as b  # noqa: E402
import darkspectrax as ds  # noqa: E402


def rows_ratio(model, out, Nn):
    free = ds.Model(Nx=model.Nx, Nn=Nn, Lx=model.Lx, alpha_s=model.alpha_s, rho_background=1.0, mode=model.mode,
                    eta=model.eta, Omega_D=model.Omega_D)
    low, high = 0.0, 0.0
    for i in np.linspace(0, out["t"].size - 1, 5).astype(int):
        y = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex)}
        d = (np.asarray(ds.rhs(out["t"][i], y, model)["Ck"]) - np.asarray(ds.rhs(out["t"][i], y, free)["Ck"]))
        d = np.abs(d.reshape(Nn, -1))
        low, high = max(low, float(d[:3].max())), max(high, float(d.max()))
    return low / high


if __name__ == "__main__":
    g0, g3 = [], []
    for kl in b.KL:
        w0 = b.find_root(kl, 0.0, lambda w: b.chi_wofz(w, kl), b.guess0[kl])[0]
        d = {nu: abs(b.chi_continuum(w0, kl, nu) - b.chi_wofz(w0, kl)) for nu in (1e-4, 1e-5, 1e-6)}
        g0.append({"k_lambda_De": kl, "d": {f"{k:g}": v for k, v in d.items()},
                   "ratios": [d[1e-5] / d[1e-4], d[1e-6] / d[1e-5]]})
        for dark in (False, True):
            for nu in b.NUS:
                for Nn in b.NNS:
                    m, out = b.simulate(kl, dark, nu, Nn)
                    g3.append({"k_lambda_De": kl, "model": "dark" if dark else "ordinary", "nu": nu, "Nn": Nn,
                               "rows_0_2_over_global_max": rows_ratio(m, out, Nn)})
                    print(json.dumps(g3[-1]), flush=True)
    gates = {"G0'": all(abs(r / 0.1 - 1) <= 0.1 for x in g0 for r in x["ratios"]),
             "G3b'": all(x["rows_0_2_over_global_max"] <= 1e-12 for x in g3)}
    rec = {"case_id": "B11-landau-followup", "command": "python studies/b11_landau_followup.py",
           "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "gates": gates, "passed": all(gates.values()), "G0_prime": g0, "G3b_prime": g3}
    (b.out_dir / "landau_followup.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(json.dumps(gates))
