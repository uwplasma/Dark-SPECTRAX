"""highk (d): Floquet exponents of the truncated AW (and SW) Fourier-Hermite operator (studies/highk_eig.py) in an
OSCILLATING field E(x,t) = E1 cos(k1 x) cos(w t), w = 1 (the k1 field of H05 oscillates near omega_pe).
Exponent = log(max |eig(monodromy)|)/period. Exact Vlasov: 0 (Liouville). Declared criterion: the AW truncation is
the failure mechanism if its exponent at the measured E1 (2|E_k1| ~ 8e-3 at v_q=0.1, t~200-450; ~8e-4 at v_q=0.01)
is O(0.1) or larger at Nx=16 (K=5), smaller at Nx=8 (K=2), and negligible at v_q=0.01; SW must give 0.
Run: python studies/highk_floquet.py -> studies/highk/floquet.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from highk_eig import operator  # noqa: E402

a = np.sqrt(2e-3)


def exponent(K, Nn, E1, basis, w=1.0, nsub=200):
    L0 = operator(K, Nn, 0.0, a, 0.0, basis)
    L1 = operator(K, Nn, E1, a, 0.0, basis) - L0
    T = 2 * np.pi / w
    h = T / nsub
    M = np.eye(L0.shape[0], dtype=complex)
    for j in range(nsub):  # exponential midpoint (2nd order Magnus)
        M = expm(h * (L0 + np.cos(w * (j + 0.5) * h) * L1)) @ M
    return float(np.log(np.abs(np.linalg.eigvals(M)).max()) / T)


if __name__ == "__main__":
    res = []
    for basis, K, Nn, E1 in itertools.product(("AW", "SW"), (2, 5), (32, 64, 128), (8e-4, 3e-3, 8e-3, 2e-2)):
        if basis == "SW" and Nn == 128:
            continue
        res.append({"basis": basis, "K": K, "Nn": Nn, "E1": E1, "floquet": exponent(K, Nn, E1, basis)})
        print(json.dumps(res[-1]), flush=True)
    (Path(__file__).resolve().parent / "highk" / "floquet.json").write_text(json.dumps(res, indent=0) + "\n")
