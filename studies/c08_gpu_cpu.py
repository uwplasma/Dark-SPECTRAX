"""C08: GPU vs CPU agreement on a short H05 Hermite case (declared gate in studies/gpures_gates.py: max relative
difference <= 1e-6 of dK_e, W_ext, Tx, |E_k|^2 k <= 8, each against its max over the run).
Run the same case twice from the same venv, then compare:
  JAX_PLATFORMS=cpu python studies/nxconv_run.py --vq 0.1 --Nx 32 --Nn 64 --field-nu 4 --T 100 --noise-floor 1e-14 --tag _c08cpu
  CUDA_VISIBLE_DEVICES=0 python studies/nxconv_run.py ... --tag _c08gpu
  python studies/c08_gpu_cpu.py -> studies/gpures/c08.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nxconv_analysis import herm  # noqa: E402

D = Path(__file__).resolve().parent / "gpures"


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


def main():
    c, g = (herm(f"vq0.1_Nx32_Nn64_c4_nf1e-14_c08{x}") for x in ("cpu", "gpu"))
    n = min(c["t"].size, g["t"].size)
    out = {"n_saves": int(n), "t_end": [float(c["t"][-1]), float(g["t"][-1])],
           "dK_e": rel(c["dKe"][:n], g["dKe"][:n]), "W_ext": rel(c["Wx"][:n], g["Wx"][:n]),
           "Tx": rel(c["Tx"][:n], g["Tx"][:n]), "Ek2_k<=8": rel(c["Ek2"][:n, :9], g["Ek2"][:n, :9]),
           "steps": [c["recs"][0]["steps"], g["recs"][0]["steps"]],
           "run_s": [c["recs"][0]["run_time"], g["recs"][0]["run_time"]]}
    out["max_rel"] = max(out[k] for k in ("dK_e", "W_ext", "Tx", "Ek2_k<=8"))
    out["gate_1e-6"] = out["max_rel"] <= 1e-6
    D.mkdir(exist_ok=True)
    (D / "c08.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
