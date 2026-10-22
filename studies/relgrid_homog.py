"""relgrid_homog: k = 0 (homogeneous) relativistic detuning of the resonant drive from the exact kinetic ODE (studies/relgrid_ref.py).

No gates (a measurement with an exact reference): for v_q/u_te = 0.03 and 1e-3, T = 5000, compare the relativistic homogeneous kinetic oscillator
with the nonrelativistic one (the linear law): Ebar(t), W_ext(t), the pump phase of Ebar in sliding 200/omega_pe windows (detuning = d phase/dt),
the electron mean momentum U_e and mean gamma - 1, and the first time the relativistic W_ext departs from the nonrelativistic by 10 percent.
This is the pure k = 0 part of the dynamics: it contains no instability and no k != 0 energy, so it isolates the relativistic detuning of the
uniform resonance (an upper bound on what relativity alone does; thermal spread included through the exact quadrature over the Maxwellian-in-u load).
Run: python studies/relgrid_homog.py -> studies/relgrid/homog.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relgrid_ref as X  # noqa: E402

HERE = Path(__file__).resolve().parent
(HERE / "relgrid").mkdir(exist_ok=True)
out = {"repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()}
T = 5000.0
t = np.arange(0, T + 1e-9, 0.5)
for vq in (0.03, 1e-3):
    rel, nr = X.homogeneous(vq, T, rel=True), X.homogeneous(vq, T, rel=False)
    yr, yn = rel.sol(t), nr.sol(t)
    cold = X.cold_fluid(vq, T, rel=True).sol(t)
    # sliding-window pump phase and amplitude of Ebar
    centers = np.arange(100.0, T - 99, 100.0)
    ph = {}
    for nm, y in (("rel", yr), ("nonrel", yn)):
        arr = [X.demod_phase(t, y[2], X.OMEGA, c - 100, c + 100) for c in centers]
        ph[nm] = (np.unwrap([a[0] for a in arr]), np.array([a[1] for a in arr]))
    dev = np.abs(yr[3] - yn[3]) / np.maximum(np.abs(yn[3]), 1e-300)
    sel = (t > 50) & (np.abs(yn[3]) > 0)
    i10 = np.nonzero(sel & (dev > 0.1))[0]
    ratioE = np.abs(ph["rel"][1]) / np.abs(ph["nonrel"][1])
    out[f"vq{vq:g}"] = {
        "t": centers.tolist(), "Ebar_amp_rel": ph["rel"][1].tolist(), "Ebar_amp_nonrel": ph["nonrel"][1].tolist(),
        "phase_rel": ph["rel"][0].tolist(), "phase_nonrel": ph["nonrel"][0].tolist(),
        "W_rel_over_nonrel_at": {str(int(c)): float(yr[3][np.argmin(np.abs(t - c))] / yn[3][np.argmin(np.abs(t - c))]) for c in (200, 500, 1000, 2000, 3000, 5000)},
        "t_W_10pc": float(t[i10[0]]) if i10.size else None,
        "Ue_rel_at": {str(int(c)): float(yr[0][np.argmin(np.abs(t - c))]) for c in (200, 500, 1000, 2000, 5000)},
        "max_abs_Ue_rel": float(np.abs(yr[0]).max()), "cold_vs_thermal_Ebar_maxrel": float(np.abs(cold[2] - yr[2]).max() / np.abs(yr[2]).max()),
        "Ebar_amp_ratio_at": {str(int(c)): float(ratioE[np.argmin(np.abs(centers - c))]) for c in (500, 1000, 2000, 3000, 5000 - 100)},
    }
(HERE / "relgrid" / "homog.json").write_text(json.dumps(out) + "\n")
for k, v in out.items():
    if k.startswith("vq"):
        print(k, {a: v[a] for a in ("W_rel_over_nonrel_at", "t_W_10pc", "Ue_rel_at", "Ebar_amp_ratio_at", "max_abs_Ue_rel")})
