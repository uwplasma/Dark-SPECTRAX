"""B02 warm two-stream and B03 bump-on-tail, ordinary and dark, against kinetic Maxwell-Proca roots.

Parent units (c = 1, omega_pe = 1). Electrostatic inputs (v_te = 1) are mapped with v_t/c = beta = 0.1.
Each population has its own drifting Hermite basis. Growth is the slope of log|E_k| over a window that
starts after the stable-branch transient and ends before |E_k| exceeds 1e-5 (code units), i.e. linear.
Reference: D_L = (Q - Omega_D^2)(1 + chi) + eta^2 Q chi = 0, chi = sum of drifting-Maxwellian terms (wofz).

Run: python examples/instabilities.py  -> docs/_static/b02_b03/{run.json, run.npz}; figure: python studies/figures.py growth
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

import darkspectrax as ds

out_dir = Path(__file__).resolve().parents[1] / "docs" / "_static" / "b02_b03"
beta, eta, Omega_D, seed = 0.1, 0.3, 1.0, 1e-13
cases = {  # name: (populations [(fraction, v_t, u)] in electrostatic units, k_es, Nn, t_max)
    "B02_two_stream_vt0.1_k0.6": ([(0.5, 0.1, 1.0), (0.5, 0.1, -1.0)], 0.6, 32, 70.0),
    "B02_two_stream_vt0.3_k0.4": ([(0.5, 0.3, 1.0), (0.5, 0.3, -1.0)], 0.4, 32, 80.0),
    "B03_bump_on_tail_k0.3": ([(0.9, 1.0, -0.45), (0.1, 0.5, 4.05)], 0.3, 64, 120.0),
}


def chi(w, k, pops):
    tot = 0.0
    for fr, vt, u in pops:
        z = (w - k * u) / (np.sqrt(2) * k * vt)
        tot = tot + fr / (k * vt) ** 2 * (1 + z * 1j * np.sqrt(np.pi) * wofz(z))
    return tot


def solve(k, pops, eta_, guess):
    def D(w):
        Q = w ** 2 - k ** 2
        c = chi(w, k, pops)
        return (Q - Omega_D ** 2) * (1 + c) + eta_ ** 2 * Q * c

    s = root(lambda x: [D(x[0] + 1j * x[1]).real, D(x[0] + 1j * x[1]).imag], [guess.real, guess.imag],
             tol=1e-14)
    w = s.x[0] + 1j * s.x[1]
    return w, abs(D(w))


def most_unstable(k, pops):
    """Newton from a grid of upper-half-plane starts; keep the converged root of largest growth."""
    best = None
    for wr in np.linspace(0.0, 2.5, 11):
        for wi in (0.05, 0.15, 0.3, 0.5):
            w, r = solve(k, pops, 0.0, wr + 1j * wi)
            if r < 1e-10 and w.real >= -1e-9 and (best is None or w.imag > best.imag + 1e-9):
                best = w
    return best


rows, arrays = [], {}
for name, (pops_es, k_es, Nn, tmax) in cases.items():
    pops = [(fr, beta * vt, beta * u) for fr, vt, u in pops_es]
    k = k_es / beta
    w0, r0 = solve(k, pops, 0.0, most_unstable(k, pops))
    wd, rd = solve(k, pops, eta, w0)
    for dark, NnRun in ((False, Nn), (True, Nn), (False, 2 * Nn), (True, 2 * Nn)):
        model = ds.Model(Nx=5, Nn=NnRun, Lx=2 * np.pi / k, qs=(-1.0,) * len(pops), Omega_cs=(1.0,) * len(pops),
                         alpha_s=tuple(np.repeat([np.sqrt(2) * vt for _, vt, _ in pops], 3)),
                         u_s=tuple(v for _, _, u in pops for v in (u, 0.0, 0.0)), rho_background=1.0,
                         mode="self_consistent" if dark else "ordinary", eta=eta if dark else 0.0,
                         Omega_D=Omega_D)
        pert = [(s, (1, 0, 0), fr * seed) for s, (fr, _, _) in enumerate(pops)]
        y = ds.consistent_fields(model, ds.maxwellian(model, [fr for fr, _, _ in pops], pert))
        out = ds.run(model, y, tmax, n_save=int(20 * tmax) + 1, rtol=1e-10, atol=1e-18)
        t, A = out["t"], np.abs(out["Fk"][:, 0, 0, 1, 0])
        big = np.nonzero(A > 1e-5 * beta)[0]
        t_end = t[big[0]] if len(big) else tmax
        t_start = t_end - 25.0
        fits = []
        for a, b in ((t_start, t_end), (t_start + 5, t_end), (t_start, t_end - 5)):
            m = (t >= a) & (t <= b)
            fits.append(np.polyfit(t[m], np.log(A[m]), 1)[0])
        ref = wd if dark else w0
        rows.append({"case": name, "model": "dark" if dark else "ordinary", "status": out["status"],
                     "k_es": k_es, "Nn": NnRun, "growth_fit": fits[0], "growth_window_spread": float(np.ptp(fits)),
                     "window": [t_start, t_end], "root": [ref.real, ref.imag], "root_residual": rd if dark else r0,
                     "growth_error": fits[0] - ref.imag,
                     "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
                     "max_gauss": [float(v) for v in out["gauss"].max(axis=0)],
                     "compile_time": out["compile_time"], "run_time": out["run_time"], "steps": out["num_steps"]})
        tag = f"{name}_{rows[-1]['model']}" + ("" if NnRun == Nn else f"_Nn{NnRun}")
        arrays[f"{tag}_t"], arrays[f"{tag}_A"] = t, A
        print(json.dumps(rows[-1]), flush=True)

out_dir.mkdir(parents=True, exist_ok=True)
import diffrax  # noqa: E402
import jax  # noqa: E402
import scipy  # noqa: E402
import spectrax  # noqa: E402
rec = {"case_id": "B02_B03", "command": "python examples/instabilities.py",
       "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
       "settings": {"beta": beta, "eta": eta, "Omega_D": Omega_D, "seed": seed, "Nx": 5, "rtol": 1e-10,
                    "atol": 1e-18, "solver": "Dopri8", "nu": 0.0,
                    "populations_es": {n: c[0] for n, c in cases.items()}},
       "versions": {"jax": jax.__version__, "diffrax": diffrax.__version__, "scipy": scipy.__version__,
                    "spectrax": spectrax.__version__},
       "results": rows}
(out_dir / "run.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
np.savez_compressed(out_dir / "run.npz", **arrays)

# Figure: python studies/figures.py growth  (shared plotting module, reads this record)
