"""B00: linear Landau damping, ordinary versus dark, against an independent kinetic root.

Electrons with thermal standard deviation v_te = 0.1 c on fixed neutralizing ions,
omega_pe = 1, a 1e-4 density seed with constraint-consistent ordinary and Yukawa
dark fields.  The reference solves D_L = (Q - Omega_D^2)(1 + chi) + eta^2 Q chi = 0,
Q = omega^2 - c^2 k^2, with the plasma dispersion function from scipy's wofz.

Run:  python examples/plasma.py            (short: Nn = 64, 128; Nx = 5, 8)
Writes docs/_static/b00/{run.json,run.npz,figure.png}.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

import darkspectrax as ds

out_dir = Path(__file__).resolve().parents[1] / "docs" / "_static" / "b00"
k_lambdas = (0.3, 0.5)
vte, eta, Omega_D, seed = 0.1, 0.3, 1.0, 1e-4
hermite_orders = (64, 128)
grids = (5, 8)
# Fit windows start after the initial ballistic transient and end before the Hermite
# front n ~ (k v_te t)^2 reaches the smallest order (64): (0.3*24)^2 = 52, (0.5*14)^2 = 49.
windows = {0.3: (8.0, 24.0), 0.5: (8.0, 14.0)}
scan_starts = (2.0, 4.0, 6.0, 8.0)


def chi(w, k):
    zeta = w / (np.sqrt(2) * k * vte)
    Z = 1j * np.sqrt(np.pi) * wofz(zeta)
    return (1 + zeta * Z) / (k * vte) ** 2


def D_L(w, k, eta_):
    Q = w ** 2 - k ** 2
    return (Q - Omega_D ** 2) * (1 + chi(w, k)) + eta_ ** 2 * Q * chi(w, k)


def kinetic_root(k, eta_, guess):
    def f(x):
        d = D_L(x[0] + 1j * x[1], k, eta_)
        return [d.real, d.imag]

    s = root(f, [guess.real, guess.imag], tol=1e-14)
    w = s.x[0] + 1j * s.x[1]
    return w, abs(D_L(w, k, eta_))


def simulate(kl, dark, Nn, Nx):
    k = kl / vte
    model = ds.Model(Nx=Nx, Nn=Nn, Lx=2 * np.pi / k, alpha_s=(np.sqrt(2) * vte,) * 3,
                     rho_background=1.0, mode="self_consistent" if dark else "ordinary",
                     eta=eta if dark else 0.0, Omega_D=Omega_D)
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), seed / 2)]))
    return model, ds.run(model, y, windows[kl][1], n_save=int(40 * windows[kl][1]) + 1, rtol=1e-10, atol=1e-16)


rows, arrays = [], {}
for kl in k_lambdas:
    k = kl / vte
    w0, r0 = kinetic_root(k, 0.0, 1.2 - 0.05j)
    wd, rd = kinetic_root(k, eta, w0)
    wp, rp = kinetic_root(k, eta, np.sqrt(k ** 2 + Omega_D ** 2) + 0j)  # massive longitudinal branch
    for dark in (False, True):
        for Nn in hermite_orders:
            for Nx in grids:
                model, out = simulate(kl, dark, Nn, Nx)
                t, z = out["t"], out["Fk"][:, 0, 0, 1, 0]
                guesses = [w0, -np.conj(w0)] + ([wp, -np.conj(wp)] if dark else [])
                ref = wd if dark else w0
                scan = {}
                for t1 in scan_starts:
                    ws, rs = ds.fit_modes(t[t >= t1], z[t >= t1], guesses)
                    scan[t1] = [ws[0].real, ws[0].imag, abs(ws[0] - ref) / abs(ref), rs]
                w, res = ds.fit_modes(t[t >= windows[kl][0]], z[t >= windows[kl][0]], guesses)
                rows.append({
                    "k_lambda_De": kl, "model": "dark" if dark else "ordinary", "Nn": Nn, "Nx": Nx,
                    "status": out["status"], "omega_fit": [w[0].real, w[0].imag],
                    "omega_root": [ref.real, ref.imag],
                    "rel_error": abs(w[0] - ref) / abs(ref), "fit_residual": res,
                    "fit_window": windows[kl], "window_start_scan": scan,
                    "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
                    "max_gauss": [float(v) for v in out["gauss"].max(axis=0)],
                    "steps": out["num_steps"], "compile_time": out["compile_time"], "run_time": out["run_time"],
                })
                tag = f"k{kl}_{rows[-1]['model']}_Nn{Nn}_Nx{Nx}"
                arrays[f"t_{tag}"], arrays[f"Ek_{tag}"] = t, z
                arrays[f"W_{tag}"] = out["W"]
                print(json.dumps(rows[-1]), flush=True)
    rows.append({"k_lambda_De": kl, "roots": {
        "ordinary": [w0.real, w0.imag], "dark": [wd.real, wd.imag],
        "massive_branch": [wp.real, wp.imag], "residuals": [r0, rd, rp]}})

out_dir.mkdir(parents=True, exist_ok=True)
rec = {"case_id": "B00", "command": "python examples/plasma.py " + " ".join(sys.argv[1:]),
       "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
       "units": "parent normalization: c = 1, omega_pe = 1, Omega_cs[0] = 1",
       "settings": {"vte": vte, "eta": eta, "Omega_D": Omega_D, "seed": seed, "fit_windows": windows,
                    "rtol": 1e-10, "atol": 1e-16, "solver": "Dopri8", "nu": 0.0,
                    "initial_fields": "ordinary Gauss field + static Yukawa dark near field"},
       "results": rows}
import diffrax  # noqa: E402
import jax  # noqa: E402
import scipy  # noqa: E402
import spectrax  # noqa: E402
rec["versions"] = {"jax": jax.__version__, "diffrax": diffrax.__version__,
                   "scipy": scipy.__version__, "spectrax": spectrax.__version__}
(out_dir / "run.json").write_text(json.dumps(rec, indent=2) + "\n")
np.savez_compressed(out_dir / "run.npz", **arrays)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for ax, kl in zip(axes, k_lambdas):
    root_row = [r for r in rows if r.get("roots") and r["k_lambda_De"] == kl][0]["roots"]
    for name, color in (("ordinary", "tab:blue"), ("dark", "tab:red")):
        tag = f"k{kl}_{name}_Nn{hermite_orders[-1]}_Nx{grids[-1]}"
        t, z = arrays[f"t_{tag}"], arrays[f"Ek_{tag}"]
        ax.semilogy(t, np.abs(z.imag) + 1e-30, color=color, lw=1, label=f"{name} run")
        g = root_row[name][1]
        ax.semilogy(t, np.abs(z[0]) * np.exp(g * t), "--", color=color, lw=1,
                    label=f"{name} root decay, $\\gamma$={g:.4f}")
    ax.set_title(f"$k\\lambda_{{De}}$ = {kl}  ($\\eta$={eta}, $\\Omega_D$={Omega_D}$\\omega_{{pe}}$)")
    ax.set_xlabel("$\\omega_{pe} t$")
    ax.set_ylabel("|Im $E_{x,k}$| (density seed $10^{-4}$, parent units)")
    ax.axvspan(*windows[kl], color="0.9", zorder=-1)
    ax.legend(fontsize=8, loc="lower left")
fig.savefig(out_dir / "figure.png", dpi=130)
