"""C05: matched physical Landau case against Dark-JAX-in-Cell PIC records.

The PIC side is NOT rerun here. Its published records (Dark-JAX-in-Cell commit
d54757962f3b9f7ec18eb981804ef59f6db6076c, docs/_static/figures/physical_kinetic_{32,64,128},
MIT license, produced at its commit 9e99dcb on a GPU) are summarized in
studies/djic_records/ by `--extract PATH_TO_DJIC_CHECKOUT`.

Same physical inputs: Maxwellian electrons with standard deviation 0.05 c, fixed neutralizing ions,
k lambda_De = 0.5, eta = 0.3, Omega_D = k c (= 10 omega_pe), density seed 0.01 (PIC: displacement
0.01/k), Yukawa-consistent dark start, longitudinal 1D. Same observables: E_x at mode 1 of the
ordinary-only run and of the mixed run, damping fitted from neighbouring-sample maxima on
2 <= omega_pe t <= 12 (the PIC rule), plus a complex least-squares fit.

Shared between the codes: the Maxwell-Proca convention, the D_L reference equation and the
work definition P_D = eta J.E_D (same group). Not shared: the kinetic discretization (PIC with
quadratic shapes and Boris push vs Hermite-Fourier), field solver, time integrator, loading.

Run: python studies/c05_dark_jax_in_cell.py   -> studies/c05/{run.json, run.npz, figure.png}
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import wofz

import darkspectrax as ds

HERE = Path(__file__).resolve().parent
REC = HERE / "djic_records"
DJIC_COMMIT = "d54757962f3b9f7ec18eb981804ef59f6db6076c"

if len(sys.argv) == 3 and sys.argv[1] == "--extract":
    src = Path(sys.argv[2]) / "docs/_static/figures"
    REC.mkdir(exist_ok=True)
    for cells in (32, 64, 128):
        d = src / f"physical_kinetic_{cells}"
        rec = json.loads((d / "run.json").read_text())
        arr = np.load(d / "data.npz")
        keep = {"settings": rec["settings"], "results": rec["results"], "producer_git": rec.get("git"),
                "versions": {k: rec.get(k) for k in ("jaxincell", "jax", "backend", "platform")},
                "source": f"uwplasma/Dark-JAX-in-Cell@{DJIC_COMMIT}:docs/_static/figures/physical_kinetic_{cells}"}
        (REC / f"physical_kinetic_{cells}.json").write_text(json.dumps(keep, indent=1, default=float) + "\n")
        s = slice(None, None, 4)
        np.savez_compressed(REC / f"physical_kinetic_{cells}.npz", t=arr["t"][s],
                            parent_Ek=arr["parent_Ek"][s], ordinary_Ek=arr["ordinary_Ek"][s])
    raise SystemExit(0)

sigma, kl, eta, seed = 0.05, 0.5, 0.3, 0.01
k = kl / sigma
Omega_D = k  # = k c


def D_L(w, eta_):
    z = w / (np.sqrt(2) * k * sigma)
    chi = (1 + z * 1j * np.sqrt(np.pi) * wofz(z)) / (k * sigma) ** 2
    Q = w ** 2 - k ** 2
    return (Q - Omega_D ** 2) * (1 + chi) + eta_ ** 2 * Q * chi


def kroot(eta_, g):
    s = root(lambda x: [D_L(x[0] + 1j * x[1], eta_).real, D_L(x[0] + 1j * x[1], eta_).imag], [g.real, g.imag],
             tol=1e-14)
    return s.x[0] + 1j * s.x[1]


def maxima_fit(t, A, window=(2.0, 12.0)):
    """Dark-JAX-in-Cell rule: strict neighbouring-sample maxima, log-linear slope, mean spacing."""
    i = np.nonzero((A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]))[0] + 1
    i = i[(t[i] >= window[0]) & (t[i] <= window[1])]
    gamma = np.polyfit(t[i], np.log(A[i]), 1)[0]
    return np.pi / np.mean(np.diff(t[i])), gamma, int(i.size)


w_ord, w_dark = kroot(0.0, 1.4 - 0.15j), kroot(eta, 1.43 - 0.146j)
rec = {"case_id": "C05", "command": "python studies/c05_dark_jax_in_cell.py",
       "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
       "dark_jax_in_cell_commit": DJIC_COMMIT,
       "physical_inputs": {"sigma_over_c": sigma, "k_lambda_De": kl, "eta": eta, "Omega_D_over_wp": Omega_D,
                           "density_seed": seed},
       "roots": {"ordinary": [w_ord.real, w_ord.imag], "dark": [w_dark.real, w_dark.imag]},
       "spectrax": {}, "pic": {}}
arrays = {}
for mode in ("ordinary", "self_consistent"):
    for Nn in (128, 256):
        model = ds.Model(Nx=5, Nn=Nn, Lx=2 * np.pi / k, alpha_s=(np.sqrt(2) * sigma,) * 3, rho_background=1.0,
                         mode=mode, eta=eta if mode != "ordinary" else 0.0, Omega_D=Omega_D)
        y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), -seed / 2)]))
        out = ds.run(model, y, 29.45, n_save=2946, rtol=1e-10, atol=1e-14)
        t, E = out["t"], out["Fk"][:, 0, 0, 1, 0]
        fw, fg, nmax = maxima_fit(t, np.abs(E))
        ref = w_ord if mode == "ordinary" else w_dark
        wfit, res = ds.fit_modes(t[(t >= 2) & (t <= 12)], E[(t >= 2) & (t <= 12)], [ref, -np.conj(ref)])
        key = f"{mode}_Nn{Nn}"
        rec["spectrax"][key] = {"status": out["status"], "maxima_fit": [fw, fg], "maxima_used": nmax,
                                "complex_fit": [wfit[0].real, wfit[0].imag], "complex_fit_resid": res,
                                "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
                                "run_time": out["run_time"], "compile_time": out["compile_time"]}
        arrays[f"{key}_t"], arrays[f"{key}_E"] = t, E
        print(key, rec["spectrax"][key], flush=True)
for cells in (32, 64, 128):
    r = json.loads((REC / f"physical_kinetic_{cells}.json").read_text())
    res_ = r["results"]
    rec["pic"][f"cells{cells}"] = {
        "particles": r["settings"]["particles"], "steps": r["settings"]["steps"],
        "dark": [res_["measured_real_over_wp"], res_["measured_imag_over_wp"]],
        "ordinary": [res_["parent_measured_real_over_wp"], res_["parent_measured_imag_over_wp"]],
        "damping_slope_stderr": res_["damping_slope_stderr"],
        "djic_root_dark": [res_["root_real_over_wp"], res_["root_imag_over_wp"]],
        "djic_root_ordinary": [res_["parent_root_real_over_wp"], res_["parent_root_imag_over_wp"]]}
    a = np.load(REC / f"physical_kinetic_{cells}.npz")
    for name, sx in (("parent_Ek", "ordinary_Nn256"), ("ordinary_Ek", "self_consistent_Nn256")):
        tp, Ap = a["t"], np.abs(a[name]) / np.abs(a[name][0])
        ts, As = arrays[f"{sx}_t"], np.abs(arrays[f"{sx}_E"]) / np.abs(arrays[f"{sx}_E"][0])
        m = (tp >= 2) & (tp <= 12)
        rec["pic"][f"cells{cells}"][f"rms_log_amplitude_diff_{name}_2_12"] = float(
            np.sqrt(np.mean((np.log(Ap[m]) - np.log(np.interp(tp[m], ts, As))) ** 2)))
rec["root_cross_check"] = {"dark_abs_diff": abs(w_dark - complex(*rec["pic"]["cells128"]["djic_root_dark"])),
                           "ordinary_abs_diff": abs(w_ord - complex(*rec["pic"]["cells128"]["djic_root_ordinary"]))}
d = HERE / "c05"
d.mkdir(exist_ok=True)
(d / "run.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
np.savez_compressed(d / "run.npz", **arrays)
print(json.dumps({"roots": rec["roots"], "pic": rec["pic"], "cross": rec["root_cross_check"]}, default=float))
