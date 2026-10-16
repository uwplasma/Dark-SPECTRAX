"""B11 extension: collisional Landau damping (ordinary and dark, eta = 0.3) with the parent's Dougherty operator (#71).

Setup = B00 (examples/plasma.py): electrons, v_te = 0.1 c, fixed neutralizing background, density seed 1e-4,
constraint-consistent ordinary + static Yukawa dark fields, Omega_D = 1, Dopri8 rtol 1e-10 atol 1e-16, Nx = 8,
basis a = sqrt(2) v_te centred at 0, plus collision_model = "dougherty" with self collisions nu (omega_pe units).
k lambda_De in {0.3, 0.5}; nu in {0.01, 0.05, 0.2}; Nn in {128, 256}. Fit windows = B00 windows.

Independent references (NumPy/SciPy only, no parent code), both for the electron susceptibility chi(omega, k, nu)
in thermal units (v_t = 1, omega_pe = 1, k -> k lambda_De), linearized Dougherty about F0 = Maxwellian:
  C[df] = nu d_v[v df + d_v df - dU F0 + dtheta d_v F0], dU = int v df, dtheta = int v^2 df - dn.
R1 (Hermite matrix): normalized Hermite functions psi_n = He_n / sqrt(n!), M c = e_1 with
   M = -i omega I + i k X + nu diag(n [n >= 3]), X the sqrt(n) tridiagonal (rows 1, 2 have no collision: the
   restoring terms cancel -nu c_1 and -2 nu c_2 exactly); chi = -c_0 / (i k); N = 4000 and 8000 (banded solve).
R2 (continuum, Fourier in v): g(s) = int exp(-i s v) df dv obeys
   (k - nu s) g' = (-i omega + nu s^2) g + [i s (q/m) E + i nu s dU + nu s^2 dtheta] exp(-s^2/2);
   the solution regular at s* = k / nu is g(0) = -int_0^s* K(s) b(s) ds with
   K(s) b(s) = exp(k s / nu + (k^2/nu^2 - i omega / nu) log(1 - nu s / k)) / (k - nu s) * [...]  (Lenard-Bernstein
   kernel), and dU = (omega/k) g(0), dtheta = -g''(0) - g(0) with g''(0) from the ODE at s = 0. The 3x3 system
   (n, dU, dtheta) for unit (q/m) E gives chi = -n / (i k). Gauss-Legendre, 4000 nodes on [0, min(s*, 60)].
Dispersion: D_L = (Q - Omega_D^2)(1 + chi) + eta^2 Q chi, Q = omega^2 - k^2 (B00, eta = 0 ordinary), physical
k = k lambda_De / v_te.

Gates (declared before running; any failure is recorded, not tuned away):
G0 references: R2 at nu = 1e-6 equals the wofz chi at the collisionless root to 1e-6 relative; R1 (N = 4000)
   and R2 roots agree to 1e-9 relative; R1 N = 4000 vs 8000 roots agree to 1e-12 relative.
G1 fitted omega (Nn = 256) within 1e-3 relative of the R2 root, every (k, model, nu).
G2 Nn refinement: |omega_fit(128) - omega_fit(256)| / |omega| <= 1e-4.
G3 ledger (collisional work = 0 for self collisions): max |ledger_defect| <= 1e-15 (B00 records <= 1.8e-17), and
   directly, at 5 saved states, the Dougherty part of dCk/dt (RHS with minus without collisions) has rows n = 0, 1, 2
   below 1e-13 of its max over all rows (relative), i.e. it does no work, carries no momentum or density.
G4 every run: status success.

Run: python studies/b11_landau.py  -> studies/b11/landau.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
from scipy.optimize import root
from scipy.special import roots_legendre, wofz

import darkspectrax as ds

out_dir = Path(__file__).resolve().parent / "b11"
vte, eta, Omega_D, seed = 0.1, 0.3, 1.0, 1e-4
KL, NUS, NNS, NX = (0.3, 0.5), (0.01, 0.05, 0.2), (128, 256), 8
windows = {0.3: (8.0, 24.0), 0.5: (8.0, 14.0)}
guess0 = {0.3: 1.16 - 0.013j, 0.5: 1.42 - 0.15j}


def chi_wofz(w, k):
    z = w / (np.sqrt(2) * k)
    return (1 + z * 1j * np.sqrt(np.pi) * wofz(z)) / k ** 2


def chi_hermite(w, k, nu, N=4000):
    n = np.arange(N)
    off = 1j * k * np.sqrt(n[1:])
    ab = np.zeros((3, N), complex)
    ab[0, 1:], ab[1], ab[2, :-1] = off, -1j * w + nu * n * (n >= 3), off
    e = np.zeros(N, complex)
    e[1] = 1.0
    return -sl.solve_banded((1, 1), ab, e)[0] / (1j * k)


def chi_continuum(w, k, nu, M=4000):
    L = min(k / nu, 60.0)
    x, wt = roots_legendre(M)
    s, wt = 0.5 * L * (x + 1), 0.5 * L * wt
    K = np.exp(k * s / nu + (k ** 2 / nu ** 2 - 1j * w / nu) * np.log1p(-nu * s / k)) / (k - nu * s)
    GE, GU, GT = (-np.sum(wt * K * f) for f in (1j * s, nu * 1j * s, nu * s ** 2))
    A = np.array([[1, -GU, -GT], [-w / k, 1, 0], [1 - 1j * nu * w / k ** 2 - w ** 2 / k ** 2, 1j * nu / k, 1]], complex)
    n = np.linalg.solve(A, np.array([GE, 0, -1j / k]))[0]
    return -n / (1j * k)


def D_L(w, kl, eta_, chi):
    k = kl / vte
    Q = w ** 2 - k ** 2
    c = chi(w)
    return (Q - Omega_D ** 2) * (1 + c) + eta_ ** 2 * Q * c


def find_root(kl, eta_, chi, g):
    s = root(lambda x: [f(D_L(x[0] + 1j * x[1], kl, eta_, chi)) for f in (np.real, np.imag)], [g.real, g.imag], tol=1e-14)
    w = s.x[0] + 1j * s.x[1]
    return w, abs(D_L(w, kl, eta_, chi))


def references(kl, dark, nu, g):
    e_ = eta if dark else 0.0
    w2, r2 = find_root(kl, e_, lambda w: chi_continuum(w, kl, nu), g)
    w1, r1 = find_root(kl, e_, lambda w: chi_hermite(w, kl, nu), w2)
    w1b, _ = find_root(kl, e_, lambda w: chi_hermite(w, kl, nu, 8000), w1)
    return w2, {"R2_continuum": [w2.real, w2.imag], "R1_hermite_N4000": [w1.real, w1.imag],
                "R1_hermite_N8000": [w1b.real, w1b.imag], "residuals": [r2, r1],
                "R1_vs_R2": abs(w1 - w2) / abs(w2), "R1_N4000_vs_8000": abs(w1 - w1b) / abs(w1b)}


def simulate(kl, dark, nu, Nn):
    k = kl / vte
    model = ds.Model(Nx=NX, Nn=Nn, Lx=2 * np.pi / k, alpha_s=(np.sqrt(2) * vte,) * 3, rho_background=1.0,
                     mode="self_consistent" if dark else "ordinary", eta=eta if dark else 0.0, Omega_D=Omega_D,
                     collision_model="dougherty", nu_dougherty=nu)
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0], [(0, (1, 0, 0), seed / 2)]))
    T = windows[kl][1]
    return model, ds.run(model, y, T, n_save=int(40 * T) + 1, rtol=1e-10, atol=1e-16)


def collision_rows(model, out, Nn):
    """Max over 5 saved states of |Dougherty dCk rows 0-2| / max |Dougherty dCk|."""
    free = ds.Model(**{**{f: getattr(model, f) for f in ("Nx", "Nn", "Lx", "alpha_s", "rho_background", "mode", "eta",
                                                        "Omega_D")}})
    worst = 0.0
    for i in np.linspace(0, out["t"].size - 1, 5).astype(int):
        y = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex)}
        d = np.asarray(ds.rhs(out["t"][i], y, model)["Ck"]) - np.asarray(ds.rhs(out["t"][i], y, free)["Ck"])
        d = d.reshape(Nn, -1)
        worst = max(worst, float(np.abs(d[:3]).max() / max(np.abs(d).max(), 1e-300)))
    return worst


if __name__ == "__main__":
    t0 = time.time()
    rows, refs, g0 = [], [], []
    for kl in KL:
        w0 = find_root(kl, 0.0, lambda w: chi_wofz(w, kl), guess0[kl])[0]
        g0.append({"k_lambda_De": kl, "collisionless_root": [w0.real, w0.imag],
                   "chi_R2_nu1e-6_vs_wofz": abs(chi_continuum(w0, kl, 1e-6) / chi_wofz(w0, kl) - 1)})
        for dark in (False, True):
            g = w0 if not dark else find_root(kl, eta, lambda w: chi_wofz(w, kl), w0)[0]
            wp = find_root(kl, eta, lambda w: chi_wofz(w, kl), np.sqrt((kl / vte) ** 2 + Omega_D ** 2) + 0j)[0]
            for nu in NUS:
                ref, info = references(kl, dark, nu, g)
                g = ref
                info.update({"k_lambda_De": kl, "model": "dark" if dark else "ordinary", "nu": nu})
                refs.append(info)
                fits = {}
                for Nn in NNS:
                    model, out = simulate(kl, dark, nu, Nn)
                    t, z = out["t"], out["Fk"][:, 0, 0, 1, 0]
                    guesses = [ref, -np.conj(ref)] + ([wp, -np.conj(wp)] if dark else [])
                    sel = t >= windows[kl][0]
                    w, res = ds.fit_modes(t[sel], z[sel], guesses)
                    fits[Nn] = w[0]
                    rows.append({"k_lambda_De": kl, "model": info["model"], "nu": nu, "Nn": Nn, "Nx": NX,
                                 "status": out["status"], "omega_fit": [w[0].real, w[0].imag],
                                 "omega_ref": [ref.real, ref.imag], "rel_error": abs(w[0] - ref) / abs(ref),
                                 "fit_residual": res, "fit_window": windows[kl],
                                 "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
                                 "collision_rows_0_2_rel": collision_rows(model, out, Nn),
                                 "steps": out["num_steps"], "run_time": out["run_time"]})
                    print(json.dumps(rows[-1], default=float), flush=True)
                rows[-1]["Nn_refinement"] = abs(fits[128] - fits[256]) / abs(fits[256])
    gates = {
        "G0": all(r["chi_R2_nu1e-6_vs_wofz"] <= 1e-6 for r in g0)
        and all(r["R1_vs_R2"] <= 1e-9 and r["R1_N4000_vs_8000"] <= 1e-12 for r in refs),
        "G1": all(r["rel_error"] <= 1e-3 for r in rows if r["Nn"] == 256),
        "G2": all(r["Nn_refinement"] <= 1e-4 for r in rows if "Nn_refinement" in r),
        "G3": all(r["max_ledger_defect"] <= 1e-15 and r["collision_rows_0_2_rel"] <= 1e-13 for r in rows),
        "G4": all(r["status"] == "success" for r in rows)}
    import diffrax
    import jax
    import scipy
    import spectrax
    rec = {"case_id": "B11-landau", "command": "python studies/b11_landau.py " + " ".join(sys.argv[1:]),
           "repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
           "units": "parent normalization: c = 1, omega_pe = 1; nu in omega_pe",
           "settings": {"vte": vte, "eta": eta, "Omega_D": Omega_D, "seed": seed, "Nx": NX, "fit_windows": windows,
                        "rtol": 1e-10, "atol": 1e-16, "solver": "Dopri8"},
           "versions": {"jax": jax.__version__, "diffrax": diffrax.__version__, "scipy": scipy.__version__,
                        "spectrax": spectrax.__version__},
           "wall_time": time.time() - t0, "gates": gates, "passed": all(gates.values()),
           "reference_checks": g0, "references": refs, "results": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landau.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(json.dumps(gates))
