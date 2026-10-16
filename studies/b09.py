"""B09: Weibel (bi-Maxwellian) and current-filamentation (counter-streaming beams) instabilities in 1D3V,
ordinary and dark, linear growth and first transverse nonlinear test.

Parent units: c = 1, electrons q = -1, Omega_cs = 1, fixed neutralizing ion background, total density 1
(omega_pe = 1). Thermal speeds are the parent basis widths a (f ~ exp(-v^2/a^2), a = sqrt(2) v_t). k along x;
the unstable mode is transverse (E_y, B_z); with no drift along x and (for the beams) equal and opposite y drifts,
it decouples exactly from E_x and from the (E_z, B_y) polarization.

Cases
- weibel: one electron species, (a_x, a_y, a_z) = (0.1, 0.3, 0.1) (T_y/T_x = 9 along k = x).
- filament: two electron beams n = 0.5 each, u_y = +-0.3, isotropic v_t = 0.1 (a = 0.1 sqrt2): the setup of
  Juno's thesis (arXiv:2005.13539, Sec. 4.2.6, Eq. 4.40: n0 = 0.5, u = 0.3c, v_the = 0.1c), reduced to the pure
  filamentation geometry theta = 0 (k perpendicular to the drift), 1X3V here instead of 2X2V.
- models: ordinary; dark eta = 0.3 with Omega_D = 0.3 (near the transverse Proca branch w^2 = k^2 + Omega_D^2 at the
  unstable k, where the dark coupling factor below is strongly k dependent) and Omega_D = 1.2 (the B08 value).

Independent kinetic dispersion relation (derived here, not from the Hermite code). For each species s with density
n_s, widths a_x,s, a_y,s and drift u_y,s, the transverse response is J-term Pi(w) E_force with
    Pi(w) = sum_s n_s [1 - R_s (1 + xi_s Z(xi_s))],  R_s = (a_y,s^2 + 2 u_y,s^2)/a_x,s^2,  xi_s = w/(k a_x,s),
Z(xi) = i sqrt(pi) wofz(xi) (Im w > 0). E_force = E + eta E_D. Ordinary: w^2 - k^2 - Pi = 0. Dark (Proca, transverse
phi_D = 0; the dark field obeys the same equation with the mass term and the source times eta):
    (w^2 - k^2 - Pi) E - eta Pi E_D = 0,   -eta Pi E + (w^2 - k^2 - Omega_D^2 - eta^2 Pi) E_D = 0,
whose determinant is D(w) = (w^2 - k^2)(w^2 - k^2 - Omega_D^2) - Pi [(1 + eta^2)(w^2 - k^2) - Omega_D^2] = 0.
Roots: imaginary-axis scan for the largest growing root at the first k, then complex secant continuation in k; the
scan is repeated at every k and the two must agree (purely growing roots, |Re w| <= 1e-10). The smallest singular
value of the 2x2 matrix at the root is recorded.

Part 1 (`python studies/b09.py linear`): k in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4), box Lx = 2 pi/k, seed B_z(k) = 1e-10.
Levels (Nx, Nn, Nm, Np): base (5, 32, 4, 1), velocity-refined (5, 64, 8, 1), space-refined (9, 32, 4, 1)
(Nm = 2 is exact for the linear transverse mode; Nm 4 -> 8 is part of the refinement). Fit window: explicit,
[t_e/2, t_e] with t_e = ln(1e6)/gamma_root (the second half of 13.8 e-folds; amplitude <= 1e-4, linear), fit of
B_z(k, t) with ds.fit_modes and guesses {i gamma_root, +-sqrt(k^2 + 1)} plus, dark, {+-sqrt(k^2 + Omega_D^2 + eta^2)}.
Gates (declared before running):
  G1  |gamma_fit - gamma_root| / gamma_root <= 1e-3 at every level, |Re w_fit| <= 1e-3 gamma_root;
      refinement: |gamma(base) - gamma(refined)| / gamma_root <= 1e-4 for both refinements.
  G2  ledger: every row of |ledger_defect| <= 1e-6 of the final perturbation field energy
      (U_gamma + U_D at t_e); Gauss residuals (ordinary, dark) <= 1e-10 of the final max |F_k|.
Part 1b (`python studies/b09.py ladder`, added after Part 1 showed Nn = 64 unconverged at k >= 1.6, declared before
running it): at small xi = gamma/(k a_x) the resonant response has width gamma/k << a_x and Hermite convergence is slow.
k in (1.6, 2.0, 2.4), (Nx, Nm) = (5, 4), Nn in (128, 256, 512), same window, guesses and fit; the reported growth is the
fitted mode nearest i gamma_root (Part 1 took the largest Im w, which picked a spurious fitted mode in one run).
Gate: |gamma_fit - gamma_root|/gamma_root <= 1e-3 at the finest Nn and |gamma(Nn) - gamma(Nn/2)|/gamma_root <= 1e-4
between the two finest levels; ledger and Gauss as G2. The Nn at which 1e-3 is first met is recorded.
Part 2 (`python studies/b09.py nonlinear`): k0 = 1.2 (near the fastest-growing k for both cases), Lx = 2 pi/k0,
seed B_z(k0) = 1e-6, run to t = 260 (weibel) / 180 (filament), 521 / 361 saves. Levels (Nx, Nn, Nm, Np):
base (16, 32, 16, 1), velocity-doubled (16, 64, 32, 1), space-doubled (32, 32, 16, 1).
Magnetic bounce estimate (Davidson, Hammer, Haber and Wagner 1972): saturation when the magnetic trapping frequency
w_B = sqrt(k v_perp |q| B / m) reaches gamma, B_sat = gamma^2 / (k v_perp), with v_perp = a_y/sqrt2 (weibel,
the thermal y speed) or u (filament); gamma = the kinetic root at k0 for the model (dark: B the force field
B_z + eta B_D,z). Measured: the amplitude 2 |B_force,z(k0)| at the first maximum of the magnetic energy
U_B = Om0^2/2 <B^2> (+ U_BD for dark; reported separately).
Gates:
  G3a resolution: resolved time T_res = first t where |U_B(base) - U_B(refined)| > 2e-2 max U_B(base) for either
      refinement; the saturation peak must lie before T_res, and the peak amplitude must agree within 2e-2 between
      base and both refinements. No claim is made past T_res.
  G3b B_peak / B_sat in [1/3, 3] (the bounce estimate is order-unity, not a precise prediction), ordinary and dark;
      the dark/ordinary ratio of B_peak is compared to the ratio of the estimates.
  G2  as above, ledger <= 1e-6 of max field perturbation energy, Gauss <= 1e-8 of max |F_k| (rho at 2 k0 is
      generated nonlinearly by the v_y B_z force).

Run: python studies/b09.py linear|ladder|nonlinear [quick]  -> studies/b09/{linear,ladder,nonlinear}.json
"""

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import newton
from scipy.special import wofz

import darkspectrax as ds
from darkspectrax._model import inner

out_dir = Path(__file__).resolve().parent / "b09"
eta = 0.3
CASES = {
    # species: (density, (a_x, a_y, a_z), u_y)
    "weibel": [(1.0, (0.1, 0.3, 0.1), 0.0)],
    "filament": [(0.5, (0.1 * np.sqrt(2),) * 3, 0.3), (0.5, (0.1 * np.sqrt(2),) * 3, -0.3)],
}
MODELS = {"ordinary": None, "dark_OD0.3": 0.3, "dark_OD1.2": 1.2}


# ------------------------------------------------------------------------------------- dispersion relation
def Zfun(x):
    return 1j * np.sqrt(np.pi) * wofz(x)


def Pi(w, k, sp):
    return sum(n * (1 - (a[1] ** 2 + 2 * u ** 2) / a[0] ** 2 * (1 + w / (k * a[0]) * Zfun(w / (k * a[0]))))
               for n, a, u in sp)


def D(w, k, sp, OD):
    P = Pi(w, k, sp)
    if OD is None:
        return w * w - k * k - P
    return (w * w - k * k) * (w * w - k * k - OD ** 2) - P * ((1 + eta ** 2) * (w * w - k * k) - OD ** 2)


def matrix_sv(w, k, sp, OD):
    P = Pi(w, k, sp)
    M = np.array([[w * w - k * k - P, -eta * P], [-eta * P, w * w - k * k - OD ** 2 - eta ** 2 * P]])
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[-1] / s[0])


def scan_root(k, sp, OD):
    g = np.linspace(1e-5, 1.0, 100001)
    v = np.array([D(1j * x, k, sp, OD).real for x in g])
    i = np.where(np.diff(np.sign(v)))[0]
    if not len(i):
        return None
    j = i[-1]
    return 1j * newton(lambda x: D(1j * x, k, sp, OD).real, g[j], tol=1e-15, maxiter=100)


def roots(ks, sp, OD):
    out, prev = [], None
    for k in ks:
        ws = scan_root(k, sp, OD)
        wc = ws if prev is None else newton(lambda w: D(w, k, sp, OD), prev, tol=1e-15, maxiter=200)
        wc = complex(wc)
        out.append({"k": k, "omega": [wc.real, wc.imag], "scan_gamma": float(ws.imag),
                    "continuation_vs_scan_rel": abs(wc - ws) / abs(ws), "abs_D": float(abs(D(wc, k, sp, OD))),
                    "matrix_sv": matrix_sv(wc, k, sp, OD if OD is not None else 1.0) if OD is not None else None})
        prev = wc
    return out


# ------------------------------------------------------------------------------------------------ runs
def model_for(case, OD, k, Nx, Nn, Nm, Np=1):
    sp = CASES[case]
    return ds.Model(Nx=Nx, Nn=Nn, Nm=Nm, Np=Np, Lx=2 * np.pi / k, qs=(-1.0,) * len(sp), Omega_cs=(1.0,) * len(sp),
                    alpha_s=tuple(x for _, a, _ in sp for x in a), u_s=tuple(x for _, _, u in sp for x in (0.0, u, 0.0)),
                    rho_background=1.0, mode="ordinary" if OD is None else "self_consistent",
                    eta=0.0 if OD is None else eta, Omega_D=1.0 if OD is None else OD)


def seeded(model, case, amp):
    y = ds.maxwellian(model, [n for n, _, _ in CASES[case]])
    return {**y, "Fk": y["Fk"].at[5, 0, 1, 0].set(amp)}


def field_energy_series(model, out):
    UB = np.array([float(0.5 * inner(model.Nx, jnp.asarray(f[3:6]), jnp.asarray(f[3:6]))) for f in out["Fk"]])
    UBD = np.array([float(0.5 * inner(model.Nx, jnp.asarray(d[3:6]), jnp.asarray(d[3:6]))) for d in out["Dk"]])
    return UB, UBD


def common(out):
    return {"status": out["status"], "failure_reason": out["failure_reason"], "steps": int(out["num_steps"]),
            "compile_time": float(out["compile_time"]), "run_time": float(out["run_time"])}


def linear(quick, ladder=False):
    ks = (1.2,) if quick else (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
    levels = ((5, 32, 4),) if quick else ((5, 32, 4), (5, 64, 8), (9, 32, 4))
    if ladder:
        ks, levels = ((2.4,), ((5, 128, 4),)) if quick else ((1.6, 2.0, 2.4), ((5, 128, 4), (5, 256, 4), (5, 512, 4)))
    ref, rows = {}, []
    for case, sp in CASES.items():
        for mname, OD in MODELS.items():
            rr = roots(ks, sp, OD)
            ref[f"{case}/{mname}"] = rr
            for r in rr:
                k, g = r["k"], r["omega"][1]
                te = np.log(1e6) / g
                guesses = [1j * g, np.sqrt(k * k + 1), -np.sqrt(k * k + 1)]
                if OD is not None:
                    wd = np.sqrt(k * k + OD ** 2 + eta ** 2)
                    guesses += [wd, -wd]
                for Nx, Nn, Nm in levels:
                    model = model_for(case, OD, k, Nx, Nn, Nm)
                    out = ds.run(model, seeded(model, case, 1e-10), te, n_save=801, rtol=1e-11, atol=1e-20)
                    t = out["t"]
                    z = out["Fk"][:, 5, 0, 1, 0] + (eta * out["Dk"][:, 5, 0, 1, 0] if OD is not None else 0)
                    win = t >= te / 2
                    wf, res = ds.fit_modes(t[win], z[win], guesses)
                    w0 = complex(wf[int(np.argmin(np.abs(wf - 1j * g)))] if ladder else wf[int(np.argmax(wf.imag))])
                    UB, UBD = field_energy_series(model, out)
                    Uw = out["U_gamma"][-1] + out["U_D"][-1]
                    Fmax = float(max(np.abs(out["Fk"][-1]).max(), np.abs(out["Dk"][-1]).max()))
                    rows.append({"case": case, "model": mname, "Omega_D": OD, "k": k, "Nx": Nx, "Nn": Nn, "Nm": Nm,
                                 "gamma_root": g, "omega_fit": [w0.real, w0.imag],
                                 "rel_error": abs(w0.imag - g) / g, "re_over_gamma": abs(w0.real) / g,
                                 "fit_residual": res, "window": [te / 2, te], "U_field_final": float(Uw),
                                 "UB_final": float(UB[-1]), "UBD_final": float(UBD[-1]),
                                 "ledger_rel": [float(v) for v in np.abs(out["ledger_defect"]).max(axis=1) / Uw],
                                 "gauss_rel": [float(v) for v in out["gauss"].max(axis=0) / Fmax], **common(out)})
                    q = rows[-1]
                    print(f"{case:8s} {mname:10s} k={k} Nx={Nx} Nn={Nn} Nm={Nm} g={g:.6f} fit={w0.real:+.1e}"
                          f"{w0.imag:+.6f}i err={q['rel_error']:.1e} res={res:.1e} "
                          f"ledger={max(q['ledger_rel']):.1e} gauss={max(q['gauss_rel']):.1e} {q['status']} "
                          f"{q['run_time']:.0f}s", flush=True)
    gates = {"rel_error": 1e-3, "re_over_gamma": 1e-3, "refinement": 1e-4, "ledger_rel": 1e-6, "gauss_rel": 1e-10}
    for r in rows:
        same = [b for b in rows if b["case"] == r["case"] and b["model"] == r["model"] and b["k"] == r["k"]]
        base = [b for b in same if (b["Nx"], b["Nn"], b["Nm"]) == levels[0]][0]
        if ladder:  # change from the next-coarser Nn
            base = ([b for b in same if b["Nn"] == r["Nn"] // 2] or [r])[0]
        r["refinement_change"] = abs(r["omega_fit"][1] - base["omega_fit"][1]) / r["gamma_root"]
        r["passed"] = bool(r["status"] == "success" and r["rel_error"] <= gates["rel_error"]
                           and r["re_over_gamma"] <= gates["re_over_gamma"]
                           and r["refinement_change"] <= gates["refinement"]
                           and max(r["ledger_rel"]) <= gates["ledger_rel"] and max(r["gauss_rel"]) <= gates["gauss_rel"])
    if ladder:
        for r in rows:
            r["passed"] = None
        top = max(lv[1] for lv in levels)
        for r in rows:
            if r["Nn"] == top:
                r["passed"] = bool(r["status"] == "success" and r["rel_error"] <= gates["rel_error"]
                                   and r["refinement_change"] <= gates["refinement"]
                                   and max(r["ledger_rel"]) <= gates["ledger_rel"]
                                   and max(r["gauss_rel"]) <= gates["gauss_rel"])
    return {"gates": gates, "passed": all(r["passed"] for r in rows if r["passed"] is not None),
            "settings": {"eta": eta, "cases": {c: [[n, list(a), u] for n, a, u in sp] for c, sp in CASES.items()},
                         "Omega_D": {m: od for m, od in MODELS.items()}, "seed_Bz": 1e-10, "rtol": 1e-11,
                         "atol": 1e-20, "levels_Nx_Nn_Nm": [list(x) for x in levels], "Np": 1, "n_save": 801},
            "roots": ref, "results": rows}


def nonlinear(quick):
    k0 = 1.2
    levels = ((16, 32, 16),) if quick else ((16, 32, 16), (16, 64, 32), (32, 32, 16))
    tmax = {"weibel": 260.0, "filament": 180.0}
    rows, groups = [], []
    for case, sp in CASES.items():
        for mname, OD in MODELS.items():
            g = roots((k0,), sp, OD)[0]["omega"][1]
            vperp = sp[0][1][1] / np.sqrt(2) if case == "weibel" else abs(sp[0][2])
            Bsat = g * g / (k0 * vperp)
            runs = []
            for Nx, Nn, Nm in levels:
                model = model_for(case, OD, k0, Nx, Nn, Nm)
                T = tmax[case] / (4 if quick else 1)
                ns = int(2 * T) + 1
                out = ds.run(model, seeded(model, case, 1e-6), T, n_save=ns, rtol=1e-10, atol=1e-16,
                             max_steps=2_000_000)
                t = out["t"]
                UB, UBD = field_energy_series(model, out)
                Bf = out["Fk"][:, 5, 0, 1, 0] + (eta * out["Dk"][:, 5, 0, 1, 0] if OD is not None else 0)
                ipk = next((i for i in range(1, len(t) - 1) if UB[i] >= UB[i - 1] and UB[i] > UB[i + 1]
                            and UB[i] > 1e3 * UB[0]), None)
                Uref = out["U_gamma"][0] + out["U_D"][0]
                dU = np.abs(out["U_gamma"] + out["U_D"] - Uref)
                Fmax = float(max(np.abs(out["Fk"]).max(), np.abs(out["Dk"]).max()))
                vals = np.abs(out["ledger_defect"]).max(axis=1)
                runs.append({"Nx": Nx, "Nn": Nn, "Nm": Nm, "t": t, "UB": UB, "UBD": UBD,
                             "row": {"case": case, "model": mname, "Omega_D": OD, "k0": k0, "Nx": Nx, "Nn": Nn,
                                     "Nm": Nm, "gamma_root": g, "v_perp": vperp, "B_sat_estimate": Bsat,
                                     "t_peak": None if ipk is None else float(t[ipk]),
                                     "UB_peak": None if ipk is None else float(UB[ipk]),
                                     "UBD_at_peak": None if ipk is None else float(UBD[ipk]),
                                     "B_peak": None if ipk is None else float(2 * abs(Bf[ipk])),
                                     "max_field_perturbation_energy": float(dU.max()),
                                     "ledger_rel": [float(v) for v in vals / max(dU.max(), 1e-300)],
                                     "gauss_rel": [float(v) for v in out["gauss"].max(axis=0) / Fmax],
                                     "K_rel_change": float(np.abs(out["K"] / out["K"][0] - 1).max()),
                                     "t_end": float(t[-1]), **common(out)}})
                q = runs[-1]["row"]
                print(f"{case:8s} {mname:10s} Nx={Nx} Nn={Nn} Nm={Nm} g={g:.4f} t_pk={q['t_peak']} "
                      f"B_pk={q['B_peak']} B_sat={Bsat:.4e} ledger={max(q['ledger_rel']):.1e} "
                      f"gauss={max(q['gauss_rel']):.1e} {q['status']} {q['run_time']:.0f}s", flush=True)
            base = runs[0]
            T_res, agree = base["t"][-1], []
            for rf in runs[1:]:
                n = min(len(base["t"]), len(rf["t"]))
                d = np.abs(base["UB"][:n] - rf["UB"][:n]) / base["UB"].max()
                bad = np.where(d > 2e-2)[0]
                if len(bad):
                    T_res = min(T_res, base["t"][bad[0]])
                if base["row"]["B_peak"] and rf["row"]["B_peak"]:
                    agree.append(abs(rf["row"]["B_peak"] - base["row"]["B_peak"]) / base["row"]["B_peak"])
                else:
                    agree.append(None)
            b = base["row"]
            grp = {"case": case, "model": mname, "Omega_D": OD, "T_res": float(T_res), "t_peak": b["t_peak"],
                   "B_peak": b["B_peak"], "B_sat_estimate": Bsat,
                   "ratio_to_estimate": None if b["B_peak"] is None else b["B_peak"] / Bsat,
                   "peak_agreement_refinements": agree}
            grp["passed_resolution"] = bool(b["t_peak"] is not None and b["t_peak"] < T_res and len(agree) > 0
                                            and all(a is not None and a <= 2e-2 for a in agree))
            grp["passed_estimate"] = bool(grp["ratio_to_estimate"] is not None and 1 / 3 <= grp["ratio_to_estimate"] <= 3)
            grp["passed_ledger_gauss"] = bool(all(max(r["row"]["ledger_rel"]) <= 1e-6 and max(r["row"]["gauss_rel"]) <= 1e-8
                                                  and r["row"]["status"] == "success" for r in runs))
            groups.append(grp)
            print(grp, flush=True)
            for r in runs:
                r["row"]["series"] = {"t": r["t"].tolist(), "UB": r["UB"].tolist(), "UBD": r["UBD"].tolist()}
                rows.append(r["row"])
    for case in CASES:
        o = [g for g in groups if g["case"] == case and g["model"] == "ordinary"][0]
        for g in groups:
            if g["case"] == case and g["B_peak"] and o["B_peak"]:
                g["dark_over_ordinary_B_peak"] = g["B_peak"] / o["B_peak"]
                g["dark_over_ordinary_estimate"] = g["B_sat_estimate"] / o["B_sat_estimate"]
    return {"gates": {"resolution_rel": 2e-2, "estimate_ratio": [1 / 3, 3], "ledger_rel": 1e-6, "gauss_rel": 1e-8},
            "settings": {"k0": k0, "eta": eta, "seed_Bz": 1e-6, "t_max": tmax, "rtol": 1e-10, "atol": 1e-16,
                         "levels_Nx_Nn_Nm": [list(x) for x in levels], "Np": 1},
            "groups": groups, "results": rows}


if __name__ == "__main__":
    part = sys.argv[1]
    quick = "quick" in sys.argv[2:]
    tic = time.perf_counter()
    rec = {"linear": linear, "nonlinear": nonlinear, "ladder": lambda q: linear(q, ladder=True)}[part](quick)
    rec.update({"repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
                "command": " ".join(["python", "studies/b09.py"] + sys.argv[1:]),
                "wall_time": time.perf_counter() - tic})
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{part}{'_quick' if quick else ''}.json").write_text(json.dumps(rec, indent=1) + "\n")
