"""B08: uniform-B0 rotation and cold/warm electromagnetic waves in 1D3V, ordinary and dark.

Parent units: c = 1, force factor q_s Omega_cs[s], Ampere source -J/Omega_cs[0], J = sum_s q_s n_s U_s.
Electrons q = -1, Omega_cs = 1 (unit mass); positron-like species q = +1. Gyrofrequency vector q Omega_cs B0.

Part 1 (rotation, `python studies/b08.py rotation`): homogeneous drifting anisotropic Maxwellian, Nx = 1, uniform
B0 (|B0| = 0.5) along z, x, (1,1,1)/sqrt3 and an oblique (0.3,-0.5,0.8) direction, q = -1 and +1, density
n = 1 (omega_p = 1, the mean field E_0 is driven by the mean current) and n = 1e-4 (test-particle limit).
References (independent of the Hermite code): the 6x6 linear system d(U,E_0)/dt for the mean flow and mean field,
dU/dt = q Om (E_0 + U x B0), dE_0/dt = -q n U/Om0, integrated by a matrix exponential; the covariance
P(t) = R P0 R^T with R = expm(q Om [x B0] t) (a uniform E only shifts U). Gates (declared before running):
max |U - U_ref| / |U0| <= 1e-8, max |P - P_ref| / |P0| <= 1e-8, fitted perpendicular gyration frequencies equal to
the 6x6 eigenfrequencies to 1e-8 relative, ledger defect |K - K0 - W_em| <= 1e-10 K0 (magnetic work zero: only
the electric work W_em changes K), |K_perp| conservation in the test-particle limit to 1e-6.

Part 2 (waves, `python studies/b08.py waves`): electrons (n = 1, omega_pe = 1, |Omega_ce| = 0.5) on a fixed ion
background, k along x, B0 = 0.5 x (k parallel: R, L, whistler, Langmuir) or B0 = 0.5 z (k perpendicular: O, X).
Dark: eta = 0.3, Omega_D = 1.2 omega_pe (the transverse Proca branch sqrt(k^2 + Omega_D^2) crosses the R and upper-X
branches near k = 1.1). Reference: the full cold field-current eigenproblem dX/dt = L X with
X = (n1, v (3), E, B, E_D, B_D, A_D, phi_D) solved by numpy.linalg.eig (no determinant, no division at
degeneracies). Independent second path: the conductivity sigma(w) = sum_s q^2 n Om (-i w I - q Om C)^-1,
C v = v x B0, inserted in the field-only system M(w) F = 0; the smallest singular value of M(w)/|M| at every
eigenfrequency is recorded. Each positive-frequency eigenvector at k = 0.5 and 1.1 initializes a Hermite run
(density, flux and all fields set from the eigenvector, amplitude 1e-5 in n1 or |v|) with v_t = 0.01 (warm) and is fitted by
ds.fit_modes with one mode over the explicit window [0, T], T = min(6 periods, 200). Gates: |w_fit - w_cold|/|w|
<= 2e-3 at v_t = 0.01 (warm corrections ~ (k v_t / w)^2, (k v_t / (w - Om))^2), Nn refinement 4 -> 8 changes
w_fit by <= 1e-4 relative, kinetic ledger |K - K0 - W| <= 1e-6 U_wave and field ledgers <= 1e-5 U_wave
(U_wave = U_gamma0 - B0^2/2 + U_D0; the round-off floor of the B0 energy is about 3e-8 U_wave), Gauss residuals
<= 1e-10 of the initial |E_k|.
The warm shift is quantified with a second v_t = 0.005 run (expected ~1/4 of the v_t = 0.01 shift).

Part 3 (Proca flux, `python studies/b08.py flux`): local Proca energy theorem in a packet of transverse and
longitudinal modes with plasma and eta = 0.3: d u_D/dt + d S_D/dx = -P_D, u_D = Om0^2/2 (E_D^2 + B_D^2 +
Omega_D^2 (A_D^2 + phi_D^2)), S_D = Om0^2 [(E_D x B_D)_x + Omega_D^2 phi_D A_D,x], P_D = Om0 eta J.E_D,
pointwise on a refined grid with the time derivatives from the RHS. Gate: residual <= 1e-10 of max|d u_D/dt|;
without the phi_D A_D term the residual must be O(1) for the longitudinal packet.

Run: python studies/b08.py rotation|waves|flux [quick]  -> studies/b08/{rotation,waves,flux}.json
"""

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

import darkspectrax as ds
from darkspectrax._model import rhs

out_dir = Path(__file__).resolve().parent / "b08"
eta, Omega_D, B0mag = 0.3, 1.2, 0.5


def cross_matrix(b):
    """C with C v = v x b."""
    bx, by, bz = b
    return -np.array([[0, -bz, by], [bz, 0, -bx], [-by, bx, 0]], float)


def common(out):
    return {"status": out["status"], "max_ledger_defect": float(np.abs(out["ledger_defect"]).max()),
            "max_gauss": [float(v) for v in out["gauss"].max(axis=0)], "steps": int(out["num_steps"]),
            "compile_time": float(out["compile_time"]), "run_time": float(out["run_time"])}


# ----------------------------------------------------------------------------------------------- part 1
def rotation(quick):
    rng_dirs = {"z": (0, 0, 1), "x": (1, 0, 0), "diag": (1, 1, 1), "oblique": (0.3, -0.5, 0.8)}
    U0 = np.array([0.02, -0.01, 0.015])
    vt = np.array([0.01, 0.02, 0.015])
    rows = []
    for name, d in rng_dirs.items():
        b = B0mag * np.asarray(d, float) / np.linalg.norm(d)
        for q in (-1.0, 1.0):
            for n in ((1.0,) if quick else (1.0, 1e-4)):
                model = ds.Model(Nx=1, Nn=4, Nm=4, Np=4, qs=(q,), Omega_cs=(1.0,), alpha_s=tuple(np.sqrt(2) * vt),
                                 u_s=tuple(U0), rho_background=-q * n)
                y = ds.maxwellian(model, [n])
                y = {**y, "Fk": y["Fk"].at[3:, 0, 0, 0].set(jnp.asarray(b, complex))}
                Om = 2 * np.pi / B0mag
                t_max = 4 * Om
                out = ds.run(model, y, t_max, n_save=401, rtol=1e-12, atol=1e-16)
                t = out["t"]
                Us, Ps = [], []
                for i in range(len(t)):
                    nn, M, M2 = (np.asarray(x[0, ..., 0, 0, 0]).real for x in ds.moments(model, jnp.asarray(out["Ck"][i])))
                    U = M / nn
                    Us.append(U)
                    Ps.append(M2 / nn - np.outer(U, U))
                Us, Ps = np.array(Us), np.array(Ps)
                E0 = np.real(out["Fk"][:, :3, 0, 0, 0])
                C = cross_matrix(b)
                L = np.zeros((6, 6))
                L[:3, :3], L[:3, 3:], L[3:, :3] = q * C, q * np.eye(3), -q * n * np.eye(3)
                X0 = np.concatenate([U0, np.zeros(3)])
                Xr = np.array([expm(L * tt) @ X0 for tt in t])
                P0 = np.diag(vt ** 2)
                Pr = np.array([expm(q * C * tt) @ P0 @ expm(q * C * tt).T for tt in t])
                lam = np.linalg.eigvals(L)
                wref = np.sort(np.abs(lam.imag))[::2]  # three |omega| (each a +-pair)
                # gyration frequency fit: perpendicular U component (complex combination e1 + i e2)
                bh = b / np.linalg.norm(b)
                e1 = np.cross(bh, [1.0, 0, 0] if abs(bh[0]) < 0.9 else [0, 1.0, 0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(bh, e1)
                z = Us @ e1 + 1j * (Us @ e2)
                guesses = [w for w in np.sort(lam.imag * -1) if abs(w) > 1e-9]
                amp = np.linalg.lstsq(np.exp(-1j * np.outer(t, guesses)), z, rcond=None)[0]
                keep = [g for g, a in zip(guesses, amp) if abs(a) > 1e-6 * np.abs(z).max()]
                wfit, res = ds.fit_modes(t, z, [1.003 * g for g in keep])  # start 0.3% off the reference
                werr = max(abs(wf - g) / abs(g) for wf, g in zip(wfit, keep))
                K0 = out["K"][0]
                Kperp = np.einsum("ti,ti->t", Us, Us) - (Us @ bh) ** 2
                Kperp = Kperp + np.trace(Ps, axis1=1, axis2=2) - np.einsum("i,tij,j->t", bh, Ps, bh)
                rows.append({
                    "B0_direction": name, "B0": b.tolist(), "q": q, "n": n, "t_max": t_max,
                    "omega_ref": wref.tolist(), "omega_gyro_test_particle": B0mag,
                    "fit_modes_ref": [float(g) for g in keep], "fit_modes": np.real(wfit).tolist(),
                    "fit_residual": res, "omega_rel_error": float(werr),
                    "U_rel_error": float(np.abs(Us - Xr[:, :3]).max() / np.linalg.norm(U0)),
                    "E0_rel_error": float(np.abs(E0 - Xr[:, 3:]).max() / max(np.abs(Xr[:, 3:]).max(), 1e-300)),
                    "P_rel_error": float(np.abs(Ps - Pr).max() / np.abs(P0).max()),
                    "P_offdiag_max": float(np.abs(Ps - np.einsum("tii->ti", Ps)[:, :, None] * np.eye(3)).max()),
                    "ledger_rel": float(np.abs(out["ledger_defect"]).max() / K0),
                    "K_rel_change": float(np.abs(out["K"] / K0 - 1).max()),
                    "Kperp_rel_change": float(np.abs(Kperp / Kperp[0] - 1).max()),
                    "max_abs_E0": float(np.abs(E0).max()), **common(out)})
                r = rows[-1]
                print(f"{name:8s} q={q:+.0f} n={n:g} U {r['U_rel_error']:.1e} P {r['P_rel_error']:.1e} "
                      f"w {r['omega_rel_error']:.1e} ledger {r['ledger_rel']:.1e} K {r['K_rel_change']:.1e} "
                      f"Kperp {r['Kperp_rel_change']:.1e}", flush=True)
    gates = {"U": 1e-8, "P": 1e-8, "omega": 1e-8, "ledger": 1e-10, "Kperp_test_particle": 1e-6}
    passed = all(r["U_rel_error"] <= gates["U"] and r["P_rel_error"] <= gates["P"]
                 and r["omega_rel_error"] <= gates["omega"] and r["ledger_rel"] <= gates["ledger"]
                 and (r["n"] > 1e-3 or r["Kperp_rel_change"] <= gates["Kperp_test_particle"]) for r in rows)
    return {"gates": gates, "passed": passed, "settings": {"U0": U0.tolist(), "v_t": vt.tolist(), "B0_mag": B0mag,
            "hermite": [4, 4, 4], "rtol": 1e-12, "atol": 1e-16}, "results": rows}


# ----------------------------------------------------------------------------------------------- part 2
def linear_matrix(k, b, dark, n=1.0, q=-1.0, Om=1.0, om0=1.0, eta_=eta):
    """Cold linear operator L (dX/dt = L X), X = (n1, v3, E3, B3[, ED3, BD3, AD3, phi])."""
    kv = np.array([k, 0.0, 0.0])
    K = np.array([[0, -kv[2], kv[1]], [kv[2], 0, -kv[0]], [-kv[1], kv[0], 0]], complex)  # K a = k x a
    N = 10 + (10 if dark else 0)
    L = np.zeros((N, N), complex)
    v, E, B = slice(1, 4), slice(4, 7), slice(7, 10)
    L[0, v] = -1j * n * kv
    L[v, v] = q * Om * cross_matrix(b)
    L[v, E] = q * Om * np.eye(3)
    L[E, v] = -q * n * np.eye(3) / om0
    L[E, B] = 1j * K
    L[B, E] = -1j * K
    if dark:
        ED, BD, AD, ph = slice(10, 13), slice(13, 16), slice(16, 19), 19
        L[v, ED] = eta_ * q * Om * np.eye(3)
        L[ED, v] = -eta_ * q * n * np.eye(3) / om0
        L[ED, BD] = 1j * K
        L[ED, AD] = Omega_D ** 2 * np.eye(3)
        L[BD, ED] = -1j * K
        L[AD, ED] = -np.eye(3)
        L[AD, ph] = -1j * kv
        L[ph, AD] = -1j * kv
    return L


def sigma_residual(w, k, b, dark, n=1.0, q=-1.0, Om=1.0, om0=1.0):
    """Smallest singular value of the field-only system M(w) (cold conductivity), relative to |M|."""
    sig = q * q * n * Om * np.linalg.inv(-1j * w * np.eye(3) - q * Om * cross_matrix(b))
    K = np.array([[0, 0, 0], [0, 0, -k], [0, k, 0]], complex)
    nf = 16 if dark else 6
    M = np.zeros((nf, nf), complex)
    E, B = slice(0, 3), slice(3, 6)
    M[E, E] = -1j * w * np.eye(3) + sig / om0
    M[E, B] = -1j * K
    M[B, B] = -1j * w * np.eye(3)
    M[B, E] = 1j * K
    if dark:
        ED, BD, AD, ph = slice(6, 9), slice(9, 12), slice(12, 15), 15
        kv = np.array([k, 0.0, 0.0])
        M[E, ED] = eta * sig / om0
        M[ED, ED] = -1j * w * np.eye(3) + eta ** 2 * sig / om0
        M[ED, E] = eta * sig / om0
        M[ED, BD] = -1j * K
        M[ED, AD] = -Omega_D ** 2 * np.eye(3)
        M[BD, BD] = -1j * w * np.eye(3)
        M[BD, ED] = 1j * K
        M[AD, AD] = -1j * w * np.eye(3)
        M[AD, ED] = np.eye(3)
        M[AD, ph] = 1j * kv
        M[ph, ph] = -1j * w
        M[ph, AD] = 1j * kv
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[-1] / s[0])


def label(vec, geom, dark):
    E = vec[4:7]
    ED = vec[10:13] if dark else np.zeros(3)
    tot = np.abs(E) ** 2
    frac_dark = float(np.sum(np.abs(ED) ** 2) / (np.sum(np.abs(E) ** 2) + np.sum(np.abs(ED) ** 2)))
    F = E if np.sum(np.abs(E) ** 2) > 1e-6 * np.sum(np.abs(ED) ** 2) else ED
    long = abs(F[0]) ** 2 / np.sum(np.abs(F) ** 2)
    if geom == "parallel":
        if long > 0.5:
            kind = "longitudinal"
        else:
            r = abs(F[1] + 1j * F[2]) ** 2 - abs(F[1] - 1j * F[2]) ** 2  # handedness about +x for e^{-iwt}
            kind = "R" if r < 0 else "L"
    else:
        kind = "O" if abs(F[2]) ** 2 > 0.5 * np.sum(np.abs(F) ** 2) else "X"
    del tot
    return kind, frac_dark, float(long)


def wave_model(geom, dark, k, vt, Nn):
    return ds.Model(Nx=5, Nn=Nn, Nm=Nn if geom == "perpendicular" else 4, Np=4, Lx=2 * np.pi / k,
                    qs=(-1.0,), Omega_cs=(1.0,), alpha_s=(np.sqrt(2) * vt,) * 3, u_s=(0.0, 0.0, 0.0),
                    rho_background=1.0, mode="self_consistent" if dark else "ordinary",
                    eta=eta if dark else 0.0, Omega_D=Omega_D)


def init_mode(model, vec, b, dark, amp):
    y = ds.maxwellian(model, [1.0])
    vec = amp * vec / np.abs(vec[:4]).max()
    a = model.alpha_s[0]
    A = a ** 3
    H = model.Nn * model.Nm * model.Np
    Ck = y["Ck"]
    from darkspectrax import hermite_index
    Ck = Ck.at[0, 0, 1, 0].set(vec[0] / A)
    for i, e in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        Ck = Ck.at[hermite_index(*e, model.Nn, model.Nm), 0, 1, 0].set(np.sqrt(2) * vec[1 + i] / (A * a))
    del H
    Fk = y["Fk"].at[3:, 0, 0, 0].set(jnp.asarray(b, complex)).at[:, 0, 1, 0].set(jnp.asarray(vec[4:10]))
    Dk = y["Dk"]
    if dark:
        Dk = Dk.at[:, 0, 1, 0].set(jnp.asarray(vec[10:20]))
    return {**y, "Ck": Ck, "Fk": Fk, "Dk": Dk}


def waves(quick):
    rows, eig = [], []
    ks = (1.1,) if quick else (0.5, 1.1)
    levels = ((0.01, 4),) if quick else ((0.01, 4), (0.01, 8), (0.005, 4))
    for geom, b in (("parallel", (B0mag, 0, 0)), ("perpendicular", (0, 0, B0mag))):
        for dark in (False, True):
            for k in ks:
                L = linear_matrix(k, np.asarray(b, float), dark)
                lam, V = np.linalg.eig(L)
                w = 1j * lam  # e^{lam t} = e^{-i w t}
                sel = [i for i in np.argsort(w.real) if w[i].real > 1e-8]
                modes = []
                for i in sel:
                    kind, fd, lg = label(V[:, i], geom, dark)
                    modes.append({"omega": float(w[i].real), "omega_imag": float(w[i].imag), "kind": kind,
                                  "dark_fraction": fd, "longitudinal_fraction": lg,
                                  "sigma_residual": sigma_residual(w[i].real, k, np.asarray(b, float), dark),
                                  "vec": V[:, i]})
                eig.append({"geometry": geom, "model": "dark" if dark else "ordinary", "k": k,
                            "modes": [{kk: vv for kk, vv in m.items() if kk != "vec"} for m in modes],
                            "zero_modes": int(np.sum(np.abs(w) < 1e-8)), "max_abs_imag": float(np.abs(w.imag).max())})
                for vt, Nn in levels:
                    model = wave_model(geom, dark, k, vt, Nn)
                    cache = {}
                    for m in modes:
                        T = float(min(6 * 2 * np.pi / m["omega"], 200.0))
                        y = init_mode(model, m["vec"], b, dark, 1e-5)
                        out = ds.run(model, y, T, n_save=601, rtol=1e-11, atol=1e-16)
                        t = out["t"]
                        comps = np.concatenate([out["Fk"][:, :, 0, 1, 0], out["Dk"][:, :, 0, 1, 0]], axis=1)
                        j = int(np.argmax(np.abs(comps[0])))
                        wf, res = ds.fit_modes(t, comps[:, j], [m["omega"]])
                        wf = complex(wf[0])
                        Uw = out["U_gamma"][0] - 0.5 * B0mag ** 2 + out["U_D"][0]
                        g0 = float(np.abs(y["Fk"][:3, 0, 1, 0]).max())
                        rows.append({"geometry": geom, "model": "dark" if dark else "ordinary", "k": k, "v_t": vt,
                                     "Nn": Nn, "kind": m["kind"], "dark_fraction": m["dark_fraction"],
                                     "omega_cold": m["omega"], "omega_fit": [wf.real, wf.imag],
                                     "rel_error": abs(wf - m["omega"]) / m["omega"],
                                     "shift": (wf.real - m["omega"]) / m["omega"], "fit_residual": res,
                                     "window": [0.0, T], "component": j, "U_wave": float(Uw),
                                     "ledger_rel": [float(v) for v in np.abs(out["ledger_defect"]).max(axis=1) / Uw],
                                     "gauss_rel": [float(v) for v in out["gauss"].max(axis=0) / g0], **common(out)})
                        r = rows[-1]
                        print(f"{geom[:4]} {r['model'][:4]} k={k} vt={vt} Nn={Nn} {m['kind']:12s} "
                              f"fd={m['dark_fraction']:.2f} "
                              f"w={m['omega']:.6f} fit={wf.real:.6f}{wf.imag:+.1e}i err={r['rel_error']:.1e} "
                              f"res={res:.1e} ledger={max(r['ledger_rel']):.1e}/{r['ledger_rel'][0]:.1e} gauss={max(r['gauss_rel']):.1e} {r['status']} {r['run_time']:.0f}s",
                              flush=True)
                    del cache
    gates = {"rel_error_vt0.01": 2e-3, "Nn_refinement": 1e-4, "kinetic_ledger_rel": 1e-6, "field_ledger_rel": 1e-5,
             "gauss_rel": 1e-10}
    return {"gates": gates, "settings": {"eta": eta, "Omega_D": Omega_D, "B0": B0mag, "omega_pe": 1.0,
            "amplitude": 1e-5, "rtol": 1e-11, "atol": 1e-16, "Nx": 5, "levels": [list(x) for x in levels]},
            "eigen": eig, "results": rows}


# ----------------------------------------------------------------------------------------------- part 3
def flux(quick):
    vt = 0.01
    model = ds.Model(Nx=32, Nn=4, Nm=4, Np=4, Lx=2 * np.pi / 0.25, qs=(-1.0,), Omega_cs=(1.0,),
                     alpha_s=(np.sqrt(2) * vt,) * 3, rho_background=1.0, mode="self_consistent", eta=eta,
                     Omega_D=Omega_D)
    rng = np.random.default_rng(0)
    y = ds.maxwellian(model, [1.0], [(0, (i, 0, 0), 1e-3 * (rng.normal() + 1j * rng.normal())) for i in (1, 2, 3)])
    y = ds.consistent_fields(model, y)
    y = {**y, "Fk": y["Fk"].at[3:, 0, 0, 0].set(jnp.asarray([0.2, 0.0, 0.4], complex))}
    rows = []
    for name, pol in (("transverse", (0.0, 1.0, 0.5j)), ("longitudinal", (1.0, 0.0, 0.0)), ("mixed", (1.0, 0.7, -0.3j))):
        yy = y
        for i in range(1, 6):  # Gaussian packet in k centred at index 3
            yy = ds.proca_mode(model, yy, (i, 0, 0), 1e-3 * np.exp(-(i - 3) ** 2 / 2) * np.asarray(pol))
        out = ds.run(model, yy, 5.0, n_save=3, rtol=1e-11, atol=1e-16)
        st = {kk: jnp.asarray(out[kk][-1]) for kk in ("Ck", "Fk", "Dk")}
        st["W"] = jnp.asarray(out["W"][-1], complex)
        d = rhs(5.0, st, model)
        Nf = 256
        x = np.arange(Nf) * model.Lx / Nf

        def real_nomask(c):
            c = np.asarray(c)[:, 0, :, 0]
            kk = np.arange(c.shape[1]) * 2 * np.pi / model.Lx
            w = np.where(np.arange(c.shape[1]) == 0, 1.0, 2.0)
            return np.real((w * c) @ np.exp(1j * np.outer(kk, x)))

        D, dD = real_nomask(st["Dk"]), real_nomask(d["Dk"])
        ED, BD, AD, ph = D[0:3], D[3:6], D[6:9], D[9]
        from darkspectrax._diagnostics import moments
        _, M, _ = moments(model, st["Ck"])
        J = -real_nomask(M[0])
        om0 = 1.0
        dudt = om0 ** 2 * (np.sum(ED * dD[0:3] + BD * dD[3:6], 0) + Omega_D ** 2 * (np.sum(AD * dD[6:9], 0) + ph * dD[9]))
        S1 = (ED[1] * BD[2] - ED[2] * BD[1]) * om0 ** 2
        S2 = Omega_D ** 2 * ph * AD[0] * om0 ** 2
        # d/dx of products by exact spectral evaluation on the fine grid (band limit 2*5 < 256/2)
        def dx_grid(f):
            fk = np.fft.rfft(f)
            kk = np.arange(fk.size) * 2 * np.pi / model.Lx
            return np.fft.irfft(1j * kk * fk, n=Nf)
        PD = om0 * eta * np.sum(J * ED, 0)
        res_full = dudt + dx_grid(S1 + S2) + PD
        res_noA = dudt + dx_grid(S1) + PD
        scale = np.abs(dudt).max()
        rows.append({"packet": name, "max_dudt": float(scale), "max_PD": float(np.abs(PD).max()),
                     "max_dSdx_phiA": float(np.abs(dx_grid(S2)).max()),
                     "residual_rel": float(np.abs(res_full).max() / scale),
                     "residual_rel_without_phiA": float(np.abs(res_noA).max() / scale), **common(out)})
        print(rows[-1], flush=True)
    return {"gates": {"residual": 1e-10}, "settings": {"Nx": 32, "Lx": float(model.Lx), "eta": eta,
            "Omega_D": Omega_D, "fine_grid": 256}, "results": rows}


if __name__ == "__main__":
    part = sys.argv[1]
    quick = "quick" in sys.argv[2:]
    tic = time.perf_counter()
    rec = {"rotation": rotation, "waves": waves, "flux": flux}[part](quick)
    rec.update({"repository_commit": ds._simulation._git_sha(), "parent_commit": ds.PARENT_COMMIT,
                "command": " ".join(["python", "studies/b08.py"] + sys.argv[1:]),
                "wall_time": time.perf_counter() - tic})
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{part}{'_quick' if quick else ''}.json").write_text(json.dumps(rec, indent=1) + "\n")
