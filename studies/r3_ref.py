"""R3 independent linear reference (NumPy/SciPy only; no kinetic code, no JAX): cold electron fluid + Maxwell + canonical
Proca in 1D (x) with a static density profile n(x) and uniform B0, advanced in time as a packet problem with a pseudo-spectral
x-derivative and scipy DOP853. Gates and set-up: r3_gates.py (declared before any run).

State (19 real fields on the grid): v(3), E(3), B(3), E_D(3), B_D(3), A_D(3), phi_D with (q = -1, Omega_c0 = 1)
  dv/dt = -(E + eta E_D + v x B0),  dE/dt = curl B + n v,  dB/dt = -curl E,
  dE_D/dt = curl B_D + Omega_D^2 A_D + eta n v,  dB_D/dt = -curl E_D,  dA_D/dt = -E_D - grad phi_D,  dphi_D/dt = -div A_D
(the same equations as Dark-SPECTRAX's cold limit; J = -n v). Energy density u = 1/2 [E^2 + B^2 + E_D^2 + B_D^2
+ Omega_D^2 (A_D^2 + phi_D^2) + n v^2]; fluxes S = E x B, S_D = E_D x B_D + Omega_D^2 phi_D A_D (x components).

Usage: python studies/r3_ref.py [case ...]  (see main) -> studies/r3/ref_*.json
"""
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_gates as g  # noqa: E402

OUT = Path(__file__).resolve().parent / "r3"
IV, IE, IB, IED, IBD, IAD, IPH = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15), slice(15, 18), 18
NF = 19


@dataclass(frozen=True)
class Case:
    eta: float
    w: float
    theta: object = None  # None: B0 = 0; else degrees
    pol: str = "y"
    amp: float = g.AMP
    ntrunc: int = 0  # >0: density band-limited to the modes kept by the 2/3 mask of an Nx = ntrunc kinetic grid

    @property
    def name(self):
        b = "B0" if self.theta is None else f"th{self.theta}"
        return f"eta{self.eta:g}_w{self.w:g}_{b}_{self.pol}" + (f"_trunc{self.ntrunc}" if self.ntrunc else "")

    @property
    def b0(self):
        return np.zeros(3) if self.theta is None else g.b0_vector(self.theta)


def profile(case, N):
    """Density on an N-point grid; band-limited like the kinetic code when case.ntrunc > 0."""
    x = np.arange(N) * (g.LB / N)
    n = g.density(x, case.w)
    if case.ntrunc:
        C = np.fft.rfft(n)
        C[(3 * np.arange(C.size)) >= case.ntrunc] = 0.0
        n = np.fft.irfft(C, n=N)
    return n


def curl_x(d, F):
    """curl of a (3, N) field depending on x only: (0, -dF_z, dF_y)."""
    return np.stack([np.zeros_like(F[0]), -d(F[2]), d(F[1])])


def make_rhs(case, N, n):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=g.LB / N)
    ik = 1j * k
    b0 = case.b0
    eta, OD2 = case.eta, g.OMEGA_D ** 2

    def rhs(t, y):
        X = y.reshape(NF, N)
        v, E, B, ED, BD, AD, ph = X[IV], X[IE], X[IB], X[IED], X[IBD], X[IAD], X[IPH]
        # one batched derivative: [B_y, B_z, E_y, E_z, BD_y, BD_z, ED_y, ED_z, phi, AD_x]
        S = np.stack([B[1], B[2], E[1], E[2], BD[1], BD[2], ED[1], ED[2], ph, AD[0]])
        D = np.fft.irfft(ik * np.fft.rfft(S, axis=-1), n=N, axis=-1)
        out = np.empty_like(X)
        out[IV] = -(E + eta * ED + np.cross(v, b0, axis=0))
        out[IE][0], out[IE][1], out[IE][2] = n * v[0], -D[1] + n * v[1], D[0] + n * v[2]
        out[IB][0], out[IB][1], out[IB][2] = 0.0, D[3], -D[2]
        out[IED][0], out[IED][1], out[IED][2] = OD2 * AD[0] + eta * n * v[0], -D[5] + OD2 * AD[1] + eta * n * v[1], D[4] + OD2 * AD[2] + eta * n * v[2]
        out[IBD][0], out[IBD][1], out[IBD][2] = 0.0, D[7], -D[6]
        out[IAD] = -ED
        out[IAD][0] -= D[8]
        out[IPH] = -D[9]
        return out.ravel()
    return rhs


def spectrum(N, nrm=True):
    """Box wavenumbers, packet amplitudes a_k (Gaussian, centred at XC0) for k > 0."""
    k = 2 * np.pi * np.fft.rfftfreq(N, d=g.LB / N)
    a = np.exp(-(k - g.K0) ** 2 / (2 * g.SIGMA_K ** 2)) * np.exp(-1j * k * g.XC0)
    a[k <= 0] = 0.0
    a[np.abs(k - g.K0) > 8 * g.SIGMA_K] = 0.0
    return k, a


def packet_coeffs(case):
    """Fourier coefficients F_k of the incident packet on the box wavenumbers (rfft index j = k Lb/2 pi, field = 2 Re sum F_k
    e^{ikx}), rows (E_D(3), B_D(3), A_D(3), phi_D) with the polarization vector, scaled so that max_x |E_D| = case.amp
    (scale fixed on the N = 4096 reference grid, identical for every grid)."""
    k, a = spectrum(4096)
    p = np.asarray(g.POLS[case.pol], float)
    om = np.sqrt(k ** 2 + g.OMEGA_D ** 2)
    A = a[None, :] * p[:, None]
    ph = k * a * p[0] / om
    ED = 1j * om * A
    ED[0] -= 1j * k * ph
    BD = 1j * k * np.stack([np.zeros_like(a), -A[2], A[1]])
    F = np.concatenate([ED, BD, A, ph[None]])
    Xp = np.fft.irfft(F[:, :] * 4096, n=4096, axis=-1)
    F = F * (case.amp / np.sqrt((Xp[:3] ** 2).sum(0)).max())
    return F


def initial_state(case, N):
    F = packet_coeffs(case)
    Nk = N // 2 + 1
    G = np.zeros((NF, Nk), complex)
    m = min(Nk, F.shape[1])
    G[IED, :m], G[IBD, :m], G[IAD, :m], G[IPH, :m] = F[0:3, :m], F[3:6, :m], F[6:9, :m], F[9, :m]
    return np.fft.irfft(G * N, n=N, axis=-1)


def energy_density(X, n):
    return 0.5 * ((X[IE] ** 2).sum(0) + (X[IB] ** 2).sum(0) + (X[IED] ** 2).sum(0) + (X[IBD] ** 2).sum(0)
                  + g.OMEGA_D ** 2 * ((X[IAD] ** 2).sum(0) + X[IPH] ** 2) + n * (X[IV] ** 2).sum(0))


def fluxes(X):
    S = X[IE][1] * X[IB][2] - X[IE][2] * X[IB][1]
    SD = X[IED][1] * X[IBD][2] - X[IED][2] * X[IBD][1]
    SDp = g.OMEGA_D ** 2 * X[IPH] * X[IAD][0]
    return S, SD, SDp


def pad(X, M):
    N = X.shape[-1]
    C = np.fft.rfft(X, axis=-1)
    Cp = np.zeros((X.shape[0], M // 2 + 1), complex)
    Cp[:, :C.shape[1] - 1] = C[:, :-1]  # drop Nyquist
    return np.fft.irfft(Cp, n=M, axis=-1) * (M / N)


# ------------------------------------------------------------------------------------------------ branch projection
def _system_matrix_k(k, nw, b0, eta):
    L = np.zeros((NF, NF), complex)
    d = lambda f: 1j * k * f  # noqa: E731
    for j in range(NF):
        X = np.zeros((NF, 1), complex)
        X[j] = 1.0
        v, E, B, ED, BD, AD, ph = X[IV], X[IE], X[IB], X[IED], X[IBD], X[IAD], X[IPH]
        out = np.zeros_like(X)
        out[IV] = -(E + eta * ED + np.cross(v, b0[:, None], axis=0))
        out[IE] = curl_x(d, B) + nw * v
        out[IB] = -curl_x(d, E)
        out[IED] = curl_x(d, BD) + g.OMEGA_D ** 2 * AD + eta * nw * v
        out[IBD] = -curl_x(d, ED)
        out[IAD] = -ED
        out[IAD][0] -= d(ph)
        out[IPH] = -d(AD[0])
        L[:, j] = out[:, 0]
    return L


def system_matrix_parts(nw, b0, eta):
    """L(k) = L0 + k L1 (exactly linear in k)."""
    L0 = _system_matrix_k(0.0, nw, b0, eta)
    return L0, _system_matrix_k(1.0, nw, b0, eta) - L0


def system_matrix(k, nw, b0, eta):
    L0, L1 = system_matrix_parts(nw, b0, eta)
    return L0 + k * L1


def weights(nw):
    W = np.ones(NF)
    W[IV] = nw
    W[IAD] = g.OMEGA_D ** 2
    W[IPH] = g.OMEGA_D ** 2
    return W


def branch_energy(Xw, nw, b0, eta):
    """Energy (int u dx over the periodic box) of the fields Xw (zero outside the window) split by direction and type.

    Returns dict with keys 'right_dark','right_ord','left_dark','left_ord','static' and 'total'."""
    N = Xw.shape[-1]
    Ck = np.fft.fft(Xw, axis=-1) / N  # int |f|^2 dx = LB sum |Ck|^2
    kk = 2 * np.pi * np.fft.fftfreq(N, d=g.LB / N)
    W = weights(nw)
    dark_rows = np.zeros(NF, bool)
    dark_rows[IED] = dark_rows[IBD] = dark_rows[IAD] = dark_rows[IPH] = True
    out = {key: 0.0 for key in ("right_dark", "right_ord", "left_dark", "left_ord", "static", "total")}
    scale = np.max(np.sum(np.abs(Ck) ** 2, axis=0)) * 1e-14
    L0, L1 = system_matrix_parts(nw, b0, eta)
    for i, k in enumerate(kk):
        x = Ck[:, i]
        e_tot = 0.5 * g.LB * np.sum(W * np.abs(x) ** 2)
        out["total"] += e_tot
        if np.sum(np.abs(x) ** 2) < scale:
            continue
        lam, V = np.linalg.eig(L0 + k * L1)
        om = 1j * lam
        groups = {}
        for j in range(NF):
            if abs(om[j]) < 1e-9:
                key = "static"
            else:
                share = np.sum(W[dark_rows] * np.abs(V[dark_rows, j]) ** 2) / np.sum(W * np.abs(V[:, j]) ** 2)
                direction = "right" if k * om[j].real > 0 else "left"
                key = f"{direction}_{'dark' if share > 0.5 else 'ord'}"
            groups.setdefault(key, []).append(j)
        for key, idx in groups.items():
            Vg = V[:, idx]
            Gm = Vg.conj().T @ (W[:, None] * Vg)
            c = np.linalg.solve(Gm, Vg.conj().T @ (W * x))
            xg = Vg @ c
            out[key] += 0.5 * g.LB * np.real(np.sum(W * np.abs(xg) ** 2))
    return out


def closure_of(ch):
    keys = ("refl_dark", "refl_photon", "inc_left", "trans_dark", "trans_photon", "left_back_in_mid", "static_left",
            "static_mid", "stored", "beyond", "absorbed")
    return abs(sum(ch[k] for k in keys) - 1.0)


def channels(X, case, N, n, E_in):
    dx = g.LB / N
    x = np.arange(N) * dx
    w = case.w
    u = energy_density(X, n)
    left = x < g.XU - 6 * w
    mid = (x > g.XU + 6 * w) & (x < g.XD - 6 * w)
    zone = np.abs(x - g.XU) <= 6 * w
    beyond = x >= g.XD - 6 * w
    wrap = x > g.XD + 6 * w  # periodic image of the left side: reflected waves that crossed x = 0 (box wraparound)
    res = {}
    for name, mask in (("left", left), ("mid", mid), ("wrap", wrap)):
        Xw = X * mask[None, :]
        nw = float(np.mean(n[mask]))
        be = branch_energy(Xw, nw, case.b0, case.eta)
        res[name] = {k: v / E_in for k, v in be.items()}
        res[name]["direct"] = float(np.sum(u[mask]) * dx / E_in)
    res["stored"] = float(np.sum(u[zone]) * dx / E_in)
    res["beyond"] = float(np.sum(u[beyond]) * dx / E_in)
    ch = {"refl_dark": res["left"]["left_dark"], "refl_photon": res["left"]["left_ord"],
          "inc_left": res["left"]["right_dark"] + res["left"]["right_ord"],
          "trans_dark": res["mid"]["right_dark"], "trans_photon": res["mid"]["right_ord"],
          "left_back_in_mid": res["mid"]["left_dark"] + res["mid"]["left_ord"],
          "stored": res["stored"], "beyond": res["beyond"], "absorbed": 0.0,
          # wraparound decomposition of 'beyond' (declared window x > xd - 6w; reflected waves reach x = 0 before T)
          "wrap_refl_dark": res["wrap"]["left_dark"], "wrap_refl_photon": res["wrap"]["left_ord"],
          "wrap_right": res["wrap"]["right_dark"] + res["wrap"]["right_ord"],
          "static_left": res["left"]["static"], "static_mid": res["mid"]["static"]}
    ch["closure"] = closure_of(ch)
    return ch, res


# ------------------------------------------------------------------------------------------------ the run
def run_cold(case, N=4096, rtol=1e-12, dt_out=0.05, fine=4):
    """Returns channels, flux budget, energy drift. Planes at the grid nodes nearest to XU -/+ 7w."""
    dx = g.LB / N
    x = np.arange(N) * dx
    n = profile(case, N)
    X0 = initial_state(case, N)
    E_in = float(np.sum(energy_density(X0, n)) * dx)
    ia, ib = int(round((g.XU - 7 * case.w) / dx)), int(round((g.XU + 7 * case.w) / dx))
    rhs = make_rhs(case, N, n)
    t_eval = np.arange(0.0, g.T_END + 1e-9, dt_out)
    tic = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, g.T_END), X0.ravel(), method="DOP853", rtol=rtol, atol=1e-11 * case.amp,
                    t_eval=t_eval, max_step=0.2)
    wall = time.perf_counter() - tic
    Y = sol.y.reshape(NF, N, -1)
    nt = Y.shape[-1]
    Etot = np.array([np.sum(energy_density(Y[:, :, i], n)) * dx for i in range(nt)])
    S = np.array([[fluxes(Y[:, ia, i:i + 1])[j][0] for j in range(3)] + [fluxes(Y[:, ib, i:i + 1])[j][0] for j in range(3)]
                  for i in range(nt)])
    Fa_full, Fb_full = S[:, 0] + S[:, 1] + S[:, 2], S[:, 3] + S[:, 4] + S[:, 5]
    Fa_nophi, Fb_nophi = S[:, 0] + S[:, 1], S[:, 3] + S[:, 4]
    M = fine * N

    def zone_energy(Xc):
        Xf = pad(Xc, M)
        xf = np.arange(M) * (g.LB / M)
        uf = energy_density(Xf, profile(case, M))
        i0, i1 = ia * fine, ib * fine
        return float((np.sum(uf[i0:i1 + 1]) - 0.5 * (uf[i0] + uf[i1])) * (g.LB / M))
    Ez0, EzT = zone_energy(Y[:, :, 0]), zone_energy(Y[:, :, -1])
    tt = sol.t
    flux_in = np.trapezoid(Fa_full - Fb_full, tt)
    flux_in_nophi = np.trapezoid(Fa_nophi - Fb_nophi, tt)
    # same budget with the flux sampled every 0.5 (what the kinetic runs save)
    sub = slice(None, None, int(round(0.5 / dt_out)))
    flux_in_sub = np.trapezoid((Fa_full - Fb_full)[sub], tt[sub])
    ch, res = channels(Y[:, :, -1], case, N, n, E_in)
    return {
        "case": asdict(case) | {"name": case.name}, "N": N, "rtol": rtol, "E_in": E_in, "wall_s": wall,
        "nsteps_eval": int(nt), "energy_drift": float(np.abs(Etot / E_in - 1).max()),
        "channels": ch, "branch_windows": res,
        "budget": {"zone_dE": (EzT - Ez0) / E_in, "flux_in": flux_in / E_in, "residual": (EzT - Ez0 - flux_in) / E_in,
                   "residual_without_phiA": (EzT - Ez0 - flux_in_nophi) / E_in,
                   "residual_sampled_0p5": (EzT - Ez0 - flux_in_sub) / E_in,
                   "flux_phiA_share": float(np.trapezoid((S[:, 2] - S[:, 5]), tt) / E_in), "planes": [ia * dx, ib * dx]},
    }


# ------------------------------------------------------------------------------------------------ analytic laws
def ramp_cross(n_c, w):
    """x_c and n'(x_c) where the up-ramp density equals n_c (down ramp is >200 away)."""
    D = g.N_HI - g.N_LO
    arg = 2 * (n_c - g.N_LO) / D - 1
    xc = g.XU + w * np.arctanh(arg)
    return xc, D / (2 * w) * (1 - arg ** 2)


def spectral_weights():
    k, a = spectrum(4096)
    om = np.sqrt(k ** 2 + g.OMEGA_D ** 2)
    wgt = np.abs(a) ** 2 * om ** 2
    sel = wgt > 0
    return k[sel], om[sel], wgt[sel] / wgt[sel].sum()


def lz_params(case, om, f):
    eta = case.eta
    n_c = g.OMEGA_D ** 2 / ((1 - eta ** 2) * f)
    xc, npr = ramp_cross(n_c, case.w)
    Lc = n_c / npr
    kbar2 = om ** 2 - f * n_c
    gc = eta * n_c * f
    slope = (1 - eta ** 2) * f * npr
    Pd = np.exp(-np.pi * gc ** 2 / (np.sqrt(np.maximum(kbar2, 1e-30)) * slope))
    return Pd, Lc, kbar2, n_c


def refined_pd(case, om, f):
    """Dykhne complex-turning-point exponent (follow-up G, r3_followup_gates.py): P_d = exp(-Theta)."""
    eta, OD2 = case.eta, g.OMEGA_D ** 2
    D = g.N_HI - g.N_LO
    out = np.zeros(np.shape(om))
    for idx, (o, ff) in enumerate(zip(np.ravel(om), np.ravel(f))):
        ns = OD2 / (ff * (1 + 1j * eta) ** 2)  # one of the conjugate pair; take the upper-half-plane member below
        if ns.imag < 0:
            ns = np.conj(ns)
        nr = ns.real
        o2 = o ** 2

        def integrand(t):
            n = nr + t * (ns - nr)
            disc_sqrt = ff * (1 + eta ** 2) * np.sqrt(n - ns) * np.sqrt(n - np.conj(ns))
            mean = 0.5 * (ff * n + OD2 + eta ** 2 * ff * n)
            lam_m, lam_p = mean - 0.5 * disc_sqrt, mean + 0.5 * disc_sqrt
            kp, km = np.sqrt(o2 - lam_m + 0j), np.sqrt(o2 - lam_p + 0j)  # k_+ (smaller lambda) is the larger wavenumber
            tau = 2 * (n - g.N_LO) / D - 1
            npr = D / (2 * case.w) * (1 - tau ** 2)
            return (kp - km) / npr * (ns - nr)
        ts = np.linspace(0.0, 1.0, 4001)
        vals = np.array([integrand(t) for t in ts])
        # integrable (n - n*)^(1/2) end point: Simpson on a grid refined by the substitution is not needed at this accuracy
        integral = np.trapezoid(vals, ts)
        out.flat[idx] = np.exp(-2 * abs(integral.imag))
    return out


def refined_prediction(case):
    k, om, wgt = spectral_weights()
    if case.theta is None:
        fs = [lambda o: np.ones_like(o)]
    elif case.theta == 0 and case.pol == "y":
        fs = [lambda o: o / (o - g.OMEGA_C), lambda o: o / (o + g.OMEGA_C)]
    else:
        return None
    sel = np.linspace(0, om.size - 1, 21).astype(int)
    tr = rp = rd = 0.0
    for fn in fs:
        f = fn(om)
        P = np.interp(np.arange(om.size), sel, refined_pd(case, om[sel], f[sel]))
        tr += np.sum(wgt * P) / len(fs)
        rp += np.sum(wgt * P * (1 - P)) / len(fs)
        rd += np.sum(wgt * (1 - P) ** 2) / len(fs)
    return {"trans_dark": float(tr), "refl_photon": float(rp), "refl_dark": float(rd)}


def lz_prediction(case):
    """Transverse (polarization y, B0 = 0 or B0 || x). Returns spectrally averaged channels and validity data."""
    k, om, wgt = spectral_weights()
    if case.theta is None:
        fs = [lambda o: np.ones_like(o)]
    elif case.theta == 0 and case.pol == "y":
        fs = [lambda o: o / (o - g.OMEGA_C), lambda o: o / (o + g.OMEGA_C)]
    else:
        return None
    tr = rp = rd = 0.0
    valid, info = True, []
    for fn in fs:
        f = fn(om)
        Pd, Lc, kbar2, n_c = lz_params(case, om, f)
        i0 = int(np.argmin(np.abs(k - g.K0)))
        kb0 = np.sqrt(kbar2[i0])
        ok = (kb0 * Lc[i0] if np.ndim(Lc) else kb0 * Lc) >= 3 and case.eta * f[i0] * (n_c[i0] if np.ndim(n_c) else n_c) <= 0.25 * kbar2[i0]
        valid = valid and bool(ok)
        info.append({"f": float(f[i0]), "Lc": float(Lc[i0] if np.ndim(Lc) else Lc), "kbar": float(kb0),
                     "Pd_carrier": float(Pd[i0]), "n_c": float(n_c[i0] if np.ndim(n_c) else n_c),
                     "reflecting": bool(f[i0] * g.N_HI > om[i0] ** 2)})
        tr += np.sum(wgt * Pd) / len(fs)
        rp += np.sum(wgt * Pd * (1 - Pd)) / len(fs)
        rd += np.sum(wgt * (1 - Pd) ** 2) / len(fs)
    return {"trans_dark": float(tr), "refl_photon": float(rp), "refl_dark": float(rd), "valid": valid, "parts": info}


def wkb_longitudinal(case):
    """Cold longitudinal dark photon (B0 = 0, pol x): local k^2(n) = om^2 - Om^2 (om^2 - n)/(om^2 - (1 + eta^2) n);
    WKB tunnelling exp(-2 int kappa dx) between the turning point n_t and the cold resonance n_p = om^2/(1 + eta^2),
    spectrally averaged."""
    k, om, wgt = spectral_weights()
    eta, OD2 = case.eta, g.OMEGA_D ** 2
    T = np.zeros(om.size)
    sel = np.linspace(0, om.size - 1, 15).astype(int)
    for i in sel:
        o2 = om[i] ** 2
        n_t = o2 * (o2 - OD2) / (o2 * (1 + eta ** 2) - OD2)
        n_p = o2 / (1 + eta ** 2)
        D = g.N_HI - g.N_LO
        xt = g.XU + case.w * np.arctanh(2 * (n_t - g.N_LO) / D - 1)
        xp = g.XU + case.w * np.arctanh(2 * (n_p - g.N_LO) / D - 1)

        def kap(x):
            nn = g.density(x, case.w)
            kk2 = o2 - OD2 * (o2 - nn) / (o2 - (1 + eta ** 2) * nn)
            return np.sqrt(max(-kk2, 0.0))
        # substitute x = xp - s^2 to remove the inverse-square-root endpoint singularity
        smax = np.sqrt(xp - xt)
        val = quad(lambda s: 2 * s * kap(xp - s * s), 0, smax, limit=200)[0]
        T[i] = np.exp(-2 * val)
    Tm = np.interp(np.arange(om.size), sel, T[sel])
    o2c = g.K0 ** 2 + OD2
    n_p0 = o2c / (1 + eta ** 2)
    xp0, npr = ramp_cross(n_p0, case.w)
    return {"T_wkb": float(np.sum(wgt * Tm)), "T_wkb_carrier": float(T[sel][np.argmin(np.abs(k[sel] - g.K0))]),
            "Lc_pole": float(n_p0 / npr), "kref_Lc": float(g.K0 * n_p0 / npr), "n_t": float(o2c * (o2c - OD2) / (o2c * (1 + eta ** 2) - OD2)),
            "n_p": float(n_p0)}


def local_dispersion_check():
    """Check the longitudinal local dispersion against the cold 19x19 matrix at several densities (returns max |dw|)."""
    eta, worst = 0.3, 0.0
    for nw in (0.3, 1.0, 1.9):
        for k in (0.5, 1.0):
            lam = np.linalg.eigvals(system_matrix(k, nw, np.zeros(3), eta))
            om = np.sort(np.abs((1j * lam).real))
            # roots of  om^2 (Q - OD^2) (1 - ...)  : (om^2-n) (om^2 - k^2 - OD^2) - eta^2 n (om^2 - k^2)... solved numerically
            f = lambda o: (o ** 2 - k ** 2 - g.OMEGA_D ** 2) * (o ** 2 - nw) - eta ** 2 * nw * (o ** 2 - k ** 2)  # noqa: E731
            roots = np.sort(np.roots([1, -(nw + k ** 2 + g.OMEGA_D ** 2 + eta ** 2 * nw), nw * (k ** 2 + g.OMEGA_D ** 2) + eta ** 2 * nw * k ** 2])
                            .real)
            wr = np.sqrt(roots[roots > 0])
            for w_ in wr:
                worst = max(worst, float(np.min(np.abs(om - w_))))
            del f
    return worst


def analytic(case):
    out = {}
    if case.pol in ("y", "z") and (case.theta is None or (case.theta == 0 and case.pol == "y")):
        out["lz"] = lz_prediction(case)
        out["refined"] = refined_prediction(case)
    if case.theta is None and case.pol == "x":
        out["wkb"] = wkb_longitudinal(case)
    return out


def run_and_save(case, N=4096, rtol=1e-12, tag=""):
    r = run_cold(case, N=N, rtol=rtol)
    r["analytic"] = analytic(case)
    r["repository_commit"] = git_sha()
    r["command"] = f"python studies/r3_ref.py {case.eta} {case.w} {case.theta} {case.pol} N={N} rtol={rtol} ntrunc={case.ntrunc}"
    OUT.mkdir(exist_ok=True)
    (OUT / f"ref_{case.name}_N{N}{tag}.json").write_text(json.dumps(r, indent=1) + "\n")
    ch = r["channels"]
    print(f"{case.name} N={N}: drift {r['energy_drift']:.1e} dark_r {ch['refl_dark']:.4f} ph_r {ch['refl_photon']:.4f} "
          f"dark_t {ch['trans_dark']:.4f} ph_t {ch['trans_photon']:.4f} stored {ch['stored']:.4f} beyond {ch['beyond']:.4f} "
          f"closure {ch['closure']:.1e} budget res {r['budget']['residual']:.1e} ({r['wall_s']:.0f}s)", flush=True)
    return r


def git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def main(argv):
    OUT.mkdir(exist_ok=True)
    mode = argv[0]
    if mode == "smoke":
        c = Case(0.3, 4.0, None, "y")
        r = run_cold(c, N=1024, rtol=1e-10)
        print(json.dumps({k: r[k] for k in ("energy_drift", "channels", "budget", "wall_s")}, indent=1))
    elif mode == "check":
        print("worst |omega| mismatch of the 19x19 longitudinal roots vs the quadratic:", local_dispersion_check())
    elif mode == "case":
        eta, w, th, pol = float(argv[1]), float(argv[2]), argv[3], argv[4]
        N = int(argv[5]) if len(argv) > 5 else 4096
        rtol = float(argv[6]) if len(argv) > 6 else 1e-12
        nt = int(argv[7]) if len(argv) > 7 else 0
        run_and_save(Case(eta, w, None if th == "none" else int(th), pol, ntrunc=nt), N=N, rtol=rtol)
    elif mode == "batch":
        for c in batch_cases(argv[1]):
            if not (OUT / f"ref_{c.name}_N4096.json").exists():
                run_and_save(c)
    else:
        raise SystemExit("unknown mode")


def batch_cases(which):
    cs = []
    if which in ("main", "all"):
        for pol in ("y", "x"):
            for eta in g.ETAS:
                for w in g.W_LIST:
                    cs.append(Case(eta, w, None, pol))
        for th in g.THETAS:
            for pol in ("x", "y", "z"):
                if th == 0 and pol == "z":
                    continue
                for w in (2.0, 4.0, 8.0):
                    cs.append(Case(0.3, w, th, pol))
    if which in ("scan", "all"):  # follow-up G eta scan (r3_followup_gates.py)
        for eta in (0.05, 0.1, 0.2):
            for w in (4.0, 8.0, 12.0):
                cs.append(Case(eta, w, None, "y"))
    if which in ("trunc", "all"):  # band-limited twins of the kinetic cases (Nx 512 and 768)
        for nt in (512, 768):
            for c in kinetic_cases():
                cs.append(Case(c.eta, c.w, c.theta, c.pol, ntrunc=nt))
    return cs


def kinetic_cases():
    cs = []
    for w in (2.0, 4.0, 8.0):
        cs.append(Case(0.3, w, None, "y"))
        cs.append(Case(0.3, w, None, "x"))
    for w in (4.0, 8.0, 12.0):
        cs.append(Case(0.15, w, None, "y"))
    cs.append(Case(0.3, 4.0, 0, "y"))
    for pol in ("x", "y", "z"):
        cs.append(Case(0.3, 4.0, 45, pol))
    for pol in ("y", "z"):
        cs.append(Case(0.3, 4.0, 90, pol))
    return cs


if __name__ == "__main__":
    main(sys.argv[1:])
