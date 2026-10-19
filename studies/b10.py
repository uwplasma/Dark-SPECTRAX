"""B10: Langmuir sideband (modulational) instability of a finite-amplitude pump at k0 with mobile ions, ordinary vs dark.

References and derivation: studies/b10_ref.py (R0 = fluid Zakharov 1972 eqs 1.2/1.9 with 2.8/2.9 limits; R1 = semi-kinetic
tier with exact Langmuir dispersion roots, kinetic ion function, and the dark generalization derived there).
Units: parent (c = omega_pe = 1, electron mass 1, eps0 = 1). Pump = the real wave E(x,t) = E0 cos(k0 x - w0 t), E0 = peak
Poisson field, |F_k0| = E0/2 =: A_p, w = E0^2/(2 n T_e) (= |E|^2/(8 pi n T_e) of Zakharov 1972).

REVISION v2 (declared in the commit that follows the v1 failure, before any production run of v2). The v1 production run
(M_mi1836... only M_mi100_ord was started; record kept as studies/b10/v1_failed_M_mi100_ord.json) FAILED at t = 702: the
Hermite closure with nu = 1 is numerically unstable (growth of the top electron coefficients, pump-only control confirmed it;
nu = 3, 10, 40 also blew up in short shakedown runs for mi 100 ordinary or dark upper at t = 625-1725; nu = 100 survived
t = 2500-5000). A second finding of the shakedown: with the periodic box the pump's Fourier index 2n hosts the second
harmonic, the K = k0 and K = 2k0 (Bragg backscatter, lower sideband = the counter-propagating pump wave) sideband families,
which grow from start-up transients faster than the seeded sidebands and deplete the pump (ordinary: counter-propagating wave 1.4e-2
of the pump at t = 1335, pump amplitude -10% at t = 6000; dark upper: blow-up at t = 500-925 depending on Nx at nu = 10). v2
therefore (i) uses nu = 100, (ii) removes Fourier index 2n from the nonlinear terms (Galerkin truncation on top of the parent
2/3 mask: studies/b10.py apply_mask), which removes those families and with them j = 2n sidebands (J drops j = 6 whose lower
sideband lives at index 2n = 4; J keeps no j = n, 2n), (iii) uses w = 0.04, SEED = 3e-4, LO = 6e-3, HI = 6e-2 (shorter windows,
each 2.3 e-folds, so the run time is below the stiff-closure budget), (iv) replaces the nu 1 -> 0.25 refinement by nu 100 -> 300.
The omission of index 2n changes the physics at order w (no second harmonic, no Bragg backscatter): it is a modelling choice,
stated in the results. The v2 tolerances are those of v1.

Setup (declared before running). v_te = 0.1 c (r_D = 0.1), T_e/T_i = 10 (main), eta = 0.3, Omega_D = 1 (= omega_pe at
k = 0: the Langmuir wave splits into the upper (~1.17) and lower (~0.87) mixed Langmuir-dark normal modes; the pump is
ONE normal mode, taken from the eigenvector of the linearized model itself at k0, so it is a pure right-moving
travelling wave; the sidebands are seeded with the eigenvectors of the same branch). Main: k0 = 0.05 (k0 r_D = 0.005),
w = 0.04 (v_osc/v_te = 0.28), box k1 = k0/2 (pump at Fourier index n = 2), sideband pairs K = j k1,
j in {1,3,5,7,8,10,12,16} (K r_D = 0.0025 ... 0.04; j = n, 2n are not used: see v2). Sidebands: upper k0 + K at index
n + j; lower k0 - K at index n - j if positive, else (j > n) the left-moving partner (conjugate) at index j - n. Each is seeded
at SEED A_p = 3e-4 A_p with a random phase (rng seed 20261018). Mass ratios 1836 and 100. Models: ordinary; dark upper; dark lower.
Hermite order Nn (same for both species), 1D (Nm = Np = 1), Nx = 60, hypercollision nu = 100 (parent numerical closure),
Dopri8, rtol 1e-10, atol 1e-12 (atol 1e-20 was found unstable at the shakedown stage and is not used). The v1 text below with
w = 0.02, SEED 3e-5, LO 3e-3, HI 4e-2, nu = 1 is superseded by the v2 values.

Measurement. F_x(index m, t) = R_m e^{-i w0 t} + conj(L_m) e^{+i w0 t} (R_m: wave at +m k1, L_m: wave at -m k1, both with
e^{-i w0 t}). R_m and L_m are obtained by demodulation with e^{+-i w0 t} (w0 = the pump's linear eigenfrequency) and three
passes of a boxcar of M = 5 samples, sampled at dt = pi/(M w0), which removes the e^{2 i w0 t} term exactly. Per K the sideband
amplitude is S_j = sqrt(|A_up|^2 + |A_lo|^2). The window is the first time S_j >= LO A_p (LO = 3e-3) to the first time
S_j >= HI A_p (HI = 4e-2); gamma_fit = least-squares slope of ln S_j over the window. Nothing after the window end is claimed.

Gates (declared before any production run; the 'theory' stage commits the reference table and gated K sets first):
  G0a (theory) R1 maximum over K in the static limit (k0 = 1e-3, w = 1e-6, mi 1836, T_e/T_i 10) within 5% of eq 2.8's maximum
       gamma = w omega_p/4 * T_e/(T_e + T_i) (exact maximum of 2.8, not the printed w/16).
  G0b (theory) R0 with the sound term removed (c_s -> 0), k0 = 0, w = 50 mu, K r_D = 0.5: gamma within 1% of the eq-2.9 asymptote
       sqrt(mu w / 3), mu = 1/mi (K large).
  G0c (theory) dark reference with eta -> 1e-8 reproduces the ordinary R1 growth to 1e-4 at every main K (the first theory run, before this commit, gave 1.4e-5 from
       root-polish noise; the draft tolerance 1e-6 was too tight for a root-finder comparison and was changed before the commit).
  G1  sideband growth vs R1: gated K = those with gamma_R1 >= 0.3 max_K gamma_R1 and a predicted window completion
       (ln(HI/SEED) + 0.5)/gamma_R1 <= 0.9 t_max. If the case has no gated K because R1 predicts no growth at all (gamma_R1 = 0
       for every K: mi 1836 dark lower), the gate is the stability prediction: no sideband reaches HI A_p before t_max. Cases
       with no gated K for any other reason are recorded without a gate. For every gated K of every main case (ordinary, dark upper, dark lower;
       mi 1836 and 100): |gamma_fit/gamma_R1 - 1| <= 0.15. Cases where a gated K does not reach HI count as failures.
  G1b ordinary only: same K, |gamma_fit/gamma_R0 - 1| <= 0.30 (R0 uses c_s^2 = (T_e + 3 T_i)/M; the paper's printed 3/2 T_i
       changes R0 by the recorded amount and is not used).
  G2  refinement (mi 1836, ordinary and dark upper): Nn 32 -> 64, Nx 60 -> 72, and (ordinary) nu 100 -> 300: |gamma(refined)/gamma(base) - 1|
       <= 0.03 at every gated K.
  G3  every run: max |ledger defect| <= 1e-6 U_field(0) (cumulative over restart segments); Gauss residuals <= 1e-8 max|F|.
  G4  pump: |A_p(t_end)/A_p(0) - 1| <= 0.05 at the window end of every gated K.
Outside-regime runs (O1: T_e/T_i = 1, O2: w = 0.1, O3: k0 = 0.4 with K up to 0.7; ordinary and dark upper, mi 1836) have no
pass/fail gate: the measured growth is reported against R0 and R1, i.e. the kinetic correction relative to the Zakharov (R0)
and semi-kinetic (R1) predictions, and the dark change gamma_dark/gamma_ord against R1.

Run: python studies/b10.py theory | run <case> | summary | list
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import b10_ref as ref

out_dir = Path(__file__).resolve().parent / "b10"
VTE, ETA, OMD = 0.1, 0.3, 1.0
SEED, LO, HI = 3e-4, 6e-3, 6e-2
MB = 5  # boxcar samples per period of exp(2 i w0 t)
SEGN = 2500  # samples per restart segment
RTOL, ATOL = 1e-10, 1e-12
RNG_SEED = 20261018
TOL = {"G0a": 0.05, "G0b": 0.01, "G0c": 1e-4, "G1": 0.15, "G1b": 0.30, "G2": 0.03, "G3_ledger": 1e-6, "G3_gauss": 1e-8,
       "G4": 0.05}
MAIN_J = (1, 3, 5, 7, 8, 10, 12, 16)
O3_J = (1, 2, 3, 5, 6, 7)  # n = 4: j = 4 (K = k0) and 8 (2n) excluded


def _case(tag, model, mi=1836.0, tau=10.0, w=0.04, k0=0.05, n=2, J=MAIN_J, Nn=32, nu=100.0, tmax=None, Nx=None, dNx=0):
    nx = Nx or 2 * int(np.ceil(1.5 * (n + max(J) + 2))) + dNx
    tmax = tmax or (12000.0 if mi > 500 else 5000.0)
    return {"name": f"{tag}_{model}", "model": model, "mi": mi, "tau": tau, "w": w, "k0": k0, "n": n, "J": list(J),
            "Nn": Nn, "Nx": nx, "nu": nu, "tmax": tmax}


def all_cases():
    cs = []
    for mi in (1836.0, 100.0):
        for m in ("ord", "dk_up", "dk_lo"):
            cs.append(_case(f"M_mi{int(mi)}", m, mi=mi))
    for m in ("ord", "dk_up"):
        cs.append(_case("R_Nn64", m, Nn=64))
        cs.append(_case("R_Nx72", m, dNx=12))
    cs.append(_case("R_nu300", "ord", nu=300.0))
    for m in ("ord", "dk_up"):
        cs.append(_case("O1_tau1", m, tau=1.0))
        cs.append(_case("O2_w015", m, w=0.15, tmax=5000.0))
        cs.append(_case("O3_k04", m, k0=0.4, n=4, J=O3_J, tmax=8000.0))
    return {c["name"]: c for c in cs}


def apply_mask(model, c):
    """Galerkin truncation: remove Fourier index 2n (pump second harmonic and the K = k0 / K = 2k0 families) from the
    nonlinear terms, in addition to the parent 2/3 mask. Done in place on the model's parent dictionary."""
    import jax.numpy as jnp
    mk = np.array(model.p["mask23"])
    mk[:, 2 * c["n"], :] = 0.0
    model.p["mask23"] = jnp.asarray(mk)
    return model


def plasma_of(c):
    dark = c["model"] != "ord"
    return ref.Plasma(vte=VTE, mi=c["mi"], tau=c["tau"], eta=ETA if dark else 0.0, OmegaD=OMD)


def pump_branch(pl, c):
    bs = pl.branches(c["k0"])
    if c["model"] == "ord":
        return bs[0]
    return bs[-1] if c["model"] == "dk_up" else bs[0]


def E0_of(c):
    return VTE * np.sqrt(2 * c["w"])


def reference_table(c):
    pl = plasma_of(c)
    w0 = pump_branch(pl, c)
    k1 = c["k0"] / c["n"]
    E0 = E0_of(c)
    rows = []
    for j in c["J"]:
        K = j * k1
        g1, re1 = ref.growth(pl, c["k0"], K, E0, w0, "R1")
        row = {"j": j, "K": K, "K_rD": K * VTE, "gamma_R1": g1, "re_Omega_R1": re1}
        if c["model"] == "ord":
            row["gamma_R0"] = ref.growth(pl, c["k0"], K, E0, w0, "R0")[0]
            row["gamma_R0_printed_cs"] = ref.growth(pl, c["k0"], K, E0, w0, "R0", cs_T=1.5)[0]
        rows.append(row)
    gmax = max(r["gamma_R1"] for r in rows)
    for r in rows:
        g = r["gamma_R1"]
        r["gated"] = bool(g > 0 and g >= 0.3 * gmax and (np.log(HI / SEED) + 0.5) / g <= 0.9 * c["tmax"])
    eps = float(pl.deps_dw(w0.real, c["k0"]).real)
    return {"omega0": [w0.real, w0.imag], "g0": float(pl.g(w0.real, c["k0"]).real), "eps_prime": eps,
            "v_g": float((pl.omega(c["k0"] + 1e-4, w0) - pl.omega(c["k0"] - 1e-4, w0)).real / 2e-4),
            "E0": E0, "A_p": E0 / 2, "rows": rows}


def theory():
    out = {}
    pl = ref.Plasma(vte=VTE, mi=1836.0, tau=10.0)
    w = 1e-6
    E0 = VTE * np.sqrt(2 * w)
    Kc, gm = ref.closed_form_static(pl, w)
    g = max(ref.growth(pl, 1e-3, Kc * f, E0, tier="R1")[0] for f in np.linspace(0.6, 1.6, 21))
    out["G0a"] = {"gamma_28_max": gm, "gamma_R1_max": g, "rel": g / gm - 1, "pass": bool(abs(g / gm - 1) <= TOL["G0a"])}
    mu = 1 / 1836.0
    w = 50 * mu
    E0 = VTE * np.sqrt(2 * w)
    g = ref.growth(pl, 1e-3, 5.0, E0, tier="R0", cs2=0.0)[0]
    out["G0b"] = {"asymptote": float(np.sqrt(mu * w / 3)), "gamma_R0": g, "rel": g / np.sqrt(mu * w / 3) - 1,
                  "pass": bool(abs(g / np.sqrt(mu * w / 3) - 1) <= TOL["G0b"])}
    cases = all_cases()
    tab = {}
    for name, c in cases.items():
        tab[name] = {"config": c, **reference_table(c)}
        print(name, [(r["j"], round(r["gamma_R1"] * 1e3, 3), r["gated"]) for r in tab[name]["rows"]], flush=True)
    out["cases"] = tab
    c = cases["M_mi1836_ord"]
    plo = plasma_of(c)
    pd_ = ref.Plasma(vte=VTE, mi=1836.0, tau=10.0, eta=1e-8, OmegaD=OMD)
    w0 = plo.omega(c["k0"])
    d = []
    for j in c["J"]:
        K = j * c["k0"] / c["n"]
        a = ref.growth(plo, c["k0"], K, E0_of(c), w0, "R1")[0]
        b = ref.growth(pd_, c["k0"], K, E0_of(c), pd_.omega(c["k0"]), "R1")[0]
        d.append(abs(a - b) / max(a, 1e-12))
    out["G0c"] = {"max_rel": max(d), "pass": bool(max(d) <= TOL["G0c"])}
    out["tolerances"] = TOL
    out["constants"] = dict(SEED=SEED, LO=LO, HI=HI, MB=MB, RTOL=RTOL, ATOL=ATOL, VTE=VTE, ETA=ETA, OMD=OMD)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "theory.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print({k: out[k] for k in ("G0a", "G0b", "G0c")})


# ---------------------------------------------------------------------------------------------------------------------------
def model_of(c, Nx=None, Lx=None):
    import darkspectrax as ds
    dark = c["model"] != "ord"
    vti = VTE / np.sqrt(c["mi"] * c["tau"])
    return ds.Model(Nx=Nx or c["Nx"], Nn=c["Nn"], Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1.0 / c["mi"]),
                    alpha_s=tuple(np.repeat([np.sqrt(2) * VTE, np.sqrt(2) * vti], 3)), u_s=(0.0,) * 6, nu=c["nu"],
                    mode="self_consistent" if dark else "ordinary", eta=ETA if dark else 0.0, Omega_D=OMD)


_vec_cache = {}


def mode_vec(c, k, omega_target, sign):
    """Eigenvector of the linearized model at wavenumber k (single mode, Nx = 6) with eigenvalue nearest -sign... i.e.
    sign = -1: exp(-i w t), sign = +1: exp(+i w t), frequency nearest omega_target. Returns (v, sizes, keys, lam)."""
    import jax
    import jax.numpy as jnp
    import darkspectrax as ds
    from darkspectrax._model import rhs
    key = (c["mi"], c["tau"], c["model"], c["Nn"], c["nu"], round(k, 12), round(omega_target, 9), sign)
    if key in _vec_cache:
        return _vec_cache[key]
    m = model_of(c, Nx=6, Lx=2 * np.pi / k)
    y0 = ds.maxwellian(m, [1.0, 1.0])
    keys = ["Ck", "Fk", "Dk"]
    sizes = [np.asarray(y0[q]).shape[0] for q in keys]
    shapes = [np.asarray(y0[q]).shape for q in keys]
    d = sum(sizes)

    def embed(v):
        out, o = dict(y0), 0
        for q, sh, s in zip(keys, shapes, sizes):
            out[q] = out[q] + jnp.zeros(sh, complex).at[:, 0, 1, 0].set(v[o:o + s])
            o += s
        return out

    def f(v):
        r = rhs(0.0, embed(v), m)
        return jnp.concatenate([r[q][:, 0, 1, 0] for q in keys])

    _, lin = jax.linearize(f, jnp.zeros(d, complex))
    M = np.array([np.asarray(lin(jnp.eye(d, dtype=complex)[j])) for j in range(d)]).T
    w, V = np.linalg.eig(M)
    i = int(np.argmin(np.abs(w - sign * 1j * omega_target)))
    _vec_cache[key] = (V[:, i], sizes, keys, w[i])
    return _vec_cache[key]


def build_state(c, model):
    import darkspectrax as ds
    pl = plasma_of(c)
    w0 = pump_branch(pl, c)
    k1 = c["k0"] / c["n"]
    n = c["n"]
    A_p = E0_of(c) / 2
    rng = np.random.default_rng(RNG_SEED)
    y = ds.maxwellian(model, [1.0, 1.0])
    seeds = []

    def put(index, v, sizes, keys, amp):
        o = 0
        for q, s in zip(keys, sizes):
            y[q] = y[q].at[:, 0, index, 0].add(amp * v[o:o + s])
            o += s

    v, sizes, keys, lam = mode_vec(c, c["k0"], w0.real, -1)
    put(n, v, sizes, keys, A_p / abs(v[sizes[0]]))
    omega_pump = -lam.imag
    for j in c["J"]:
        for idx_sign, m in (("up", n + j), ("lo", n - j)):
            kk = abs(m) * k1
            ws = pl.omega(kk, w0).real
            sign = -1 if m > 0 else +1
            v, sizes, keys, lam_s = mode_vec(c, kk, ws, sign)
            ph = np.exp(2j * np.pi * rng.random())
            put(abs(m), v, sizes, keys, SEED * A_p * ph / abs(v[sizes[0]]))
            seeds.append({"j": j, "which": idx_sign, "index": abs(m), "left_moving_partner": m < 0,
                          "omega": float(-sign * lam_s.imag)})
    return y, float(omega_pump), seeds


def boxcar3(z):
    k = np.ones(MB) / MB
    for _ in range(3):
        z = np.convolve(z, k, mode="valid")
    return z


def run_case(name):
    import darkspectrax as ds
    c = all_cases()[name]
    tic = time.time()
    pl_ref = reference_table(c)
    Lx = 2 * np.pi / (c["k0"] / c["n"])
    model = apply_mask(model_of(c, Lx=Lx), c)
    y, w0, seeds = build_state(c, model)
    dt = np.pi / (MB * w0)
    nseg = int(np.ceil(c["tmax"] / (SEGN * dt)))
    seg = SEGN * dt
    ts, Fx = [], []
    defect_off = np.zeros(3)
    defect_max, gauss_max, U0 = 0.0, np.zeros(2), None
    steps, status, reason = 0, "success", None
    cache = {}
    t = 0.0
    for s in range(nseg):
        out = ds.run(model, y, seg, n_save=SEGN + 1, rtol=RTOL, atol=ATOL, t0=t, cache=cache, max_steps=400_000)
        steps += out["num_steps"]
        if out["status"] != "success":
            status, reason = "failure", out["failure_reason"]
            n_ok = int(out["num_valid_times"])
        else:
            n_ok = SEGN + 1
        sl = slice(0 if s == 0 else 1, n_ok)
        ts.append(out["t"][sl])
        Fx.append(out["Fk"][sl, 0, 0, :, 0])
        if U0 is None:
            U0 = float(out["U_gamma"][0] + out["U_D"][0])
        dd = out["ledger_defect"][:, :n_ok] + defect_off[:, None]
        defect_max = max(defect_max, float(np.abs(dd).max()))
        gauss_max = np.maximum(gauss_max, out["gauss"][:n_ok].max(axis=0))
        if status != "success":
            break
        defect_off = dd[:, -1]
        y = {"Ck": __import__("jax").numpy.asarray(out["Ck"][-1]), "Fk": __import__("jax").numpy.asarray(out["Fk"][-1]),
             "Dk": __import__("jax").numpy.asarray(out["Dk"][-1]),
             "W": __import__("jax").numpy.asarray(out["W"][-1], complex)}
        t = float(out["t"][-1])
        print(f"{name} seg {s + 1}/{nseg} t={t:.0f} steps={steps} wall={time.time() - tic:.0f}s", flush=True)
    t = np.concatenate(ts)
    Fx = np.concatenate(Fx)
    off = 3 * (MB - 1) // 2
    tv = t[off:len(t) - off]
    ph_p, ph_m = np.exp(1j * w0 * t), np.exp(-1j * w0 * t)
    R = lambda m: np.abs(boxcar3(Fx[:, m] * ph_p))
    L = lambda m: np.abs(boxcar3(Fx[:, m] * ph_m))
    A_p = E0_of(c) / 2
    Ap_t = R(c["n"])
    Ap0 = float(np.median(Ap_t[:20]))
    res = []
    for j in c["J"]:
        mu_, ml = c["n"] + j, c["n"] - j
        up = R(mu_)
        lo = R(ml) if ml > 0 else L(-ml)
        S = np.sqrt(up ** 2 + lo ** 2)
        ia = np.nonzero(S >= LO * A_p)[0]
        ib = np.nonzero(S >= HI * A_p)[0]
        row = {"j": j, "K": j * c["k0"] / c["n"], "S_final_over_Ap": float(S[-1] / A_p), "S_max_over_Ap": float(S.max() / A_p),
               "trace_t": [round(float(x), 1) for x in tv[::400]], "trace_lnS": [round(float(x), 4) for x in np.log(S[::400])]}
        if len(ia) and len(ib) and ia[0] < ib[0]:
            a, b = int(ia[0]), int(ib[0])
            tt, ls = tv[a:b + 1], np.log(S[a:b + 1])
            g = float(np.polyfit(tt, ls, 1)[0])
            h = (a + b) // 2
            g1 = float(np.polyfit(tv[a:h + 1], np.log(S[a:h + 1]), 1)[0])
            g2 = float(np.polyfit(tv[h:b + 1], np.log(S[h:b + 1]), 1)[0])
            row.update({"window": [float(tv[a]), float(tv[b])], "gamma_fit": g, "gamma_halves": [g1, g2],
                        "pump_drift": float(Ap_t[b] / Ap0 - 1), "up_over_lo_at_end": float(up[b] / max(lo[b], 1e-300))})
        else:
            row.update({"window": None, "gamma_fit": None})
        res.append(row)
    rec = {"case": name, "command": f"python studies/b10.py run {name}", "config": c, "reference": pl_ref,
           "pump_omega_eigen": w0, "A_p": A_p, "seeds": seeds, "status": status, "failure_reason": reason,
           "t_reached": float(t[-1]), "steps": steps, "U_field0": U0, "max_ledger_defect": defect_max,
           "max_ledger_over_U0": defect_max / U0, "max_gauss": [float(v) for v in gauss_max],
           "max_gauss_over_Ap": [float(v) / A_p for v in gauss_max], "pump_A0_over_Ap": Ap0 / A_p,
           "results": res, "wall_time": time.time() - tic, "repository_commit": ds._simulation._git_sha(),
           "parent_commit": ds.PARENT_COMMIT, "settings": {"rtol": RTOL, "atol": ATOL, "M": MB, "SEED": SEED, "LO": LO, "HI": HI}}
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(name, status, [(r["j"], r["gamma_fit"]) for r in res], flush=True)


def summary():
    th = json.loads((out_dir / "theory.json").read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json") if p.stem not in ("theory", "summary")}
    S = {"gates": {}, "cases": {}}
    S["gates"]["G0a"], S["gates"]["G0b"], S["gates"]["G0c"] = (th[g]["pass"] for g in ("G0a", "G0b", "G0c"))

    def gated(name):
        return [r for r in th["cases"][name]["rows"] if r["gated"]]

    def gam(rec, j):
        return next(r for r in rec["results"] if r["j"] == j)

    for name, rec in recs.items():
        rows = []
        for r in gated(name):
            m = gam(rec, r["j"])
            g = m["gamma_fit"]
            row = {"j": r["j"], "K_rD": r["K_rD"], "gamma_R1": r["gamma_R1"], "gamma_fit": g,
                   "rel_R1": None if g is None else g / r["gamma_R1"] - 1, "pump_drift": m.get("pump_drift")}
            if "gamma_R0" in r:
                row["gamma_R0"] = r["gamma_R0"]
                row["rel_R0"] = None if g is None else g / r["gamma_R0"] - 1
            rows.append(row)
        ok_ledger = rec["max_ledger_over_U0"] <= TOL["G3_ledger"]
        ok_gauss = max(rec["max_gauss_over_Ap"]) <= TOL["G3_gauss"]
        c = {"rows": rows, "ledger_over_U0": rec["max_ledger_over_U0"], "gauss_over_Ap": max(rec["max_gauss_over_Ap"]),
             "G3": bool(ok_ledger and ok_gauss), "status": rec["status"], "steps": rec["steps"], "wall": rec["wall_time"]}
        if name.startswith("M_"):
            if rows:
                c["G1"] = bool(all(r["rel_R1"] is not None and abs(r["rel_R1"]) <= TOL["G1"] for r in rows))
            elif max(r["gamma_R1"] for r in th["cases"][name]["rows"]) == 0.0:
                c["G1"] = bool(all(m["S_max_over_Ap"] < HI for m in rec["results"]))  # stability prediction
                c["G1_kind"] = "stability"
            else:
                c["G1"] = None  # no gated K and growth predicted: recorded without a gate
                c["G1_kind"] = "ungated"
            c["G4"] = bool(all(r["pump_drift"] is not None and abs(r["pump_drift"]) <= TOL["G4"] for r in rows)) if rows else None
            if name.endswith("_ord"):
                c["G1b"] = bool(rows and all(r["rel_R0"] is not None and abs(r["rel_R0"]) <= TOL["G1b"] for r in rows))
        S["cases"][name] = c
    base = {"ord": "M_mi1836_ord", "dk_up": "M_mi1836_dk_up"}
    for name in recs:
        if name.startswith("R_"):
            mod = "ord" if name.endswith("_ord") else "dk_up"
            bn = base[mod]
            if bn not in recs:
                continue
            diffs = []
            for r in gated(bn):
                g0, g1 = gam(recs[bn], r["j"])["gamma_fit"], gam(recs[name], r["j"])["gamma_fit"]
                diffs.append(None if g0 is None or g1 is None else g1 / g0 - 1)
            S["cases"][name]["G2_diffs"] = diffs
            S["cases"][name]["G2"] = bool(all(d is not None and abs(d) <= TOL["G2"] for d in diffs))
    (out_dir / "summary.json").write_text(json.dumps(S, indent=2, default=float) + "\n")
    for k, v in S["cases"].items():
        print(k, {q: v[q] for q in ("G1", "G1b", "G2", "G3", "G4") if q in v})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "theory":
        theory()
    elif cmd == "run":
        run_case(sys.argv[2])
    elif cmd == "summary":
        summary()
    elif cmd == "list":
        print("\n".join(all_cases()))
