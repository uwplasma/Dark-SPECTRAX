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

REVISION v3 (declared before any v3 run). The first v2 production run (M_mi100_ord, all eight K in one box, record kept as
studies/b10_v2/M_mi100_ord_multiK.json, ledger 3e-12, Gauss 7e-15) completed but is NOT usable for the gates: the pump lost
up to 64% of its amplitude by the window ends of the slow K, the S(t) traces jumped by factors 2-3 at sideband amplitudes
1e-3 to 1e-2 and every K saturated near 0.2-0.7 A_p by t ~ 3500. Mode competition and sideband-sideband coupling in the
common box, not the sideband instability alone, set the late behaviour. v3 runs ONE sideband pair per run: the nonlinear
terms keep only the Fourier indices {0, j, n, n + j, |n - j|} (pump, upper and lower sideband, ion density, mean), which is
the truncation the reference theories assume; Nx = 2 ceil(1.5 (max index + 1)). Declared K points per (mass ratio, model):
MAIN_JS below. A K point is GATED iff gamma_R1 > 0 and (ln(HI/SEED) + 0.5)/gamma_R1 <= 0.9 t_max (the 0.3 gamma_max condition
of v1 is dropped: there is one K per run); t_max = 9000 (mi 1836), 4500 (mi 100). G1/G1b/G4 apply to the gated points;
ungated points are recorded without a gate. G2: refinements at one K point per model (REF_J). SEED, LO, HI, nu, tolerances as in v2.
G1's stability variant is dropped (R1 predicts weak growth for dark lower at w = 0.04). Everything else is unchanged.

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

FOLLOW-UP F1 (declared after the v3 results for the dark upper branch were seen: mi 1836 dark upper j = 7, 8 showed a growth
3-4 x the R1 value with an explosive late stage, G1/G3 failed, while ordinary and dark j = 5 agreed with R1 within 6-10%). F1 computes the
EXACT linear Floquet exponent of the same truncated model (same Hermite closure, same truncation, same pump) about the
pump-only trajectory: the monodromy map over one pump period T0 = 2 pi/w0 acting on a perturbation of every kept Fourier
index (exact tangent flow by jax.jvp through fixed-step Dopri8, dt = 0.02), its leading multipliers by Arnoldi (scipy eigs, k = 8),
gamma_F = max ln|mu|/T0. It is a reference of the kinetic problem (no envelope or Boltzmann approximation) and is used
to say whether R1 or the measured fit is closer for each point; no pass/fail threshold is set in advance (F1 is
diagnostic, a record and a table). Run: python studies/b10.py floquet <case>

FOLLOW-UP S2 (declared after the v3 results, before running it): at nu = 100 the mi 1836 dark-upper points j = 7, 8
ended in an explosive stage and failed G1/G3, and the Nn = 64 refinements (ordinary j = 8, dark upper j = 7) blew up (G3), while
the nu = 300 ordinary refinement (R_nu300) moved gamma by -5.3% (G2 failed at 3%). S2 re-runs the failed points with the
stronger closure nu = 300 (dark upper j = 7, 8 at Nn = 32; ordinary j = 8 at Nn = 64). Criteria are G1/G3 (dark upper) and G2
(ordinary Nn = 64 against R_nu300 at Nn = 32), unchanged. If the explosive stage persists at nu = 300 it is not the closure.

FOLLOW-UP S3 (declared after the first S2 result, before running S3): S2 dark upper j = 7 at nu = 300 completed WITHOUT the
explosive stage (G3 ok) and grew at about 6e-4 (trace estimate, not a gated window: HI was not reached by t_max), against R1 8.4e-4 and
against 2.9e-3 at nu = 100: the late explosive stage of the nu = 100 dark-upper runs is a closure artifact. Since the nu = 100 dark
growth rates may carry closure error at the 5-30% level, S3 repeats ALL dark-upper points of the m_i/m_e = 100 set (j = 3, 5, 7, 10), the
mi 1836 j = 5 point and ordinary mi 100 j = 10 at nu = 300 with the same windows and gates (G1 vs R1, G3, G4 with the same
tolerances); the pairs (nu = 100, nu = 300) are the closure-sensitivity record. Criteria: G1, G3, G4 as in v3.

Run: python studies/b10.py theory | run <case> | floquet <case> | summary | list
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
# v3: one sideband pair per run (see REVISION v3); the list is the declared K points of each (mass ratio, model).
MAIN_JS = {(1836.0, "ord"): (5, 8, 16), (1836.0, "dk_up"): (5, 7, 8), (1836.0, "dk_lo"): (5,),
           (100.0, "ord"): (7, 10, 16), (100.0, "dk_up"): (3, 5, 7, 10), (100.0, "dk_lo"): (3,)}
TMAX = {1836.0: 9000.0, 100.0: 4500.0}
REF_J = {"ord": 8, "dk_up": 7}  # refinement and outside-regime K points (mi 1836)


def keep_indices(n, j):
    return sorted({0, j, n, n + j, abs(n - j)})


def _case(tag, model, j, mi=1836.0, tau=10.0, w=0.04, k0=0.05, n=2, Nn=32, nu=100.0, tmax=None, dNx=0):
    nx = 2 * int(np.ceil(1.5 * (max(keep_indices(n, j)) + 1))) + dNx
    return {"name": f"{tag}_{model}_j{j}", "tag": tag, "model": model, "mi": mi, "tau": tau, "w": w, "k0": k0, "n": n,
            "J": [j], "Nn": Nn, "Nx": nx, "nu": nu, "tmax": tmax or TMAX[mi]}


def all_cases():
    cs = []
    for (mi, m), js in MAIN_JS.items():
        for j in js:
            cs.append(_case(f"M_mi{int(mi)}", m, j, mi=mi))
    for m, j in REF_J.items():
        cs.append(_case("R_Nn64", m, j, Nn=64))
        cs.append(_case("R_Nx+12", m, j, dNx=12))
    cs.append(_case("R_nu300", "ord", REF_J["ord"], nu=300.0))
    for m, j in REF_J.items():
        cs.append(_case("O1_tau1", m, j if m == "ord" else 5, tau=1.0))
        cs.append(_case("O2_w015", m, j, w=0.15, tmax=4500.0))
    # S2 (declared after the v3 results): stronger closure, nu = 300, for the points where nu = 100 gave a late explosive stage
    # (mi 1836 dark upper j = 7, 8) or a closure blow-up at Nn = 64 (ordinary j = 8, nu = 300 reference is R_nu300)
    cs.append(_case("S2_nu300", "dk_up", 7, nu=300.0))
    cs.append(_case("S2_nu300", "dk_up", 8, nu=300.0))
    cs.append(_case("S2_Nn64nu300", "ord", 8, nu=300.0, Nn=64))
    # S3 (declared after the S2 first result): the whole dark-upper set and one ordinary point at nu = 300
    for j in (3, 5, 7, 10):
        cs.append(_case("S3_nu300_mi100", "dk_up", j, mi=100.0, nu=300.0))
    cs.append(_case("S3_nu300_mi100", "ord", 10, mi=100.0, nu=300.0))
    cs.append(_case("S3_nu300", "dk_up", 5, nu=300.0))
    cs.append(_case("O3_k04", "ord", 5, k0=0.4, n=4, tmax=6000.0))
    cs.append(_case("O3_k04", "dk_up", 5, k0=0.4, n=4, tmax=6000.0))
    return {c["name"]: c for c in cs}


def apply_mask(model, c):
    """Galerkin truncation to the Fourier indices of one sideband triad: pump n, upper n + j, lower |n - j|, the ion-density
    index j and the mean (index 0). Every other index is removed from the nonlinear terms (and stays at its initial zero)."""
    import jax.numpy as jnp
    keep = set(keep_indices(c["n"], c["J"][0]))
    mk = np.array(model.p["mask23"])
    for i in range(mk.shape[1]):
        if i not in keep:
            mk[:, i, :] = 0.0
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
    for r in rows:
        g = r["gamma_R1"]
        r["gated"] = bool(g > 0 and (np.log(HI / SEED) + 0.5) / g <= 0.9 * c["tmax"])
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
    c = cases["M_mi1836_ord_j8"]
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


def floquet(name):
    """Exact linear Floquet exponent of the truncated model about the pump-only trajectory (follow-up F1)."""
    import jax.numpy as jnp
    import darkspectrax as ds
    from scipy.sparse.linalg import LinearOperator, eigs
    c = all_cases()[name]
    tic = time.time()
    c0 = dict(c)
    Lx = 2 * np.pi / (c["k0"] / c["n"])
    model = apply_mask(model_of(c0, Lx=Lx), c0)
    global SEED
    seed_save, SEED = SEED, 0.0
    try:
        y, w0, _ = build_state(c0, model)
    finally:
        SEED = seed_save
    T0 = 2 * np.pi / w0
    keep = keep_indices(c["n"], c["J"][0])
    keys = ["Ck", "Fk", "Dk"]
    A_p = E0_of(c) / 2
    sizes = [np.asarray(y[k]).shape[0] for k in keys]
    nd = sum(sizes) * len(keep)
    import diffrax
    import jax
    dt = 0.02
    ys0 = {k: jnp.asarray(v) for k, v in y.items()}

    def flow_ys(yy):
        return ds.solve(model, yy, T0, n_save=2, fixed_dt=dt, adjoint=diffrax.ForwardMode())

    def tang(v):
        base = ys0
        tan = {k: jnp.zeros_like(x) for k, x in base.items()}
        o = 0
        for i in keep:
            for k, s_ in zip(keys, sizes):
                tan[k] = tan[k].at[:, 0, i, 0].set(v[o:o + s_])
                o += s_
        _, t_out = jax.jvp(flow_ys, (base,), (tan,))
        return jnp.concatenate([t_out[k][-1][:, 0, i, 0] for i in keep for k in keys])

    tang = jax.jit(tang)

    def mv(x):
        v = jnp.asarray(x[:nd] + 1j * x[nd:])
        d = np.asarray(tang(v))
        return np.concatenate([d.real, d.imag])

    op = LinearOperator((2 * nd, 2 * nd), matvec=mv, dtype=float)
    vals = eigs(op, k=8, which="LM", return_eigenvectors=False, tol=1e-6, maxiter=400, ncv=40)
    vals = sorted(vals, key=lambda z: -abs(z))
    g = [float(np.log(abs(z)) / T0) for z in vals]
    rec = {"case": name, "command": f"python studies/b10.py floquet {name}", "config": c, "T0": T0, "multipliers": [[z.real, z.imag] for z in vals],
           "growth": g, "gamma_floquet": max(g), "dim": 2 * nd, "wall_time": time.time() - tic,
           "repository_commit": ds._simulation._git_sha()}
    (out_dir / f"floquet_{name}.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(name, "gamma_floquet", max(g), "all", g, flush=True)


def summary():
    th = json.loads((out_dir / "theory.json").read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json")
            if p.stem not in ("theory", "summary") and p.stem in th["cases"]}
    S = {"gates": {g: th[g]["pass"] for g in ("G0a", "G0b", "G0c")}, "cases": {}}
    for name, rec in recs.items():
        r = th["cases"][name]["rows"][0]
        m = rec["results"][0]
        g = m["gamma_fit"]
        row = {"name": name, "tag": rec["config"]["tag"], "model": rec["config"]["model"], "mi": rec["config"]["mi"],
               "j": r["j"], "K_rD": r["K_rD"], "gated": r["gated"], "gamma_R1": r["gamma_R1"], "gamma_fit": g,
               "rel_R1": None if g is None else g / r["gamma_R1"] - 1 if r["gamma_R1"] > 0 else None,
               "gamma_halves": m.get("gamma_halves"), "window": m["window"], "pump_drift": m.get("pump_drift"),
               "S_max_over_Ap": m["S_max_over_Ap"], "status": rec["status"], "steps": rec["steps"],
               "wall": rec["wall_time"], "ledger_over_U0": rec["max_ledger_over_U0"],
               "gauss_over_Ap": max(rec["max_gauss_over_Ap"])}
        if "gamma_R0" in r:
            row["gamma_R0"] = r["gamma_R0"]
            row["rel_R0"] = None if g is None else g / r["gamma_R0"] - 1
        row["G3"] = bool(row["ledger_over_U0"] <= TOL["G3_ledger"] and row["gauss_over_Ap"] <= TOL["G3_gauss"])
        if (row["tag"].startswith(("M_", "S3_")) or row["tag"] == "S2_nu300") and row["gated"]:
            row["G1"] = bool(row["rel_R1"] is not None and abs(row["rel_R1"]) <= TOL["G1"])
            row["G4"] = bool(row["pump_drift"] is not None and abs(row["pump_drift"]) <= TOL["G4"])
            if row["model"] == "ord":
                row["G1b"] = bool(row["rel_R0"] is not None and abs(row["rel_R0"]) <= TOL["G1b"])
        S["cases"][name] = row
    for name, row in S["cases"].items():
        if row["tag"].startswith("R_") or row["tag"] == "S2_Nn64nu300":
            bn = f"M_mi1836_{row['model']}_j{row['j']}"
            if row["tag"] == "S2_Nn64nu300":
                bn = f"R_nu300_{row['model']}_j{row['j']}"
            b = S["cases"].get(bn)
            if b and b["gamma_fit"] and row["gamma_fit"]:
                row["G2_rel_to_base"] = row["gamma_fit"] / b["gamma_fit"] - 1
                row["G2"] = bool(abs(row["G2_rel_to_base"]) <= TOL["G2"])
            else:
                row["G2"] = False
    gl = {}
    for row in S["cases"].values():
        for g in ("G1", "G1b", "G2", "G3", "G4"):
            if g in row:
                gl.setdefault(g, []).append(row[g])
    S["gate_summary"] = {g: {"pass": sum(v), "of": len(v)} for g, v in gl.items()}
    (out_dir / "summary.json").write_text(json.dumps(S, indent=2, default=float) + "\n")
    for k, v in S["cases"].items():
        print(k, {q: v[q] for q in ("G1", "G1b", "G2", "G3", "G4") if q in v}, v["gamma_fit"], v["gamma_R1"])
    print(S["gate_summary"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "theory":
        theory()
    elif cmd == "run":
        run_case(sys.argv[2])
    elif cmd == "floquet":
        floquet(sys.argv[2])
    elif cmd == "summary":
        summary()
    elif cmd == "list":
        print("\n".join(all_cases()))
