"""R2: sustained dark-to-plasma absorption when collisions act, compared with the linear collisional (drag) absorption law.

DECLARED BEFORE ANY R2 PRODUCTION RUN (an unrecorded short smoke test of the runner precedes the analysis commit; no gate or rule
below depends on it).  Gates are never amended after results; a wrong gate is recorded as a failure and a follow-up is declared in a
new commit before it is run.  Runner: studies/r2_run.py (+ studies/r2_batch.py); linear theory: studies/r2_theory.py; analysis
(committed before the production runs, implements exactly the rules below): studies/r2_analysis.py.

Question. For a prescribed drive Ed = E0 cos(omega t) acting on mobile-ion (m_i/m_e = 1836) H05-setup plasma with the Dougherty e-e/e-i/i-i
operator, what is the late-time absorption rate, and does it equal the linear drag law P = wL^2 nu_eff omega^2 E0^2 / (2[(wL^2 - omega^2)^2 +
nu_eff^2 omega^2]) with nu_eff the momentum-relaxation rate implied by the operator?  Literature form being compared with: the
cosmological estimates (Dubovsky and Hernandez-Chifflet; McDermott and Witte) use the same cold-plasma friction -m_e nu (u_e - u_i) with
the Spitzer-like nu ~ n_e logLambda / T_e^{3/2}; in the model here nu is constant (velocity independent), so a heating-driven fall of nu
that those estimates contain is NOT in the model.  The uniform relative velocity relaxes at nu_eff = (nu_ei/2)(1 + m_e/m_i) (Morse rate;
identical in both conventions), i.e. the literature nu corresponds to nu_eff, not nu_ei.

Setup (all runs): H05 (studies/lane_c_run.py / h08.py): v_te^2 = 1e-3, L = 40, T_i = T_e, mobile ions 1836, seeds r0 (deterministic gate
seeds), pump frame + remaps every 20, saves every 0.5, noise floor 1e-14, rtol 1e-10, atol 1e-14, T = 1000, restart chunk 200,
strong closure field_closure(model, c = 64) integrated by the Lawson wrapper (as r1nl).  Collisions: nu_dougherty =
[[nu_ei, nu_ei], [0, nu_ei sqrt(m_e/m_i)]]; convention EM = collision_model "dougherty" (LBO-EM), G = "dougherty_g" (LBO-G, beta = 0).
v_q/v_te is the effective force: E0 = (v_q/v_te) v_te.  omega: "res" = wL = sqrt(1 + 1/1836); detuned omega = 0.9 and 1.1 (omega_pe = 1).
Grid: v_q/v_te in {0.01, 0.03, 0.05}, nu_ei in {1e-3, 3e-3, 1e-2, 3e-2}, omega in {res, 0.9, 1.1}, convention in {EM, G}.
Members: B = (Nx 32, Nn 64), X = (Nx 64, Nn 64), N = (Nx 32, Nn 128), all c = 64.  B for all 72 cases; X and N for the 36 EM cases only
(the G window is the EM window of the same case, which is checked by CV below).  Controls: unseeded Nx 8, Nn 64 at v_q 0.03 and 0.05, all
nu_ei, omega and both conventions (48 runs).

Observables per save: K_s, T_x,s (k = 0), W_ext (ledger column 2), ledger defect, Ebar, collisional power P_s per species
(Dougherty part of the RHS), mean-field identity residual R (as h08.py).  Cycle average = mean over |t - tq| <= pi/omega.

GATES.
 T0 (theory self-check, no kinetics): the uniform ODE (r2_theory.exact_linear), nu_ei = 0.6, omega in {wL, 0.9, 1.1}, v_q = 0.03, t to 400:
    the mean of dW/dt over the last 80 drive periods equals r2_theory.p_ss within 1e-3 (relative).
 C0 controls: Ebar and W_ext of every control run equal the exact ODE to 1e-6 of the maximum, t_reached = 1000.
 L1 ledger: max |ledger defect| / max |W_ext| <= 1e-6 for every run.
 L2 collisions do no net work: max |P_e + P_i| <= 1e-9 max |P_e| for every collisional run.
 L3 mean-field identity: max |R| <= 1e-8 wL^2 E0 for every run.
 W  window of a case (EM): Wn = min(1000, t_pair(B,X), t_pair(B,N) each on dK_e and on W_ext by r1_common.disagreement (10% of the running
    max, t > 50), first save with K_s <= 0 or T_x <= 0 of any member, t_reached of any member).  Certified iff L1 holds for all members and
    Wn > 50.  Wp = min(Wn, first t with T_e(t)/T_e(0) >= 90) is the "nonrelativistic" window (v_te >= 0.3 c beyond it); claims about the
    model use Wn, statements about physical applicability use Wp.  The G window is the EM Wn of the same case.
 K2 kinetic vs linear (B member, every case with Wn >= 300 in both conventions): tb = Wn (largest save <= Wn).  Delta = 250 for res; for
    detuned Delta = n_b 2pi/|omega - wL| with n_b = floor(250 |omega - wL| / 2pi) (so whole beat periods).  P_kin = [Wc(tb) - Wc(tb - Delta)]/Delta with
    Wc the W_ext cycle average; R_late = P_kin / P_lin on the same interval, P_lin from r2_theory.exact_linear; R_cum = Wc(tb)/W_lin(tb).
    PASS iff |R_late - 1| <= 0.05 and |R_cum - 1| <= 0.05, AND for EM the refinement members X and N give R_late within 0.02 of B's.
    Reported per case; failures kept.  A case with Wn < 300 gets no K2 verdict (window reported).
 K3 steady state: f_ss = P_lin / p_ss(E0, nu_ei, omega) on the same interval.  For cases with 0.9 <= f_ss <= 1.1: PASS iff |P_kin / p_ss - 1| <= 0.05.
    Otherwise only f_ss is reported (the transient has not decayed; the sustained law is then not tested by that run).
 CV conventions (EM vs G, B member, cycle averages at tb): |rel change| in W_ext <= 1e-3 and in dK_e, T_e <= 1e-2.  T_i, dK_i, Q_i are reported,
    not gated (they are fed by the e-i exchange, which is halved by construction; lesson of H08 LBO-G G4).
 K4 deposition-law fit (reported with uncertainty, not pass/fail): over certified EM cases separately for res, 0.9 and 1.1,
    ln P_kin = a + p ln(v_q) + q ln(nu_ei) by least squares, the same fit to P_lin (same intervals); report (p, q) kinetic, (p, q) linear and
    their differences with LSQ standard errors (residual scatter), plus kappa = geometric mean of R_late with its standard error over cases.
    Expectation: p = 2 exactly; q differs by the transient state, not by physics.
 Reported, not gated: T_e/T_e(0), T_i/T_i(0), dK_e/W_ext, dK_i/W_ext, Q_i/W_ext (cumulative e-to-i collisional exchange) at tb; the ratio R_late
    versus T_e/T_e(0) along the run (does heating change the rate: nu is constant in the model); OTSI-type fluctuation field energy fraction at tb.
Scope: no claim beyond Wn.  Velocity-dependent Coulomb nu (nu ~ T_e^-3/2), radiative/Compton cooling, cosmological expansion, relativistic
electrons, ion-electron nu beyond the m_e/m_i scaling and the Z = 1 choices nu_ee = nu_ei, nu_ii = nu_ei sqrt(m_e/m_i) are not in the model.
"""
import itertools

VQ = (0.01, 0.03, 0.05)
NU = (1e-3, 3e-3, 1e-2, 3e-2)
OMEGA = ("res", "0.9", "1.1")
CONV = ("EM", "G")
MEMBERS = {"B": (32, 64), "X": (64, 64), "N": (32, 128)}
CTRL_VQ = (0.03, 0.05)
T_END, CHUNK, C_CLOSURE = 1000.0, 200.0, 64.0
T0_NU, T0_TOL, C0_TOL, L1_TOL, L2_TOL, L3_TOL = 0.6, 1e-3, 1e-6, 1e-6, 1e-9, 1e-8
DELTA, TOL_R, TOL_REF, TOL_SS = 250.0, 0.05, 0.02, 0.05
CV_W, CV_K = 1e-3, 1e-2
W_MIN, TE_REL = 300.0, 90.0

# DECLARATION DEFECT in T0 (found by the first analysis pass on partial data, before the production runs finished; recorded, T0 is kept as
# declared and is reported as FAILED because its averaging window (80 periods = 502 units) is longer than its record (400)).  Declared
# follow-up, theory-only (no kinetic run involved):
#  T0b same check as T0 with the ODE integrated to t = 1000 and the mean slope taken over the last 40 drive periods; tolerance unchanged.
T0B_T, T0B_PERIODS = 1000.0, 40


def cases():
    """All production runs: (conv, vq, nu, omega, member, control)."""
    out = []
    for c, v, n, w in itertools.product(CONV, VQ, NU, OMEGA):
        out.append((c, v, n, w, "B", False))
        if c == "EM":
            out += [(c, v, n, w, m, False) for m in ("X", "N")]
    for c, v, n, w in itertools.product(CONV, CTRL_VQ, NU, OMEGA):
        out.append((c, v, n, w, "C", True))
    return out
