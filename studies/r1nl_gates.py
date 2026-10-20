"""r1nl: does nonlinear detuning / saturation of dark-photon energy transfer appear at the strongest certifiable drives,
with finite reservoir and finite coherence?  (Science question R1, second step.)

DECLARED BEFORE ANY r1nl PRODUCTION RUN. Gates are never amended after results; a wrong gate is recorded as a failure and a
follow-up is declared in a new commit before it is run. Runs: studies/r1nl_run.py; analysis: studies/r1nl_analysis.py
(committed after the stage-1 runs but implements exactly the rules below; any analysis bug fix is recorded, no rule changes).

Setup (all runs): studies/r1_common.py H05 set-up (m_i/m_e = 1836, T_i = T_e, v_te^2 = 1e-3, L = 40, mobile ions, seeds r0,
pump frame + remaps every 20, omega = omega_L, saves every 0.5), noise floor 1e-14, rtol 1e-10, atol 1e-14, T = 1000, up to
200000 steps per 20-unit segment, run stops at the first segment with K_s <= 0 or k=0 T_x <= 0.
Closure: darkspectrax.field_closure(model, c) (SPECTRAX #66 rate, strength c), integrated exactly by the Lawson wrapper
(run(..., integrating_factor=rate_fn)); c = 64 unless stated. v_q/v_te is the effective initial force (E0 = (v_q/v_te) v_te).
Observables: dK_e = K_e - K_e(0), W_ext = ledger column 2 (prescribed drive) or W_D = column 1 (self-consistent reservoir).

STAGE 1 -- certified windows at v_q/v_te in {0.03, 0.05, 0.07}.
 Members per v_q: B = (Nx 32, Nn 64, c 64); X = (Nx 64, Nn 64, c 64); N = (Nx 32, Nn 128, c 64); XN = (Nx 64, Nn 128, c 64);
 C = (Nx 32, Nn 64, c 32).  Pairs: (B,X), (B,N), (X,XN), (N,XN) [refinement], (B,C) [closure strength].
 S1  pair agreement: t_pair = first t > 50 where |a - b| > 0.1 * running max of max(|a|,|b|) (r1_common.disagreement) on dK_e AND
     on W_ext; if no such t, the pair's shorter t_reached. Common save times.
 S2  positivity: first save with K_s <= 0 or k=0 T_x <= 0 of any member caps the window.
 S3  ledger: max |(K - K(0)) - sum_j W_j| / max |W_ext| <= 1e-6 for every member (else v_q is not certified at all).
 S4  solver: every member's t_reached caps the window.
 Window(v_q) = min(1000, all t_pair, all first-nonpositive times, all t_reached). Certified iff S3 holds and Window > 50.
 Grid reference (reported, NOT part of the window; the grid goes negative before the Hermite runs, see docs/results.md):
 studies/nxconv_grid.py Nx 32/64 at the same v_q, comparison of B with the grid by the S1 rule on dK_e and W_ext.
 Strongest certifiable drive v* = the largest v_q with certified and Window >= 300; if none, the v_q with the largest Window.

STAGE 2 -- inside Window(v*) (all quantities evaluated for t in (50, Window(v*)]; "members" = B, X, N, XN of the coherent run).
 N1  W_ext vs the linear resonant law: Wlin(t) = r1_common.prescribed_linear(t, E0) (the exact homogeneous mean-field law).
     D_m(t) = W_ext,m / Wlin - 1. DEPARTURE claimed iff some t has |D_B| > 0.10 and |D_m| > 0.05 for all of X, N, XN at that t.
     Otherwise LINEAR within uncertainty, with upper bound  B_NL = max_t max_m |D_m(t)|  (includes the refinement spread).
 N2  effective frequency shift of the mean field: lock-in phase phi(t) = arg of int_{t-P}^{t} Ebar(s) exp(-i omega_L s) ds,
     P = 5 * 2 pi/omega_L; Dphi = phi_kin - phi_lin (lin = homogeneous mean-field ODE Ebar); delta_omega = LSQ slope of Dphi
     (unwrapped) over the last 20% of the window (>= 4 pump periods). RESOLVED iff |delta_omega|/omega_L > 1e-3 and
     |delta_omega| exceeds 3 * (max over refinement members of |delta_omega_m - delta_omega_B|). Otherwise bound
     |delta_omega| + that spread.
 N3  (reported, ungated) energy partition at the window end, in units of n T_e V: dK_e, dK_i, W_ext, fluctuating field energy,
     k=0 thermal energy; ion share dK_i/W_ext; cumulative integral of Q_i = <dn_i dE_x>; OTSI onset t_otsi = first t > 100 where
     sum_{k=4}^{floor(Nx/3)-1} |E_k|^2 >= 0.1 * sum_{k>=1} |E_k|^2 (all with k <= floor(Nx/3) - 1 counted); growth rate of that sum.
 N4  finite reservoir at v*: self-consistent runs with u = U_D0/(n T_e V) in {1, 10} (eta = (v_q/v_te)/sqrt(2u), Omega_D = omega_L)
     in B and XN. Window_sc = min(Window(v*), t_reached, first-nonpositive of the two, t_pair of (B, XN) on dK_e and W_D).
     (i) |U_D - U_D^lin|/U_D0 <= 0.1 for t <= Window_sc (U_D^lin = r1_common.coupled_solution) -> reservoir stays linear, else
     the first exceedance time t_dev is the onset of nonlinear reservoir response; (ii) W_D/W_D^lin - 1 with W_D^lin = U_D0 - U_D^lin,
     evaluated while W_D^lin > 0.2 * its running max (departure rule of N1 with thresholds 0.10 / 0.05 on B / XN);
     (iii) t_split of W_D from the matched coherent W_ext (10% rule, t > 50) compared with linear (factor 1.5, r1_reservoir G7).
 N5  finite coherence at v*: v2 definition (field coherence time tau; r1_common.noise_knots(tau, seed, fac=2)), tau omega_pe in {100, 300},
     realizations seeds 0..5 in B and seed 0 in XN. Window_coh = min(Window(v*), t_reached and first-nonpositive of the six B
     realizations and the XN one). Per realization W_lin,r = r1_common.linear_work_realization with the realized knots. Ensemble
     R = <W_kin>/<W_lin> over the six realizations at t in [0.8, 1.0] * Window_coh with its standard error over realizations.
     DEPARTURE claimed iff |R - 1| > 0.05 and > 3 se and the XN realization 0 has the same sign with |W_kin/W_lin - 1| > 0.025.
     Otherwise bound |R - 1| + 2 se, and per-realization max |D_r| reported.
 N6  ledger for every stage-2 run <= 1e-6 (as S3); sc runs also the Gauss/ledger columns of the parent (reported).
 Scope: no claim beyond Window; Gaussian line, combined finite reservoir + finite coherence in one run, v_q/v_te > 0.07 and
 t > 1000 are not attempted. Statistical power is stated by the bounds above, not by the 'no departure' wording.
"""
V_LIST = (0.03, 0.05, 0.07)
MEMBERS = {"B": dict(Nx=32, Nn=64, c=64.0), "X": dict(Nx=64, Nn=64, c=64.0), "N": dict(Nx=32, Nn=128, c=64.0),
           "XN": dict(Nx=64, Nn=128, c=64.0), "C": dict(Nx=32, Nn=64, c=32.0)}
PAIRS = (("B", "X"), ("B", "N"), ("X", "XN"), ("N", "XN"), ("B", "C"))
LEDGER_TOL, TOL10, TOL5, TOL25 = 1e-6, 0.10, 0.05, 0.025
W_MIN_STRONG = 300.0
U_LIST, TAU_LIST, NREAL = (1.0, 10.0), (100.0, 300.0), 6
