"""R3 (inhomogeneous / magnetized), first step: how do a smooth density variation n(x) and a magnetic field B0 change
dark-photon -> plasma conversion and escape?

DECLARED BEFORE ANY r3 PRODUCTION RUN (cold reference or kinetic). Gates are never amended after results; a wrong gate is
recorded as a failure and a follow-up is declared in a new commit before it is run. Files: this module (set-up, case lists,
gates, channel definitions), r3_ref.py (independent NumPy cold-fluid + Maxwell-Proca time-domain reference and the analytic
Landau-Zener / WKB formulas), r3_kin.py (Dark-SPECTRAX runs), r3_analysis.py (applies the rules below, writes the summary).
Smoke tests of the code paths on small or different set-ups are not production runs and are not reported as results.

UNITS (parent normalization): c = 1, Omega_cs[0] = 1, electrons q = -1, m = 1, omega_pe^2(x) = n(x), fixed ion background
(ions never enter the dynamics, only Gauss's law). Omega_D = 1. Dark coupling eta in {0.15, 0.3}.

SET-UP (all runs). Periodic box Lb = 640. Density n(x) = n_lo + (n_hi - n_lo) * [tanh((x - xu)/w) - tanh((x - xd)/w)]/2
with n_lo = 0.02, n_hi = 2.98, xu = 220, xd = 480 (plateau 260: the transmitted packet does not reach the down ramp before
T). Incident packet: right-moving free Proca mode of the vacuum, carrier k0 = 1 (omega0 = sqrt 2, group velocity 0.707),
Gaussian in k with sigma_k = 0.04 on the box wavenumbers k_j = 2 pi j/Lb, centred in x at 120 (amplitude normalized to a
peak |E_D| of 1e-4 per unit of the polarization vector), polarization p in {x (longitudinal), y, z}. T = 330.
The same discrete spectrum is used by the cold reference and by Dark-SPECTRAX. Warm kinetic runs: v_te = 0.01 (T_e = 1e-4 m c^2),
Hermite orders (Nn, Nm, Np) = (4, 4, 4) unless stated, alpha = sqrt(2) v_te, immobile ions with density n_e(x) + dE0/dx,
electron profile = Maxwellian n(x) in a static field E0(x) = -v_te^2 n'/n (an exact kinetic equilibrium: f depends only on
energy; also with B0 because the isotropic f has no Lorentz force), dark near field of the residual charge dE0/dx set from
(k^2 + Omega_D^2) phi_D = eta rho_net. B0 magnitude |Omega_c| = 0.3, direction (cos th, 0, sin th), th in {0, 45, 90} deg.
Density-gradient scale: L_c = n_c / n'(x_c) at the crossing density of the case (analytic definition below), reported per case.

CHANNELS (energies in the perturbation, quadratic form u = 1/2 [E^2 + B^2 + E_D^2 + B_D^2 + Omega_D^2 (A_D^2 + phi_D^2) + n |v|^2],
v = electron flow = flux/n; "total" kinetic energy of the kinetic code also has a thermal part). Energy fractions of
E_in = total perturbation energy at t = 0. At t = T:
 refl_dark, refl_photon : left window x in (0, xu - 6w), energy in left-moving (direction sign(k omega) = -1) eigen-branches of the
   uniform cold plasma at the local mean density, branches labelled 'dark-like' when the dark share of the eigenvector energy
   exceeds 1/2, else 'ordinary-like' (the cold eigen-decomposition with the energy inner product, groups made W-orthogonal);
 inc_left : right-moving energy in the left window (should be ~0);
 trans_dark, trans_photon : window x in (xu + 6w, xd - 6w), right-moving branches likewise;
 stored : energy (all forms, incl. coherent flow) in the zone |x - xu| < 6w;
 beyond : energy at x > xd - 6w (should be ~0);
 absorbed (kinetic only) : thermal kinetic-energy increase K_th(T) - K_th(0), K_th = K - 1/2 n |v|^2 integrated (includes the
   small Langmuir compression energy; cold reference: identically 0, the cold resonance energy sits in 'stored').
 closure : |sum of all channels - E_in|/E_in.

FLUXES: S = E x B and S_D = E_D x B_D + Omega_D^2 phi_D A_D (x components, epsilon_0 = c = 1) at planes xa = xu - 7w and
xb = xu + 7w, time-integrated with the trapezoid rule on the saved times. Budget: [E_zone(T) - E_zone(0)] =
int (F(xa) - F(xb)) dt over the zone between the planes (cold: exact; warm: neglects the electron convective kinetic flux
and the pressure work, both O(v_te^2)).

GATES
 Reference (r3_ref.py; N_ref = 4096 points, scipy DOP853 with rtol 1e-12, atol 1e-11 * amplitude (second pre-run correction: 1e-20 forced the step size through round-off in empty regions), dense output every 0.05;
  PRE-RUN CORRECTION committed before any production run: the first draft said RK4 dt = 0.05, whose energy loss per step
  (omega dt)^6/72 would give ~1e-5 over T = 330 and could never meet R1):
  R1 energy: max_t |E_tot(t) - E_in|/E_in <= 1e-8.
  R2 refinement (N 6144, rtol 1e-13 on the cases eta 0.3, w 4, B0 = 0, polarizations x and y, and B0 45 deg polarization y):
     every channel fraction changes by <= 1e-4.
  R3 flux budget (planes, saved every 0.05): |residual|/E_in <= 1e-4 with the phi_D A_D term, and without that term the
     residual must exceed 1e-2 E_in for B0 = 0 polarization x (longitudinal), demonstrating the term is needed.
  R4 Landau-Zener, transverse, B0 = 0 and B0 || x with polarization y (R and L circular parts, each with n -> n f_s,
     f_R = omega/(omega - Omega_c), f_L = omega/(omega + Omega_c), energy-weighted half and half):
     local two-level law  P_d = exp(-pi g^2 / (kbar |dDelta/dx|)),  g = eta n_c f, Delta(x) = M2_22 - M2_11 = Omega_D^2 + (eta^2 - 1) f n,
     n_c = Omega_D^2/((1 - eta^2) f), kbar^2 = omega^2 - f n_c, |dDelta/dx| = (1 - eta^2) f n'(x_c), with transmitted dark
     = P_d, reflected photon = P_d (1 - P_d), reflected dark = (1 - P_d)^2 (needs f n_hi > omega^2, true here). Spectral average:
     weights |A_k|^2 omega_k^2, P_d evaluated per k with omega_k = sqrt(k^2 + Omega_D^2). VALIDITY PREDICATE V: kbar L_c >= 3
     and eta f n_c <= 0.25 kbar^2 (R and L both, evaluated at the carrier). Gate (only cases in V): |trans_dark - P_d|,
     |refl_photon - P_d(1 - P_d)|, |refl_dark - (1 - P_d)^2| <= 0.03 each. Outside V: reported, not gated, together with the
     deviation, so the domain of validity is read from the table.
  R5 longitudinal (B0 = 0, polarization x): the local dispersion k^2 = omega^2 - Omega_D^2 (omega^2 - n)/(omega^2 - (1 + eta^2) n)
     (derived in r3_ref.py) has a turning point and a cold resonance (pole) at n_p = omega^2/(1 + eta^2); WKB tunnelling to the
     pole T_WKB = exp(-2 int kappa dx) (spectral average as in R4). Prediction: stored+absorbed fraction (cold: 'stored') = T_WKB
     (everything that tunnels is absorbed at the pole, nothing returns), the rest is reflected dark. Gate (cases with
     kbar L_c >= 3 at the pole): |stored - T_WKB| <= 0.10 and |refl_dark - (1 - T_WKB)| <= 0.10.
  R6 B0 effect (reported, ungated): all channel fractions of every B0 case minus the B0 = 0 case of the same eta, polarization
     and w.
 Kinetic (r3_kin.py; unless stated v_te = 0.01, Nx 512, Nn = Nm = Np = 4, rtol 1e-10, atol 1e-14, chunked in 3 segments of 110):
  K1 equilibrium: a background-only run to T = 330 (eta 0.3, w 4, B0 0): max |n_e(T) - n_e(0)|/n_hi <= 1e-6 and max |E|, |E_D|
     over the box <= 1e-3 of the packet amplitude (1e-4 -> 1e-7) plus the initial equilibrium values (reported: E0 max, dark
     near-field max).
  K2 solver status success for every run; ledger |K - K0 - W|, |U_gamma - U_gamma0 + W_em|, |U_D - U_D0 + W_D| <= 1e-6 E_in.
  K3 closure <= 2e-3 and flux budget residual <= 5e-3 E_in (saves every 0.5; cold reference sampled the same way is reported
     to separate time-sampling error), omission-of-phi_D A_D residual reported.
  K4 kinetic vs cold reference, same case: every channel within 0.02 absolute (transverse B0 = 0), 0.03 (any B0), longitudinal:
     stored + absorbed within 0.05 and refl_dark within 0.05. The cold reference of the same case is computed at the same
     discrete spectrum.
  K5 refinement pairs (w 4, eta 0.3: B0 = 0 y and x; th 45 deg y): members (Nx 512, Nn 4) and (Nx 768, Nn 6); every channel
     changes by <= 0.005, else the case is flagged not converged and excluded from K4.
  K6 amplitude linearity (w 4, eta 0.3, B0 = 0, y): amplitude x 3: channels change by <= 0.005.
 Scope: first step. Not attempted: nonlinear amplitudes (see R1), mobile ions, nonthermal tails, 2-D/3-D geometry,
 non-relativistic k0 << Omega_D (needs a longer box: LZ validity needs eta f n_c <= 0.25 kbar^2), packets of finite
 bandwidth beyond sigma_k/k0 = 4%, frequency-domain boundary-value solution of the ionospheric kind (Beadle, Caputo, Ellis,
 arXiv:2405.13882: read at abstract depth only; their statement is that the linearized conversion calculation is not suitable
 when the dark matter is non-relativistic and the plasma varies strongly, and they solve a second-order boundary-value problem; our
 time-domain cold solution is that same linear second-order problem with outgoing packets, at relativistic k0).
"""
import numpy as np

OMEGA_D, K0, SIGMA_K, LB, XU, XD, XC0, T_END = 1.0, 1.0, 0.04, 640.0, 220.0, 480.0, 120.0, 330.0
N_LO, N_HI, OMEGA_C, VTE, AMP = 0.02, 2.98, 0.3, 0.01, 1e-4
ETAS = (0.15, 0.3)
W_LIST = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
THETAS = (0, 45, 90)
POLS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
GATE = {"R1": 1e-8, "R2": 1e-4, "R3": 1e-4, "R3_omit": 1e-2, "R4": 0.03, "R5": 0.10,
        "K1_n": 1e-6, "K1_field": 1e-3, "K2": 1e-6, "K3_closure": 2e-3, "K3_flux": 5e-3, "K4_T": 0.02, "K4_B": 0.03,
        "K4_L": 0.05, "K5": 0.005, "K6": 0.005}


def density(x, w):
    return N_LO + (N_HI - N_LO) * 0.5 * (np.tanh((x - XU) / w) - np.tanh((x - XD) / w))


def dens_prime(x, w):
    return (N_HI - N_LO) * 0.5 / w * (1 / np.cosh((x - XU) / w) ** 2 - 1 / np.cosh((x - XD) / w) ** 2)


def b0_vector(theta_deg):
    th = np.deg2rad(theta_deg)
    return OMEGA_C * np.array([np.cos(th), 0.0, np.sin(th)])
