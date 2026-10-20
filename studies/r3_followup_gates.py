"""R3 follow-up F (declared AFTER the first-step records and BEFORE its runs; r3_gates.py is unchanged).

Why: in the first-step records the kinetic runs of cases with energy deposited at a cold resonance layer (longitudinal
polarization, oblique B0) are not converged between (Nx 512, Nn 4) and (Nx 768, Nn 6) (K5 fails: change 0.08 and 0.15 for
eta 0.3 w 4 and 2 longitudinal, 0.0085 / 0.0057 for th 45 x and th 90 y) and then differ from the cold reference (K4 fails). The cold
phase-mixing wavenumber of a layer oscillating at omega_pe(x) grows as k(t) = |d omega_pe/dx| t (about 40 at T = 330 for
w = 4), far above the Nx 512 mask (k = 1.67), so the first-step grids cannot represent the layer. The follow-up asks
whether the propagating channels converge with Nx and then agree with the cold reference, and how large the layer energy
('remnant') is for the converged kinetic runs.

Cases: (eta 0.3, w 4, B0 = 0, x) and (eta 0.3, w 4, th 45, y). Members: Nx in {1024, 1536, 2048, 3072}, Nn = Nm = Np = 4, v_te = 0.01,
same set-up/solver as r3_kin.py; cold twins with ntrunc = Nx (r3_ref.py case ... ntrunc). Extra thermal members at the largest
Nx of the converged pair: v_te in {0.005, 0.02}.
 remnant := 1 - (refl_dark + wrap_refl_dark) - (refl_photon + wrap_refl_photon) - trans_dark - trans_photon - absorbed
            (energy that is neither in a propagating outgoing branch nor deposited as thermal energy).
 FU1 convergence: the maximum change of {refl_dark, refl_photon, trans_dark, remnant} between consecutive members
     (2048 -> 3072) <= 0.01; else the case is 'not converged at the largest grid'.
 FU2 agreement with the cold twin at the largest member: |kin - cold| <= 0.02 for refl_dark, refl_photon, trans_dark and
     <= 0.03 for remnant (computed from the cold channels in the same way with absorbed = 0).
 FU3 (reported, ungated): the same channel differences between v_te 0.005, 0.01, 0.02 at Nx 3072.
Nothing in FU1-FU3 changes a first-step gate.
"""
NX = (1024, 1536, 2048, 3072)
VTES = (0.005, 0.02)
GATE = {"FU1": 0.01, "FU2": 0.02, "FU2_remnant": 0.03}

# ---------------------------------------------------------------------------------------------------------------------
# Follow-up G (declared after the first-step R4 result and before it is computed): is the failure of R4 at eta 0.15 (w 8, 12:
# deviations 0.035 / 0.045 > 0.03; transmitted dark larger than exp(-pi g^2/(kbar |dDelta/dx|))) a first-order-in-eta
# correction of the two-level law that a complex-turning-point (Dykhne) exponent captures?
# Refined law: with the local eigenvalues lambda_pm(n) of M^2 = [[f n, eta f n], [eta f n, Omega_D^2 + eta^2 f n]]
# (f = 1, or f_R, f_L of R4), k_pm = sqrt(omega^2 - lambda_pm), the diabatic (stay-dark) probability is
#   P_d = exp(-Theta),  Theta = 2 |Im int_{n_r}^{n*} (k_+ - k_-) dn / n'(n)|,   n* = Omega_D^2/(f (1 +- i eta)^2) (the complex point where
# lambda_+ = lambda_-), n_r = Re n*, straight path in the complex n plane, n'(n) = (D/(2w)) (1 - tau^2), tau = 2(n - n_lo)/D - 1, D = n_hi - n_lo,
# the branch of sqrt(disc) chosen continuous (positive on the real axis). Channels as in R4 (T = P_d, photon P_d (1 - P_d),
# dark (1 - P_d)^2), spectral average as in R4.
#  FU4 gate: for every R4 reference case with kbar Lc >= 3 (B0 = 0 and th 0, eta 0.15 and 0.3) the refined T, refl_photon, refl_dark are
#      within 0.02 of the reference channels (refl_photon with the wrap correction, declared left-window value also reported).
#  FU4b: new cold reference runs (same code, N = 4096): eta in {0.05, 0.1, 0.2}, w in {4, 8, 12}, B0 = 0, polarization y; the same comparison
#      (simple R4 law and refined law), and the exponent ratio ln T_ref / ln P_LZ(simple) reported against eta f n_c/kbar^2.
# The simple law (R4) is not replaced: its failure stays recorded.
