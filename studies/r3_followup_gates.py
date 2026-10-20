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
