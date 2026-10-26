"""picconv: convergence study of the relativistic PIC electron heating dW_th for the HHS resonant run
(v_q/v_te = 0.03), declared BEFORE any new run (committed first; never amended after seeing results).
Follows the relpic study (gates.py), whose G4 (ppc 103 -> 412 gave +19 %) and G5 (dx 0.04 -> 0.02 gave +19 %) failed.

Observable: dW_th(t) = trailing 200/omega_pe mean of W_th(t) - W_th(0) (analysis.py definition), in units of
E_th = 5e-4 n m_e c^2 L, at t = 1000 and 2000.  Also recorded: mean-field energy W_E0 and k != 0 field energy W_Ek
(trailing means and the maximum).

Reference (re-digitized from the rendered PRL figure, 600 dpi, vector line; digitization/hhs_fig2.json):
HHS Fig. 2 upper panel, black solid 'electron energy change (k != 0)' = 50.1 E_th at t = 1000 and 52.7 at 2000
(line-width uncertainty about 3 %; the earlier '~25 E_th' reading of relpic was a misreading).  HHS used the Vay
pusher, Nx = 1000, dx = 0.04, Npc 2e2 to 2e5 (SM S34-S38).

Ladders (all ratio 0.03, relativistic Boris unless noted; tags -> arguments of relpic_run.py; LADDER below):
  ppc:   103 (R_s0-2, existing, T 2000), 206, 412 (five seeds F412_s0-4, T 2000), 824, 1648 at dx 0.04, dt 0.02, seed 0, T 1000
  dx:    0.04 (R), 0.02 (X2, existing), 0.01 (X1) at ppc 103 with dt = dx / 2, seed 0, T 1000
  shape: quintic vs quadratic at ppc 412 (Q412 vs F412_s0..4), seed 0; plus existing Q_s0, Q_s1 at ppc 103
  dt:    dt 0.01 (D, existing) vs R at ppc 103
  physics: Newtonian at the finest affordable point ppc 412: N412_s0, N412_s1, T 2000

Gates (thresholds fixed here):
  C1 ppc plateau:  successive refinements 412 -> 824 and 824 -> 1648 each change dW_th(1000) by < 5 % of the finer value,
                   or by less than 2 sqrt(2) sigma_seed (sigma_seed = std of the five F412 seeds), whichever is larger.
  C2 dx plateau:   0.02 -> 0.01 changes dW_th(1000) by < 5 % (same noise allowance with the R-seed std).
  C3 shape:        |Q412 - F412 mean| < 5 % or 2 sigma_seed; C4 dt: |D - R mean| < 5 % or 2 sigma_R.
  C5 extrapolated value: W_inf = F412 mean x [fit(ppc -> inf)/fit(412)] x [fit(dx -> 0)/fit(0.04)] with Richardson-type
                   fits dW = W_inf + c ppc^-p (p = 1 and p = 0.5, ppc points 103-1648) and dW = W_inf + c dx^2
                   (dx points 0.04, 0.02, 0.01); uncertainty = quadrature of seed s.e. at 412, half spread between
                   the p = 1 and p = 0.5 extrapolations, and the dx-extrapolation shift.  Reported with the ppc/dx gates'
                   pass/fail; a failed plateau gate is recorded as unconverged, not repaired.
  C6 HHS comparison: W_inf (relativistic) and the Newtonian N412 mean against 50.1 (t 1000) and 52.7 (t 2000) with a
                   10 % digitization-plus-extrapolation tolerance; outcome recorded either way.
  C7 relativistic reduction at the finest point: (R412 mean)/(N412 mean) at t = 1000 and 2000, in s.e.
"""
SEEDS_FINE = (0, 1, 2, 3, 4)
C1_REL = 0.05
C2_REL = 0.05
C3_REL = 0.05
C4_REL = 0.05
SIGMA = 2.0
HHS = dict(t1000=50.1, t2000=52.7, digitization_rel=0.03, tolerance=0.10)
FIT_P = (1.0, 0.5)
# tag: (relpic_run.py arguments, threads)
LADDER = {
    "F412_s0": ("--T 2000 --ppc 412 --seed 0", 1), "F412_s1": ("--T 2000 --ppc 412 --seed 1", 1),
    "F412_s2": ("--T 2000 --ppc 412 --seed 2", 1), "F412_s3": ("--T 2000 --ppc 412 --seed 3", 1),
    "F412_s4": ("--T 2000 --ppc 412 --seed 4", 1),
    "N412_s0": ("--T 2000 --ppc 412 --seed 0 --nonrel", 1), "N412_s1": ("--T 2000 --ppc 412 --seed 1 --nonrel", 1),
    "P206": ("--T 1000 --ppc 206 --seed 0", 1), "Q412": ("--T 1000 --ppc 412 --seed 0 --shape 5", 1),
    "P824": ("--T 1000 --ppc 824 --seed 0", 2), "P1648": ("--T 1000 --ppc 1648 --seed 0", 3),
    "X1": ("--T 1000 --cells 4000 --dt 0.005 --seed 0", 3),
}
