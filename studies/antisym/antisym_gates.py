"""antisym: decisive frozen-field / Floquet / OTSI prototype of the anti-symmetric candidate (Issan et al.,
arXiv:2312.05439, 2407.09626) against the asymmetrically weighted (AW) Hermite truncation.  Gates DECLARED BEFORE RUNNING.

Literature reading (done before the gates; see docs/design/l2-stable-basis.md, "Verdict on E"):
  * 2312.05439: symmetrically weighted (SW, weight 1) orthonormal Hermite functions, truncation closure, centred x
    differences or a Fourier-Galerkin projection => the semi-discrete advection operator is anti-symmetric (Galerkin
    projection of the skew Liouville operator, trial = test space) and sum|C|^2 = int f^2 is conserved for ANY E(t), u(t).
    New in that paper: the square-root form g = sqrt(f) (same linear equation for g, Prop. 1) whose norm sum_n C_n^2 is the
    PARTICLE NUMBER, so density is positive and exactly conserved for every Nv, and |j|, kinetic energy are bounded by the
    number.  Momentum and energy are NOT conserved (drift rates depend on the last coefficients, Thm 8-9), and
    2407.09626 shows no closure restores number, momentum and energy together in SW.  There is no AW statement: the paper
    calls AW conservative but unstable and gives no stability theorem (the L2(f) norm is not bounded for AW).
  * The frozen-field linear operator of the sqrt formulation is the SW operator on g (identical matrix to the SW option
    already rejected on H05), so gate 1 cannot discriminate SQ from SW; what is new is the nonlinear moment map.

Variants: AW_c0 (control, no closure), AW_c1 (#66 closure c = 1, control), SQ (sqrt-SW; frozen operator = SW width a).
Points (highk/dai): frozen E1 in {1e-3, 4e-3}, |k| <= K k1 with K in {2, 5, 10}, Nn in {32, 64, 128}, u = 0, a = sqrt(2e-3).
Floquet (oscillating field E1 cos(k1 x) cos(t), Magnus 100 substeps): E1 in {8e-3, 2e-2}, (K, Nn) in
{(2,32),(2,64),(2,128),(5,32),(5,64),(10,32)}.  OTSI: every k5/k12/k20 point of studies/dai/otsi.json with positive Silin
rate, Nn in {32, 64, 128}, Nni = 16, linearised sqrt system (antisym_ops.sq_otsi_floquet).
Gates:
  G1  SQ: dense max Re lambda <= 1e-10 at all 18 frozen points and Floquet exponent <= 1e-10 at all Floquet points.
      Reference behaviour recorded for AW_c0/AW_c1 (expected positive, 0.07-1.7).
  G2  SQ: OTSI Floquet rate within 1% of the Silin relation at every gate point where AW is itself within 1%.
  G3  (report) finite-N conservation of the sqrt moments: instantaneous defects of N, P, KE (relative) on random states
      with Hermite spectra n^-p, p in {2, 4, 8}, Nn 32/64/128, K = 5, frozen E = E1 cos + random modes.  No pass/fail;
      statement: number exact to round-off, others reported.
  G4  (report, soft flag) pseudo-spectral RHS wall time of SQ / AW at Nx = 32 (K = 10), Nn = 64, 128; flag if ratio > 2.
  G5  SQ: (a) normality defect ||L L^H - L^H L|| / ||L||^2 <= 1e-12, (b) |dense max Re lambda - time-domain growth rate|
      <= 1e-8 (time-domain: expm_multiply, 200 time units, renormalised every 10), (c) the AW_c0 values of the same
      diagnostics and the eigenvector condition number at (K 5, Nn 64, E1 4e-3) are recorded for contrast.
Rejection of SQ as a basis candidate: G1 or G2 fails.  Pass of G1, G2, G5 means only "linearly stable and Silin-accurate";
the nonlinear question (SW blew up at t 640-740 on H05) is NOT settled by any of these gates and is addressed by the
separately declared stage 2 (studies/antisym/antisym_h05.py), declared after stage 1 results are committed.
Run: python studies/antisym/antisym_run.py PART  (frozen, floquet, otsi, cons, cost, cond) -> studies/antisym/PART.json
"""

Amendments (made after the first batch, before the re-runs; no gate threshold changed):
  A1 cons: the first run crashed (array-shape bug in the x-mode truncation, no result produced); fixed and re-run.
  A2 cost (G4): the first run used dense N x N matmuls for both variants and a dense einsum for the SQ moment, giving a
     SQ/AW RHS ratio 3.2 (Nn 64) and 5.8 (Nn 128); that record is kept as cost_dense_firstrun.json.  Both variants are
     re-run with banded (tridiagonal) kernels, the implementation a production code would use.
