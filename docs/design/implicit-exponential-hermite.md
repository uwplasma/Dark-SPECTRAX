# Implicit and exponential treatment of the top Hermite modes: design note (no implementation)

Status: proposal with a literature review and one ungated toy check. Context: the AW Fourier-Hermite system
(`docs/design/l2-stable-basis.md`, results sections highk, rclosure, gpures) has a growing truncation mode in the top
third of the ladder; a field-scaled hypercollision only delays it, and adaptive Dopri8 runs end at the step budget
once those modes grow. This note asks what a time integrator can and cannot do about that.

## 1. Which part of the operator is which

For one species, 1D1V, frozen field E(x), AW coefficients C_n(k), the semi-discrete system is
`dC/dt = (L_s + L_E + L_nu) C` with

| block | form | spectrum / structure | role |
|---|---|---|---|
| streaming `L_s` | `-i k a T`, T real symmetric tridiagonal, `T[n,n+1] = sqrt((n+1)/2)`; a pump-frame shift adds the scalar `-i k u` | purely imaginary, `+- i k a x_j` with x_j the Gauss-Hermite nodes (largest about sqrt(2 Nn)); skew-Hermitian, so its exponential is unitary | stiff in the CFL sense (rate grows with K and sqrt(Nn)); not unstable |
| force `L_E` | `(q/m) (sqrt(2 n)/a) E(x) *` acting `C_{n-1} -> C_n`, a convolution in k | strictly lower triangular in n, hence nilpotent for frozen E: no eigenvalues of its own; exponential is a finite polynomial in E (exact velocity translation at each x) | norm about `|q/m| |E| sqrt(2 Nn)/a`; comparable to streaming at low k, smaller at high k |
| closure `L_nu` | diagonal, `-nu s(n)` | real negative | stiff only for strong closures; exactly exponentiable |
| field update | Maxwell/Proca, nonlinear through the current (moment n = 1 of C) | couples all k | not part of any exponential block |

The instability belongs to the sum `L_s + L_E`, not to either piece: T is symmetric and `L_E` is nilpotent, so
growth needs both (upward streaming couplings and downward force couplings in a non-normal product). In the toy
check below (K = 5, Nn = 64, E1 = 4e-3) the frozen-field spectrum has max Re lambda = 0.448, in line with the
rclosure fit.

## 2. What an integrator can and cannot remove

Two different failures must be kept apart.

1. Stiffness (step-size limit from the imaginary spectral radius of `L_s`, and from strong closures). A Lawson
   (integrating-factor) or Strang scheme that exponentiates `L_s` exactly removes this limit: Lawson schemes have
   the stability function of the underlying Runge-Kutta method applied to the remainder, independent of the
   exponentiated block (Crouseilles-Einkemmer-Massot, arXiv:1910.12720, sec. 3.1). Classical exponential
   integrators (ETD, ExpRK) do not share this property for imaginary-axis problems: their stability domain
   changes with the phase and need not contain a symmetric imaginary interval (same paper, sec. 3.2).
2. Instability (Re lambda > 0 of the semi-discrete operator). A consistent integrator reproduces it as h -> 0.
   Exact exponentiation of `L_s` and `L_E` cannot change it, and the toy check shows Lawson-RK4 and a Strang step
   with exact sub-flows return 0.448 at every h tested (0.42 and 0.38 at h = 3). Only a scheme whose amplification
   factor is below one on the unstable modes lowers the growth, and then the damping is set by h, not by the
   physics: backward Euler gave 0.43, 0.34, 0.085, 0.045, 0.015 at h = 0.01, 0.05, 0.2, 1, 3; implicit midpoint
   (symmetric, no damping) 0.448, 0.446, 0.43, 0.22, 0.075. An L-stable implicit step on the top modes is therefore a
   hidden, step-dependent closure that stops working as h -> 0; it would have to be declared as a regularizer and
   shown to leave rows 0-2 and the resolved physics untouched. Implicit midpoint (Delzanno 2015, Pagliantini et al.,
   Issan et al.) conserves quadratic invariants but does not cure the AW growth; it only steps over fast frequencies.

Consequence: exponential or implicit treatment can lift the step-size collapse only to the extent that the
collapse is stiffness. If the adaptive controller is limited by accuracy on growing top-band content (an
unresolved e-fold per few steps from a 1e-14 seed), a larger stability region buys little. The first experiment
below separates the two.

## 3. Literature map (details in the agent report; all equations paraphrased)

- Lawson / integrating factor for hyperbolic kinetic problems: arXiv:1910.12720 (Vlasov-Poisson, drift kinetic);
  stability equals the underlying RK method, CFL from the nonlinear part only.
- Exponential Rosenbrock, ETDRK4, Krylov and Leja phi-functions (Hochbruck-Ostermann reviews, Cox-Matthews,
  Kassam-Trefethen, Niesen-Wright, Al-Mohy-Higham): cited from the literature, not fetched here.
- Strang splitting for Vlasov: second order under regularity (Einkemmer-Ostermann, arXiv:1207.2090).
  Hamiltonian splitting for Vlasov-Maxwell (arXiv:1401.4477) has a published comment (arXiv:1504.07785) that the bracket
  used is wrong; a corrected construction is arXiv:1505.06076.
- Hermite codes: Camporeale et al. (arXiv:1311.2098) treat the x-shift implicitly (BDF1 or Crank-Nicolson) inside a
  Strang split with RK4 on the v-shift, and report the all-explicit RK4 version unstable at dt 0.05-0.1 where the
  semi-implicit one is stable to 0.5; Delzanno 2015 (fully implicit CN, Jacobian-free Newton-Krylov) shows forward
  Euler unstable at all dt and RK stable only to about 0.5 in its test; Pagliantini et al. (arXiv:2110.11511):
  all RK schemes keep linear invariants, Gauss-Legendre keeps quadratic ones, explicit schemes do not;
  Koshkarov et al. (arXiv:2004.12010) explicit adaptive RK plus a hypercollision; Parker-Dellar (arXiv:1407.1932)
  third-order Adams-Bashforth with iterated Lenard-Bernstein or Hou-Li filtering; Issan et al. (arXiv:2412.07073)
  the Hou-Li filter equals an exponential (integrating-factor) factor on a diagonal damping, with strength set per
  step, and artificial collisions are an equivalent regularization.
- No 2024-2026 paper found that applies Lawson/ETD/IMEX to the AW truncation instability itself (searches
  returned only the sources above).

## 4. Structure and conservation

- Charge (k = 0 density row) is a linear invariant with `l L_s = 0` and `l L_E = 0` (n = 0 row has no force
  term); every Lawson, ETD, IMEX or split scheme preserves it to round-off.
- Momentum and energy are quadratic and tied to the field update. Explicit RK, Lawson-RK, ETD and Strang keep them
  only to the scheme's order; only Gauss-Legendre collocation keeps them exactly (Pagliantini et al.). In this
  code the ledger states are integrated with the same stages, so Lawson with those states left out of the exponent
  (A = 0 block) keeps the existing "ledger closes to solver tolerance" property; the closure must keep acting only on
  n >= 3.
- Exact streaming exponentials are unitary in the Hermite l2 norm; exact force flows are not (nilpotent, polynomial),
  so no scheme here is l2-conservative.

## 5. Cost in JAX

- T has no free parameter: one real `eigh` of an Nn x Nn tridiagonal matrix per axis gives V (orthogonal), shared by
  every k, species and width a (a and k enter as scalars). `exp(h L_s) = V diag(exp(-i h k a x_j)) V^T`. This is
  the Gauss-Hermite nodal transform, so the exponential is a velocity-node phase multiplication.
- Cost per application: two real Nn x Nn matmuls on the (Nk x Nn) real and imaginary parts, about `8 Nn^2 Nk` flops,
  against about `10-20 Nn Nx log Nx` for the existing convolution RHS: roughly 3-4x one RHS at Nn = 128, 7x at
  Nn = 256 (estimate, to be measured on the GPU). A Dopri8 interaction-picture wrapper needs two applications per
  stage (26 per step); a genuine Lawson-RK form needs about one per stage. Step-size gain must exceed this.
- Phase accuracy: the per-segment phase `k a x_j t` reaches about 2e3 at t = 1000 in the toy; build `exp(h L_s)`
  per step from the increment, not from the segment start.
- `jnp.linalg.eig` is CPU-only in JAX, so closure plus streaming must not be exponentiated jointly through a
  complex-symmetric eigendecomposition; use a Strang factor `exp(-h nu s(n)/2)` (diagonal) around the streaming factor.
- Krylov phi-functions on the full Jacobian cost one RHS-like matvec per Krylov vector and are only needed for an
  exponential Rosenbrock variant.

## 6. Candidate schemes, in order

- S0 (diagnostic, no new integrator). Step-limiter attribution on a recorded run.
- S1. Lawson or interaction-picture wrapper: exponentiate `L_s` (and `u` phase), diagonal closure as a split
  factor, explicit adaptive RK for `L_E` plus fields, all in the existing run driver. Targets stiffness only.
- S2. Strang split with exact streaming and exact nilpotent force sub-flow (pseudo-spectral in x, E held fixed
  across the substep, field update between): fixed step, second order, cheap; 1D1V electrostatic prototype only.
- S3. Exponential Rosenbrock-Euler or EXPRB32 with Krylov: only if S0 shows the force block, not streaming, is the
  limiter.
- S4. Any L-stable implicit step on the top band: control experiment only, never a production path (hidden
  closure, section 2).

## 7. Decisive experiments (gates to be committed before each run)

- D0 (S0). Replay the vq 0.1, Nx 64, Nn 128, c = 4 chunk around t = 500-595 with the committed driver and log, for
  accepted and rejected steps: h, `h * rho(L_s)` against the Dopri8 imaginary-axis limit, and the Hermite-band
  composition of the error norm. Reject S1/S2 as a step-budget cure if the median accepted `h * rho(L_s)` is below
  0.3 of the limit or if the top third carries under half of the error norm in rejected steps.
- D1 (frozen field). For the highk_eig/highk_floquet cases, Lawson and Strang one-step maps reproduce the exact
  growth rate to 2% for h up to 1; accept as expected behavior, reject any scheme whose growth is lowered by more
  than 10% at h <= 0.2 (that would be hidden damping, not exactness).
- D2 (real run, S1). Same record as D0, same rtol and atol: success needs steps (or wall time including the transform
  overhead) at least 3x lower to the same t, dK_e and W_ext within 1e-6 of the Dopri8 record to t = 500, ledger at or
  below 1e-9, first non-positive K_s or T_x not earlier than the baseline. Failing to pass t = 600 at vq 0.1 is the
  expected outcome and does not by itself reject S1; the instability stays and needs a basis or closure change.
- D3 (S2 prototype). Linear OTSI Floquet rate within 1% of the Silin relation at k5, k12, k20 (as for the AW baseline)
  and charge to round-off.
- Rejection overall: wall-time ratio to baseline above 1 at equal accuracy; any ledger drift above 1e-3 W_ext before
  t = 600; any claim that the schemes remove the truncation instability without D1 evidence at h -> 0.

## 8. Ungated toy check (`studies/imex/frozen_toy.py`, record `studies/imex/frozen_toy.txt`)

Frozen single-cosine field, K = 5, Nn = 64, E1 = 4e-3, a = sqrt(2e-3), q/m = -1. Streaming spectral radius 2.35, force
norm 0.97, force block nilpotent to round-off (checked). Per-step growth rate of the one-step map against the exact
0.448:

| h | RK4 | Lawson-RK4 | midpoint | backward Euler | Strang, exact sub-flows |
|---|---|---|---|---|---|
| 0.01 | 0.448 | 0.448 | 0.448 | 0.428 | 0.448 |
| 0.2 | 0.447 | 0.448 | 0.43 | 0.085 | 0.447 |
| 1 | 0.388 | 0.447 | 0.217 | 0.045 | 0.441 |
| 3 | 1.4 | 0.421 | 0.075 | 0.015 | 0.384 |

RK4 loses stability where `h * rho` passes about 2.8 (h about 1.2 here); Lawson and Strang do not. Larger K and Nn raise
rho by K sqrt(Nn), so the RK4 limit scales down accordingly. This is a linear frozen-field check: it says nothing yet about the
nonlinear run or the step budget.

## 9. Result of S1 for the diagonal closure only (2026-10-19, `studies/expint_gates.py`, results.md "Exact integration of the diagonal closure damping")

Built: SPECTRAX `LawsonSolver` (PR #81) exponentiates only the diagonal closure/hypercollision (rate frozen at the step start; `f = -r y + N` is an identity for any `r`; order of the
wrapped tableau retained for h -> 0 at fixed rate (Lawson 1967), no stiff-order claim when r h >> 1 since the stiff order of Lawson methods drops, to 1 for Lawson-Runge (Hochbruck and Ostermann 2010, Acta Numerica 19, Thm 2.17, Ex. 2.32, Table 2.2); empirical support: agreement with plain Dopri8 records, 5e-10 in H05 c=64); streaming is NOT exponentiated, so D1 (frozen-field one-step maps)
and the streaming part of S1 remain open. D2 as declared (both c = 4 and c = 64 at least 3x fewer steps) FAILED: c = 64 gives 5.0x (t <= 500)
and 4.3x (to 740) at equal per-step cost, c = 4 gives 0.96x (the closure is not the stiff mode before t ~ 540). The accuracy, ledger and
positivity parts of D2 passed (dK_e, W_ext within 5e-10 of the record, ledger 8e-13 to 2e-11, no earlier positivity loss), the window did not
change (760; 579.5 vs 578), as expected. A Lawson interpolant is only accurate for r h of order one: dense output of a stiff damped mode is
garbage, so steps are clipped onto the saves (found by a failing test, not a design choice).

## 10. Result of S1 for the streaming operator (2026-10-20, `studies/streaming_gates.py`, results.md "Exact Hermite streaming in the integrating factor")

Built: SPECTRAX PR #83 (stacked on #82): the streaming operator `-i k (a T + u)` per Hermite axis is exponentiated exactly with one real eigh of the
Hermite tridiagonal (Gauss-Hermite nodal transform shared by all k, species, widths, drifts), and composed with the diagonal closure by the Strang product
`Phi = A(h/2) B(h) A(h/2)` inside the integrating factor. The transformed system `w' = Phi^-1 (f(Phi w) - Phi' w)` is an exact identity (no splitting error in the
scheme), but the part left explicit is `O((h |G|)^2 |r|)` (symmetric BCH), so a stiff closure is removed completely only while `h |G|` is small. Measured: per-step cost
2.7x the closure-only Lawson (two Nn x Nn transforms per stage); H05 c = 64 identical step counts (9652 vs 9606: streaming was not the limit), 2.5x slower wall; c = 4
steps identical to t = 400, then step collapse (201927 steps in a chunk, run ends at 456.5 by the budget); B10 reaches the 400000-step budget (closure-only 25845). Exact
streaming alone is correct (analytic Hermite front at 13x the Dopri8 stability step, order, gradient) and cuts steps 5-36x on smooth streaming-limited problems
(SPECTRAX default two-stream, Nx 64-128 Nn 128), which is not this regime. A joint exponential per k (non-Hermitian eig of `G - R`, CPU-only, rate-dependent every step) would
remove the leftover and was not built. D1 (frozen-field one-step maps) remains open.

