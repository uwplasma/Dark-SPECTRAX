# L2-stable velocity basis: design note (no implementation)

Status: proposal. Context: the asymmetric-Hermite (AW) Fourier-Hermite operator `-v d/dx + (q/m) E d/dv`, truncated
at `g_N = 0`, is not anti-self-adjoint in a non-uniform field. Its spurious growing modes sit in the top third of the
Hermite ladder at the largest retained |k|, with rates that grow with Nn and with the Fourier cut-off (frozen-field max
Re lambda 0.26-0.84 at E1 = 4e-3; Floquet 0.146 at K = 5, Nn 64). Today's fix is a declared regulariser: the
field-scaled order-2 hypercollision of SPECTRAX PR #66. This note compares three options.

## Options

| | A. current: AW + field-scaled closure (#66) | B. symmetric (SW) Hermite functions | C. Dai's stable AW Galerkin (arXiv 2608.09827) |
|---|---|---|---|
| trial / test | AW functions / Hermite polynomials (Petrov-Galerkin) | SW functions / same (Galerkin, weight 1) | AW functions / AW functions (Galerkin, unweighted L2) |
| L2 of f | not conserved; bounded only by the closure | conserved by the truncated system (frozen-field spectrum Re lambda <= 4e-15 measured) | proven stable in unweighted L2 (paper: fixed basis, no shift, Vlasov-Poisson) |
| mass, momentum, energy | exact (closure leaves rows 0-2 untouched) | not simultaneously conserved (paper intro, citing the SW literature) | expected not exact: the extra last-column term feeds rows 0-2 from g_N (inferred from paper eq. 27, not checked) |
| Maxwellian in 1 mode | yes (a = sqrt2 v_t) | yes, if the SW scale equals the Maxwellian sigma (psi_0 ~ exp(-v^2/2 sigma^2)) | yes (same functions as AW) |
| operator sparsity | tridiagonal in n | tridiagonal | tridiagonal plus one dense last column (paper sec. 7) |
| cost vs A | 1x (+ one max|E| reduction) | ~1x | ~1x + O(N) per k per step for the column |
| moving / rescaled basis (pump frame, remaps) | implemented and tested | shift/scale generators change form (no longer lower-triangular in the weight); must be re-derived | not covered by the paper; stability with u(t), a(t) unproven |
| free parameter | c (H05 insensitive over [0.5, 2]) | none | none |

## What to test (decisive experiment)

1. Frozen-field and Floquet spectra (studies `highk_eig.py`, `highk_floquet.py`, on main) for B and C at the
   same (K, Nn, E1, omega): accept only if max Re lambda <= 1e-10 for every case where A without closure is unstable.
2. Linear OTSI benchmark (studies/instab_otsi.py): the per-k Floquet rate of the quivering two-species plasma must
   match the exact kinetic Silin matrix dispersion relation to 1% at k5, k12, k20 (A already does, Nn 32-128).
3. H05, v_q/v_te = 0.1, Nx 16, Nn 64, nu = 0, no closure: run to t = 700. Compare dK_e and W_ext with the A + c = 1
   run and the mobile-ion grid under the 10% running-max rule.
4. Conservation ledger over the same run: momentum and energy drift relative to W_ext.

Rejection criterion: reject an option if (1) fails, or (3) reaches t = 700 but departs from A + c = 1 and the grid
before t = 600 (where they agree), or if (4) shows an energy-ledger drift > 1e-3 W_ext before t = 600 (the level at
which the closure's conservation advantage matters for the conversion ledger). Prefer C over B if both pass, because C
keeps the one-mode Maxwellian and the AW shift algebra; adopt either only after the moving-basis generators are
re-derived and checked to 1e-10 against finite differences, as done for AW.

## Recommendation

Keep A as the production path (exact conservation, tested). Prototype C first in a standalone linear 1D1V script (no
SPECTRAX changes) to run tests 1-2; proceed to 3-4 only if they pass.

## Verdict on C (2026-10-15, `studies/dai/`, docs/results.md)

Test 1 fails for C as the paper writes it (eq. 27, AW coordinates, float64): max Re lambda 1e-2 to 1.1 at Nn 64-128
(frozen) and Floquet exponents up to 1.0; the L2 norm grows by 1e88 by t = 1000 at Nn 64. The cause is round-off, not
the method: Dai's trial space equals span of Hermite functions of width a/sqrt2, so the method is option B in disguise,
and the AW-to-orthonormal change of basis has condition ~2^N (column entries ~3e10 at Nn 128). Written in that
orthonormal basis it passes tests 1-2 (max Re lambda <= 6e-15; OTSI within 0.96% of Silin). C also conserves only
the moments of one parity of N+1 (never density, momentum and energy together) and costs 1.4x nnz, 2.5x matvec.
C is rejected; A stays in production and B is the remaining candidate for tests 3-4.

## Verdict on B (2026-10-15, `studies/sw/`, docs/results.md)

Tests 3-4 fail as declared. H05 v_q 0.1, nu = 0, RK4 dt 0.02: every SW run (Nx 16/32, Nn 64-256) goes non-finite
between t = 637.5 and 737.5, and Nn 256 dies earlier than Nn 128 at Nx 32. The Nx16 Nn pair agrees only to t = 634
(rejection threshold 650) and the Nx16 Nn64 energy ledger reaches 1.2e-3 W_ext before t = 600 (Nx32: <= 1.9e-5).
Until blow-up SW tracks the grid through the OTSI growth. Electron density and momentum are not exact (2.3e-3 and
1.1e-3 at Nx16 Nn64, down to ~5e-8 at Nx32 Nn256). B is rejected; A (#66 closure) stays in production.

## Verdict on E: anti-symmetric square-root SW (Issan et al.), 2026-10-19, `studies/antisym/`, docs/results.md

Literature (read in full: arXiv:2312.05439, 2407.09626, 2412.07073; follow-up search found nothing newer): what is provably
bounded is sum|C|^2 = int f^2 dv dx (SW f-form) or sum C^2 = particle number (square-root form g = sqrt f), in the symmetrically
weighted orthonormal Hermite basis with weight 1, closure by truncation (any modified last row or source breaks it), x by
centred differences or Fourier-Galerkin, for ANY E(t) and ANY moving centre u(t) (the shift generator is skew; a width change
is skew in the symmetrised form (XD+DX)/2, not stated in the paper), and the argument extends to magnetised terms ((v x B).d_v
is divergence-free), but the papers treat only Vlasov-Poisson. Number is exact for every Nv in the square-root form; momentum
and energy are not (drift rates depend on the last coefficients, which vanish only as Nv -> infinity), and 2407.09626 shows no
closure restores all three in SW. There is no AW statement: the AW L2(f) norm is unbounded and the only AW quadratic invariant
is the linear free-energy norm int df^2/f_M, valid for perturbations of a fixed Maxwellian (this is what the OTSI Floquet
test exercises), not for the frozen non-uniform field or the nonlinear problem.

Result. Stage 1 (declared gates G1-G5): the square-root form is exactly normal and neutral (max Re lambda <= 4e-15 at 18/18
frozen and 12/12 Floquet points where AW_c0 is unstable at 18/18 and 11/12 and AW + #66 closure at 18/18 and 9/12), reproduces the
Silin OTSI rate to 0.25% and equals the AW linearisation to 2.5e-10, has eigenvector condition 40-105 against 1e19-1e29 for AW,
and costs 1.5x the AW RHS and nnz. Finite-N conservation: number to 1e-18; momentum and kinetic energy defects scale with the
top-two Hermite fraction (3e-3 at fraction 6e-4; 1e-12 to 1e-14 for steep spectra). Stage 1 cannot separate SQ from SW (identical
frozen operator), so stage 2 (the H05 v_q 0.1 pilot that would have been planned for SPECTRAX) was run standalone with gates
identical to the SW study, declared before the runs: SQ departs from the grid at t = 391 (Nx 16) and 498 (Nx 32), earlier than SW
(634-738), independent of Nn (64-256) and dt (0.02 vs 0.005), fails G3 and G4, and blows up within about 10 time units after the
top Hermite band fills (a diagnostic Nx 64 run fails at 538). E is rejected; A (#66 closure) stays in production. No SPECTRAX
implementation plan is written because the pilot it would serve has been run and failed. Open question for a different study
(not run): why the square-root x-discretisation fails at an Nn-independent, Nx-dependent time while the f-form tracks the grid
(the two drivers agree to 1e-9 until t = 250), for example the 2/3 mask on g versus on f = g^2.
