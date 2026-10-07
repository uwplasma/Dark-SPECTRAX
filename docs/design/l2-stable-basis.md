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

1. Frozen-field and Floquet spectra (studies `highk_eig.py`, `highk_floquet.py` on branch highk) for B and C at the
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
