# SPECTRAX upstream review (2026-10-03)

The open SPECTRAX pull requests and what Dark-SPECTRAX needs from them. Nothing was merged or closed in this
review; every recommendation is for the owner to act on.

- Base: main `ab87385`. Dark-SPECTRAX was pinned to that main commit at review time; it is now pinned to `integration/dark-baseline` (see below).
- Identity: all new commits are by Rogerio Jorge. Original authors are kept.
- Rewritten branches: updated with `--force-with-lease`, with local backup branches kept.
- Tests are local runs. Where GitHub CI is cited, it is Python 3.11/3.12 plus codecov on the new head.

## Open pull requests

| PR | Target | Old → new head | Depends on | Scope after cleanup | Tests and evidence | Recommendation |
|---|---|---|---|---|---|---|
| #48 dependencies | main | 1cf130d → 1e795e9 | none | `pyproject.toml` +2/−2: declares diffrax and orthax; `requires-python >=3.11` | Main fails `import spectrax` in a clean venv. The branch imports and runs on 3.11 and 3.13 with identical energy. CI green. | merge |
| #46 logo, bytecode | main | cfef220 → dac2ff6 | none | 6 binary files | Logo checked by eye side by side (mean pixel difference 3.1/255). Docs build passes. CI green. | merge |
| #13 structured `(Ck, Fk)` state | main | 1dc9820 → 53777eb | none | 2 files, +77/−15 | RHS and trajectories bitwise identical to main (1D and 3D/3V, 3 species). Layout and continuation tests added. CI green. | personally review before merge: the public `ode_system` signature changes from a flat array to a tuple |
| #12 multi-population bookkeeping | main | 94960e3 → 3f89a09 | none | 2 files, +30/−2 | Tests for 1–4 populations, keeping the mean current. The 3- and 4-population cases fail on main. CI green. | merge |
| #9 rFFT Parseval weights, tail diagnostic | main | c773d98 → 6eeebb8 | none | 3 files, +177/−5 | Energy checked against a full FFT for odd and even grids. Main undercounts EM energy on odd grids (ratio 0.898 at Nx = 7). CI green. | merge |
| #55 strict 2/3 mask (new) | main | – → 4f8939f | conflicts textually with #13 | 5 files, +117/−18 (~20 production lines) | Boundary alias of amplitude 0.25 reproduced at N = 6, 12, 33, 129. 78 new tests, 39 of which fail on main. CI green. | merge (owner may glance at the initial-projection policy) |
| #50 step budget, failure flags | main | 2f19bd6 → 2744422 | none; one-line conflict with #9 | 3 files, +72/−12 | Default step floor removed. Stiff transient now succeeds; failure metadata added. CI green. | personally review before merge (stopping policy) |
| #18 implicit midpoint | main | 60ca682 → 806c2e4 | none | 3 files, +205/−29 (isolated from 25 commits) | NaN-success bug fixed and real failure results added; 10 failure-path tests, 9 of which fail on the old head. Landau run is second order with energy at roundoff. CI green. | personally review before merge |
| #51 Maxwellian width, bump-on-tail | main | ff6a0a0 → 30e67c4 | none | 3 files, +204/−26 | Weighted-norm bound σ < α tested at σ/α = 0.5, 1/√2, 0.85, 0.95; coefficients match independent quadrature to 1e-11. CI green. | personally review before merge (width mathematics) |
| #44 adjoint, progress, step budget | main, stacked on #50 | 49c4c69 → 3ad4a9d | #50; one-line conflict with #9 | 4 files, +216/−6 (unrelated diffs dropped) | Finite-difference plateau test. Example: reverse-mode vs finite difference 2.9e-8. CI green. | personally review before merge (API and gradient contract), after #50 |
| #21 Newton-Krylov solver + preconditioner | main | 9c066cc (unchanged) | needs a restack on new #18 | review only | On one stiff collisional case the gain comes from the preconditioner, not the Newton-Krylov solver. Dopri8 is 4–10× cheaper on the shipped low-collision examples. No reverse-mode autodiff. | defer |
| #54 sync DG with main | Discontinuous-Galerkin | 049d008 → 2f4ee44 | none | merge of main; README note marks Fourier-only examples | 9 tests pass. DG spatial discretization is kept. | personally review before merge (integration strategy) |
| #45 DG autodiff | `sync/dg-with-main` | 27dc7e7 → c715971 | #54 | 4 files | Constant Hermite eigenbasis matches `eigh` to 2.2e-15 and is finite at α = 0. Derivatives tested at zero speed. CI green. | merge into DG after #54 |
| #39 DG PyTree state, sharding | `agent/dg-autodiff` | bca8405 → 99298f1 | #45 | 3 files, about +150/−60 | Serial and sharded runs agree to 5e-13. The sharded gradient fails (device placement inside a traced function). No real-device cost data. | defer |
| #34 reconnection example | main | c7b068f → 87c4cf7 | #9; #12 for exact 4-population tails | 1 example file | Fit window and observable now match. Kinetic current is plotted. Spatial-order claim removed. CI green; the example itself is not run in CI. | personally review before merge |
| #33 tearing-sheet fix | Reconnection_testing | 655309b (unchanged) | none | 1 file | Normalization derived independently and the residuals reproduced; the content is superseded by #34. | close as superseded (owner decision) |
| #41 complex-FFT rollback (external contributor) | main (stale base) | 2a13468 (not pushed) | none | 14 files | Wrong on even grids: relative difference 0.41 caused by a k-grid shifted by one bin. Also regresses #49 and the per-axis mode masking. | close as superseded (do not merge) |

## Suggested merge order

1. **main:** #48 → #46 → #13 → #12 → #9 → #55, then #50 → #44, #18 and #51.
   - #55 needs the same conflict resolution against #13 that is used on the integration branch.
2. **Discontinuous-Galerkin:** #54 → #45 → (#39 on hold).
3. **Close after owner approval:** #33 and #41.

An unmerged integration commit containing #48, #46, #13, #12, #9 and #55 exists (`6781d80`; 115 local tests pass). It is superseded as the companion pin by `integration/dark-baseline` (`9d0982d`, below).

## What Dark-SPECTRAX needs upstream

Dark-SPECTRAX is pinned to the SPECTRAX integration branch `integration/dark-baseline`
(`9d0982d`: main `ab87385` plus the unmerged PRs #48, #46, #13, #12, #9, #55, #50, #44, #18, #51, merged in that order),
not a release. Until those PRs merge upstream:

- **#55 (strict mask):** used through the parent mask; grids divisible by 3 are valid.
- **#9 (Parseval weights):** `inner` uses the parent's `_rfft_weights`.
- **#13 (structured state):** the zero-mixing regression calls the parent's tuple `(Ck, Fk)` `ode_system`.
- **#50 (step limits):** `run` follows the same `max_steps`/`dtmin`/`num_valid_times` semantics; the parent's
  `_stepsize_controller` is not reused because it ties `rtol = atol`.
- **#18 (implicit midpoint):** available at the pin; not yet used in Dark-SPECTRAX.
- **#48 (dependencies):** the companion still declares the dependencies itself.
