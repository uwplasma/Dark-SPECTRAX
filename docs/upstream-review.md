# SPECTRAX upstream review (2026-10-03)

The open SPECTRAX pull requests and what Dark-SPECTRAX needs from them. Nothing was merged or closed in this
review; every recommendation is for the owner to act on.

- Base: main `ab87385`. Dark-SPECTRAX was pinned to that main commit at review time; it is now pinned to `integration/dark-baseline` (see below).
- Identity: all new commits are by Rogerio Jorge. Original authors are kept.
- Rewritten branches: updated with `--force-with-lease`, with local backup branches kept.
- Tests are local runs. Where GitHub CI is cited, it is Python 3.11/3.12 plus codecov on the new head.
- Heads in the table are as of 2026-10-03. The PR branches were updated on 2026-10-07 and their current heads differ (`gh pr list -R uwplasma/SPECTRAX`); the integration branch `f33dfda` is built from these branches.

## Resync of 2026-10-24 (current pin)

SPECTRAX main is now `b668d87`: the owner merged #46, #48, #12, #13 and #50. Every remaining open PR of ours was updated by
merging `origin/main` (and, in stacks, its parent) into its head, with ordinary pushes and no history rewrite, and a new parent
branch `integration/dark-baseline-2` was built from `b668d87`. Dark-SPECTRAX is pinned to it:
`a6a560bafd42e0f2c957d2b080b7ac786c14b81e`. The earlier branch `integration/dark-baseline` (`35231ab`) is untouched.

Merged since the previous review: #46, #48, #12, #13, #50 (all in main `b668d87`).
Note: #48 sets `requires-python >=3.11`, so main no longer supports Python 3.10.

| PR | Head, old → new | Conflicts with main | Local tests on the new head |
|---|---|---|---|
| #9 | e4355ee → 7979f97 | `_simulation.py` (output dict) | 41 passed |
| #55 | eaec43c → 1b48a7e | `_simulation.py` (initial state now a pytree; mask re-applied before the reshape) | 97 passed |
| #44 | e931983 → 97d9605 | none | 25 passed |
| #18 | a9a145b → a2367ed | tests (kept both sides) | 32 passed |
| #51 | fc10b11 → cd59474 | none | 30 passed |
| #21 | 3a64507 → 3d07157 | README, pyproject, `_simulation.py`, tests | 38 passed; still conflicts with #18 (two midpoint implementations) |
| #34 | 5c4a70f → b9d8fe3 | none | 19 passed |
| #62 | 445ba0c → 7c0c692 | tests (kept both sides) | 21 passed |
| #63 | 71faa7e → 7948bdb | none | 22 passed |
| #64 | 5996a67 → c26b154 | none | 22 passed |
| #65 | 8420322 → e8a66e2 | README (kept the TOML-first text, moved `dtmin`, `max_steps`, `throw` into its example) | 22 passed |
| adaptive/base | 287c9e9 → f259f03 | none | 149 passed |
| #56 | 2a5c4c7 → 63efe6b | none | 154 passed |
| #57 | 75477e8 → 140b3c0 | none | 158 passed |
| #58 | f722430 → dbeb008 | none | 170 passed |
| #59 | 38b08a8 → 842b72c | none | 179 passed |
| #60 | c63a8a2 → f94b118 | none | 184 passed |
| #61 | dd6fc97 → 5e900c2 | none | 191 passed |
| #66 | 08944c9 → 5f7cb98 | none | 187 passed |
| #71 | afe685a → a7f583d | none | 198 passed |
| #73 | 3de6b2b → a724a08 | none | 223 passed, 1 xfailed |
| #72 | 7d868a2 → 951aa44 | none | 188 passed |
| #68 | b9f729f → c5d2824 | none | 26 passed, 1 xfailed |
| #69 | 551e6d7 → d709ce8 | none | 21 passed |
| #70 | 5a53018 → 4edb3ed | none | 42 passed |
| #82 | 26fd219 → d78e87d | `_simulation.py` (Lawson rate now follows the `(Ck, Fk)` state) | 29 passed |
| #83 | 9630d72 → 6c62840 | none against #82 (the streaming factor now uses the pytree layout) | 36 passed |

`integration/dark-baseline-2` (`a6a560b`) is main `b668d87` plus #9, #55, #18, #44, #51, #34, #62-#65, adaptive/base and #56-#61,
#66, #71, #72, #68, #69, #70, the Francisquez commit `3de6b2b` of #73 (cherry-picked: the head of #73 carries the old integration
history) and #82, #83. #21 is not included (it replaces #18's midpoint solver). The full suite gives 240 passed, 1 xfailed.
Dark-SPECTRAX: 85 passed; `examples/landau.toml` and the H05 pump case (v_q/v_te = 0.03, Nn = 32, nu = 0, t <= 100) reproduce the
recorded results bitwise (`dK_e`, `dK_i`, `W_ext`, and the Landau `run.npz` arrays differ by exactly 0.0 from a run at `35231ab`).

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

Dark-SPECTRAX is pinned to the SPECTRAX integration branch `integration/dark-baseline-2` (`a6a560b`, see the resync above). The text below describes the earlier branch `integration/dark-baseline`
(`f33dfda`: main `bf41830` plus the unmerged PRs #48, #46, #13, #12, #9, #55, #50, #44, #18, #51, merged in that order,
then the moving-Hermite-basis stack #56-#60, the lane-A filter #61, the field-scaled closure #66 and the species-TOML usability stack #62-#65), not a release.
Earlier pins: `9d0982d` (first ten PRs), `c0910a1` (+ #56-#60), `16400f8` (+ #61, #66), `26605e5` (+ #62-#65, on main `ab87385`); the side branches `integration/highk` and
`integration/lane-a` are superseded by `integration/dark-baseline`. Until those PRs merge upstream:

- **#55 (strict mask):** used through the parent mask; grids divisible by 3 are valid.
- **#9 (Parseval weights):** `inner` uses the parent's `_rfft_weights`.
- **#13 (structured state):** the zero-mixing regression calls the parent's tuple `(Ck, Fk)` `ode_system`.
- **#50 (step limits):** `run` follows the same `max_steps`/`dtmin`/`num_valid_times` semantics; the parent's
  `_stepsize_controller` is not reused because it ties `rtol = atol`.
- **#18 (implicit midpoint):** available at the pin; not yet used in Dark-SPECTRAX.
- **#66 (field-scaled closure):** `studies/highk_run.py` calls `spectrax.field_scaled_closure_rate`; equal to the
  former study implementation to 3.4e-16 (`studies/highk_parent_equiv.py`).
- **#48 (dependencies):** the companion still declares the dependencies itself.

## Upstream PRs of 2026-10-16

Opened from the gap inventory of 2026-10-16. Each PR is stacked, so it gets no CI; the tests were run locally. All three are merged into
`integration/dark-baseline` `d6cedcb`, where the full suite gives 201 passed and 1 xfailed.

- **#71** (on #66): conservative Dougherty collision operator, exact in the moving Hermite basis (lower-triangular
  projection; density, momentum, energy conserved to round-off for N >= 3; pair collisions with Greene beta = 0), opt-in via
  `collision_model = "dougherty"`. Merged into `integration/dark-baseline` as `a4bdd72` (212 passed, 1 xfailed). Used by B11.
- **#72** (on #71): `field_scaled_closure_rate(..., u_s=None, magnetic=False)`; `magnetic=True` adds the non-uniform
  v x B force scale (the B09 EB rate). Merged into `integration/dark-baseline` as `dd80b0f`. Used by B09 nonlinear.
- **#68** (on #44): a strict-xfail regression test for the Diffrax 0.7.2 checkpoint-adjoint defect ([diffrax#788](https://github.com/patrick-kidger/diffrax/issues/788); 201 saves on the
  fixed-step grid, 2.5% off ForwardMode), a passing test of `checkpoints = max_steps`, and a docstring warning. The default is
  not changed: the step count is traced inside `simulation`, and the default `max_steps = 1e6` rules out storing a checkpoint
  for every step.
- **#69** (on #50): `ode_rtol` and `ode_atol` (default `ode_tolerance`, bitwise unchanged), and `FlooredPIDController` (a
  per-component absolute error floor). Dark-SPECTRAX `noise_floor` now uses it, and the former `_FlooredPID` is removed.
- **#70** (on #9): the public names `rfft_weights` and `twothirds_mask`. The adapter in `darkspectrax/_model.py` now imports only
  public parent names. Study scripts keep their recorded imports.

Kept in the companion: the differentiable `solve`/`with_params`, compile-once segments with a traced `t0`, the positivity stop
and the per-species row policy of `noise_floor`. The field-scaled closure is installed by wrapping `rhs` in the studies; that
needs a companion RHS hook, not a parent change.
