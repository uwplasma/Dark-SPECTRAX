# Project status (2026-10-15)

A fresh-start review: what is established, what is partial, what has not started, and what to do next.
Results and their records are in [results.md](results.md); the model and the literature-reading status are in
[physics.md](physics.md); the parent PRs are in [upstream-review.md](upstream-review.md).

## Code health

- Parent pin: SPECTRAX `integration/dark-baseline` `3de6b2b` (main `bf41830` plus the open PRs listed in the README).
  The repin from `a4bdd72` (2026-10-16) only adds #72 (magnetic option of `field_scaled_closure_rate`, default off and unchanged); `studies/b09_closure.py` now calls it instead of its own copy (equal to round-off, test in `tests/test_studies.py`).
  The repin from `d6cedcb` (2026-10-16) only adds #71 (Dougherty collisions, default off, RHS unchanged when off).
  Records made before 2026-10-16 name `f33dfda`; the repin to `d6cedcb` only adds #68-#70 and changes no record: a seeded pump-frame run
  with plain PID and with `noise_floor = 1e-14` is bitwise equal before and after.
  This is the current head of that branch. It is not a release, and none of those PRs is merged upstream.
- Tests: 72 pass locally (CPU, float64, Python 3.13, `pytest -n 2 --cov`), coverage 99.8% of 468 statements;
  flake8 is clean. GitHub CI runs the same tests on Python 3.11 and 3.13.
- Spot check: 10 README numbers were compared with their JSON records, and all match to the printed digits:
  the B00 roots and fits at k = 0.2-0.7, the two-stream and bump-on-tail growth rates, the mixing shifts
  (+3.8/+1.8/+5.3% growth, +2.7% frequency, -9.3% damping), the B06 echo amplitudes (ordinary/dark, Hermite/grid,
  agreement 2.0e-5/1.6e-5, mixing -3.3%), the B00 worst relative error 1.3e-4, and the phase-space drawn times
  (100, 29.25, 38.75). The damping shift of -23.6% at k = 0.3 comes from the roots; the fitted rates give -23.7%.
- Record parents. B00, B02/B03 and B01/B06 were rerun at the pin `f33dfda` (2026-10-15): every recorded number
  other than timings is bit-identical to the `9d0982d` records (B00 887 values, B02/B03 182), except two B06
  ledger defects of 6e-11 that changed by 1e-20 (relative 2e-10); the B01/B06 arrays agree to 8e-15 relative.
  Timings rose (compile 0.8-1.9 s, B01 runs 11-28 s) because the laptop was loaded (load about 35).
  Still older than the pin: the dark grid (`9d0982d`), the H05 lane and high-k records (`c0910a1`, 152),
  nxconv (`16400f8`, 122) and the H05/H06 scan (`ab87385`).
- Study code: 12 scripts in `studies/` are not named in any doc (`c05_dark_jax_in_cell`, `h05_pump_frame`,
  `hhs_summary`, `highk_alias_check`, `highk_analysis`, `highk_grid_linear`, `lane_b_analysis`, `lane_c_analysis`,
  `nxconv_analysis`, `nxconv_grid`, `nxconv_grid_validate`, `nxconv_run`). Most are drivers or analysis steps behind
  documented tables. The lane B/C and Nx = 8 scripts are superseded by `nxconv_*`. They are kept because they produced
  records that are still cited. Small helpers (`t_res`, `disagreement`, `install_closure`, `kick`) are copied between
  study scripts. A shared `studies/_common.py` would remove the copies, at the cost of touching record-producing scripts.
- Fixed in this review: the parent-description comment in `pyproject.toml` (main `ab87385` changed to `bf41830`);
  the design note's pointer to a branch `highk` (the scripts are on main); a note that the PR heads in
  upstream-review.md are as of 2026-10-03.

## Test ladder

Status: **done** = passing test or record on main; **partial** = covered in part or only a pilot; **not started**;
**blocked** = cannot proceed without the named item.

### A: algebra and exact solutions (`tests/`)

| ID | Item | Status |
|---|---|---|
| A00 | moments, Lorentz operator vs quadrature | done (1e-12) |
| A01 | zero mixing = parent | done (RHS 1e-14, trajectory) |
| A02 | Proca vacuum | done |
| A03 | cold coupled oscillators | done (1e-11) |
| A04 | finite-k cold branches | done (2e-5 / 1e-4) |
| A05 | static Yukawa | done |
| A06 | eta-sign symmetry | done |
| A07 | exact mean pump | done (1e-9) |
| A08 | convergence (time order, Hermite free streaming) | partial: exact solutions only, no manufactured-source (MMS) hook |
| A09 | one to three populations | done |
| A10 | failure policy | done |

### B: ordinary and dark plasma benchmarks

| ID | Item | Status |
|---|---|---|
| B00 | linear Landau | done ([record](_static/b00/run.json)) |
| B01 | nonlinear Landau | done vs grid ([record](../studies/b01_b06/run.json), [dark grid](../studies/dark_grid/run.json)); rebound correction in results.md; O'Neil 1965 read: rebound is on the bounce scale (2 pi tau = 28 at onset), no extremum-time prediction, so stays grid-vs-Hermite |
| B02 | two-stream | done ([record](_static/b02_b03/run.json)) |
| B03 | bump-on-tail | done (same record) |
| B04 | ion-acoustic | done vs kinetic roots ([record](../studies/b04_b05/b04.json)): T_e/T_i = 1, 5, 20, frequency to 1.4e-3 (7e-3 at T_e/T_i = 1), damping to 1.5e-2 at T_e/T_i <= 5; at T_e/T_i = 20 damping 8% weak, unresolved; Nx = 8 runs at T_e/T_i >= 5 hit the step budget |
| B05 | Buneman | done vs kinetic roots ([record](../studies/b04_b05/b05.json)): m_i/m_e = 100, 1836, ordinary and dark, growth within 1.5e-4, Nn/Nx-converged to 2e-7 |
| B06 | echo | done vs grid (same B01 records); Gould-O'Neil-Malmberg 1967 read: echo time tau k2/(k2-k1) = 30 is what B06 uses |
| B07 | free streaming | done (ordinary, `studies/refs/B07`, test A08); no dark variant (none needed: force off) |
| B08 | magnetized rotation, EM waves | done for uniform B0 ([records](../studies/b08/), results.md): rotation of U, E_0 and the full covariance vs the exact 6x6/expm reference to 2e-11 for four B0 directions and both charge signs, zero magnetic work; 40 cold R/L/whistler/Langmuir/O/X eigenmodes (ordinary and dark, eta = 0.3, Omega_D = 1.2) from the full field-current eigenproblem, fit within 1e-3 at v_t = 0.01, warm error scaling as v_t^2, Nn-converged; Proca flux incl. Omega_D^2 phi_D A_D closes pointwise to 3e-14. Failed gates (n = 1e-4 rotation: fit window, small-K0 ledger ratio, mis-specified K_perp gate) documented. Open: warm/Bernstein reference, nonuniform B0 |
| B09 | Weibel, filamentation | linear done in 1D3V ([records](../studies/b09/), results.md): bi-Maxwellian Weibel and counter-streaming filamentation, ordinary and dark (eta = 0.3, Omega_D = 0.3 near the transverse Proca branch and 1.2), growth vs an independent kinetic transverse dispersion relation with Proca coupling, k = 0.4-2.4; within 1e-3 at every k (Nn = 512 needed at k >= 1.6), ledger <= 1.1e-7. Failed gates: Weibel k = 2.4 Nn 256->512 step > 1e-4 (all models), one spurious fit-mode pick (dark Omega_D = 0.3, k = 0.4, Nx = 9). Nonlinear (Weibel, `studies/b09_closure.py`): **declared criterion 1 and gate G3a FAILED; the peak below relies on post-hoc amendments 1 and 3 (committed before their runs, after seeing the failed results).** the step collapse at t ~ 110 is the AW truncation instability driven by v x B (frozen-field growing mode 12-25x gamma, all in the top Hermite third; the electric-only #66 rate does not remove it, a Lorentz-force-scaled rate does). With it, the first U_B maximum (t = 115-121) gives B_peak = 1.93-1.97 B_sat for ordinary and dark, stable to <= 0.5% in Nx, closure strength and (c = 2) Nn; dark/ordinary ratio within 2.3% of the bounce estimate. Open: nothing past the first maximum (late collapse at every level), filamentation nonlinear not run |
| B10 | Langmuir modulational / Zakharov | done for the ordinary pump, partial for the dark ([records](../studies/b10/), results.md B10): finite-amplitude Langmuir pump (eigenvector of the model, w = 0.04, k0 r_D = 0.005) with one sideband pair per run vs Zakharov R0 (eqs 1.2/1.9, with 2.8/2.9 limits checked: static maximum w/4, not the printed w/16) and a semi-kinetic R1 (exact dispersion, kinetic ions, derived dark coupling g0/eps_g'). Ordinary m_i/m_e = 1836, 100: growth within 1.8-6.9% of R1 (below), R0 = R1 to <1%, pump drift <= 0.25%, ledger <= 1e-9; closure sensitivity nu 100 -> 300 moves it by 5-9% (G2 3% FAILED); Nn = 64 blows up (G2 Nn not met). Dark upper: within 10-12% at the R1 peak (j = 5), over-predicted by 16-40% elsewhere (G1 10/21 overall); m_i/m_e = 1836 dark j = 7, 8 failed at nu = 100 (closure artifact, nu = 300 clean, no gated window). Dark lower unresolved. Outside regime: T_e/T_i = 1 and k0 r_D = 0.04 within 5-13%, w = 0.15 closure blow-up. Floquet cross-check only for m_i/m_e = 100 (ordinary 1.2% from the fit; dark inconclusive, 1836 rejected). Declared v1 (nu = 1) failed, v2 multi-K box aborted (mode competition), v3 single-triad truncation. Open: Nn-converged closure, resolved dark lower, K < k0, Silin-kinetic finite-k pump |
| B11 | collisional relaxation | done for homogeneous uncharged cases ([record](../studies/b11/run.json), results.md B11): conservative Dougherty operator (parent #71, `collision_model = "dougherty"`); two-beam relaxation at exactly n nu per mode, 3V anisotropy exp(-2 nu t) to 4e-11, two-species temperature equilibration (m ratio 4, 100) and drift relaxation vs the exact rates to 1e-13, momentum and energy to round-off. Collisional Landau done ([record](../studies/b11/landau.json), [follow-up](../studies/b11/landau_followup.json)): ordinary and dark (eta = 0.3), k lambda_De 0.3/0.5, nu 0.01/0.05/0.2, fitted omega within 5.7e-4 of the linear Dougherty-Vlasov root (two independent references, Hermite N = 4000/8000 and a continuum Lenard-Bernstein-kernel form, agree to 4e-12), Nn 128/256 to 6e-7, ledger <= 1e-17. **Declared gates G0 and G3 FAILED (mis-specified: O(nu) physics in G0, a t = 0 ratio that is 1 by construction in G3); follow-up G0' passed, G3b' FAILED (rows 0-2 at 3.3e-12 of the rate max vs gate 1e-12, round-off).** Open: charged two-species relaxation; cross-species beta convention not checked against Francisquez et al. |
| B12 | off-resonant forcing with drag | partial: response and absorbed power derived (algebra only); no run |

### C: numerical verification

| ID | Item | Status |
|---|---|---|
| C00 | constraint-satisfying random states | done |
| C01 | long-oscillator phase error | partial: midpoint error formula derived; Dopri8 order tested (A08); no long-run phase-error measurement |
| C02 | refinement | done for H05 (Nx 8-128, Nn 32-256, [nxconv](../studies/nxconv/summary.json)) and the movies |
| C03 | recurrence / closure | done as findings: recurrence-limited fit windows, B07 front, AW truncation instability, #66 closure |
| C04 | remap | done upstream (SPECTRAX #56-#60 tests); used in H05 lanes |
| C05 | Dark-JAX-in-Cell comparison | done for Landau k = 0.5 ([record](../studies/c05/run.json), CPU rerun); PIC noise limits it to about 1% |
| C06 | Gkeyll / grid | partial: own NumPy semi-Lagrangian VP and VAP grids (done); Gkeyll only via imported p = 2 records; Gkeyll docs not read |
| C07 | restart | done |
| C08 | CPU / GPU | done for one short case (2026-10-18, RTX A4000, float64, jax 0.11.2): H05 Hermite v_q 0.1, Nx 32, Nn 64, c = 4, T = 100, GPU vs CPU max relative difference 9.7e-9 (dK_e 9e-9, W_ext 5e-10, T_x 5e-11, |E_k|^2 1e-8; gate 1e-6 passed; adaptive step counts 476 vs 478). GPU integration ~4x faster than 4 CPU threads. Eight 1000-unit runs executed on GPU ([results](results.md), [c08.json](../studies/gpures/c08.json)); one case only, other physics not compared |
| C09 | AD vs FD | done ([record](../studies/c09/run.json), results.md C09): 19 derivatives of cold-oscillator frequencies, W_D(60) and a dark Landau abs(E_k(10))^2 w.r.t. eta, Omega_D, n0, T, seed; reverse = forward to 2e-13, FD plateau 6e-9 or better, analytic 5e-10 or better; Diffrax RecursiveCheckpointAdjoint wrong (2-15%) for many constant-step saves (DirectAdjoint used) |

### H: HHS-v1-inspired ladder (nonrelativistic; not a reproduction)

| ID | Item | Status |
|---|---|---|
| H00 | homogeneous resonant mean field | done (1e-9, t <= 1000) |
| H01, H02 | finite-k weak/strong drive, mobile ions | done as pilots ([record](../studies/hhs/run.json)) |
| H03 | PIC-noise loading | not started (needs Dark-JAX-in-Cell runs with HHS loading) |
| H04 | relativistic drive | **done 2026-10-25 (negative for the 50 E_th claim):** relativistic 1D1V grid `studies/relgrid.py`; gates A, B, C2 pass; D fails (ledger 2.3e-4 of W for the det run; only dt 0.01 passes, 2.9e-5) and D2 fails (ratio 8.0); E windows agree to T=1000. Relativity switches the mean field off at t~370-390 vs 770-870 nonrelativistic (HHS-like timing), but the k!=0 energy reaches ~50 E_th by t=1000 (Nx 64 shot 51, Nx 128 30) only after f<0 (t 240-280), edge contact and a broken ledger (t 350-510); valid window gives <0.01 E_th. See docs/results.md |
| H05 | strong resonant drive | partial: certified windows v_q/v_te = 0.01 to 1000, 0.03 to about 960-1000, 0.1 to about 600 (grid reference to about 750); 0.1 to t = 1000 blocked by the AW truncation instability (not by resolution of the OTSI cascade: more Nx or Nn ends runs sooner, so the limit is the instability outgrowing the closure; the grid reference stays certified only to ~750). **Resolution extension on GPU (2026-10-18, gates in `studies/gpures_gates.py`) FAILED**: v_q 0.1 and 0.05 with Nx 64/128 x Nn 128/256 (c = 4) all end at the step budget inside the instability (0.1: t = 595, 383, 453, 328; 0.05: 752, 555, 671, 440), more Nn or Nx ends sooner; pairs agree to the end of the shorter run but G1 (to 1000) and G4 fail, no new window; going further needs a stable basis (design option B), not resolution; **Closure-strength route (rclosure, 2026-10-19, gates in `studies/rclosure_gates.py`) REJECTED**: in a frozen field no closure strength c up to 1024 brings the AW growth to <= 1e-3 at peak E 4e-3 (39/39 points, 23/39 at 1e-3), a floor ~ K^0.15 Nn^0.30 E^1.26 remains, so no resolution-consistent c(K, Nn) exists and the scaled-closure SPECTRAX option and the Nx x Nn pair test were not built; a stronger fixed c (16/64/256) extends the Nx 64 Nn 128 window from 595 only to 731/757/713 with dK_e, W_ext identical to c = 4 within 1.2e-5 before t = 500 ([stagef](../studies/rclosure/stagef.json), [measure](../studies/rclosure/measure.json)); v_q 0.05 certified only to ~440-555 ([summary](../studies/gpures/summary.json)); **Stiffness vs instability at the failure (stiffdiag, 2026-10-19, gates in `studies/stiffdiag_gates.py`)**: the stiff eigenvalue is the field-scaled closure damping itself (rho(J) = nu_e = c sqrt(2 Nn) max\|E\|/a, real negative; h_stab = 5.17/nu_e), so c = 64 runs at the explicit stability bound (dt/h_stab 0.7-1.9, tolerance-insensitive) and removing the closure damping cuts the steps per unit 6-9 times (c = 4: 7 times at t = 576); that part is curable by an exponential/implicit treatment of the diagonal closure; the window is not: the tangent growth max Re lambda(J) = 2.1-2.7 of the AW instability (from the field-nonlinear part; J_L only 0.16-0.43) ends both runs by loss of positivity/blow-up ([summary](../studies/stiffdiag/summary.json)); **Exact integration of the closure damping (expint, 2026-10-19, gates in `studies/expint_gates.py`, SPECTRAX `LawsonSolver` PR #81, `run(..., integrating_factor=)`)**: c = 64 needs 5.0x fewer steps to t = 500 (4.3x to 740; 5.3x same-CPU wall incl. compile) with dK_e, W_ext equal to the Dopri8 record to 5e-10 and ledger <= 1e-11; c = 4 gains nothing (0.96x to 500: not stiff there) so D2 (both cases >= 3x) FAILS as declared; window unchanged (760 by positivity; c = 4 579.5 by the step budget), the AW instability is not an integrator problem; ordinary B10 4.6x fewer steps, gamma_fit to 6e-10, wall only 1.5x, ledger gate B3 (1e-9) failed at 8.4e-9 (accuracy-limited steps) ([summary](../studies/expint/summary.json)); **Exact Hermite streaming in the integrating factor (stream, 2026-10-20, gates in `studies/streaming_gates.py`, SPECTRAX #83, `run(..., integrating_factor=streaming(rate_fn))`)**: FAILS S4 as declared: c = 64 has the same steps as the closure-only Lawson (9652 vs 9606 to t = 500) at 2.7x the cost per step (wall 130 s vs 52 s), c = 4 collapses at t = 456.5 on the step budget (the Strang leftover O((h\|G\|)^2 \|r\|) is stiff), B10 hits the 400000-step budget (closure-only 25845); accuracy where finished 3e-10; the exact streaming factor is correct and helps only on smooth streaming-limited problems, not here ([summary](../studies/stream/summary.json)) |
| H06 | swept drive | partial: homogeneous check done; kinetic runs lose resolution before the crossing |
| H07 | finite dark reservoir | done (homogeneous, eta 1e-3 / 0.03) |
| H08 | collisions | pilot done ([summary](../studies/h08/summary.json), results.md H08): H05 nxconv setup with Dougherty e-e and e-i collisions, nu_ei 1e-3/1e-2 (nu_ee = nu_ei, nu_ii = nu_ei sqrt(m_e/m_i)), v_q/v_te 0.01/0.03, Nx 32/64 and Nn 64/128 agree to t = 1000; claims to 1000 (0.01) and 950 (0.03). Homogeneous controls vs the exact driven oscillator with drag to 3e-8, mean-field identity with the collisional current term to 3e-13, collisions do no net work. nu_ei = 1e-2 halves W_ext by t = 950 and heats electrons to 9-77 T_e0 (77 at v_q 0.03 is relativistic, outside the model). **Declared gate G1 FAILED** (nu = 0 rerun vs the nxconv record: 6e-6 in dK_e, gate 1e-6; cause found by the G1 follow-up study studies/h08_g1_cause.py: restart chunk length 240 (nxconv) vs 200 (H08), not the parent change; with chunk 240 the dd80b0f rerun reproduces the record exactly; the gate was mis-specified relative to restart sensitivity, recorded not amended). Limits: velocity-independent nu, beta convention unchecked, one seed realization. **LBO-G rerun** (`collision_model="dougherty_g"`, beta 0, studies/h08_lbog.py, [summary](../studies/h08_lbog/summary.json)): G0/G2/G3 pass; electron-ion exchange halves (-0.4999 to -0.4987), W_ext unchanged to 9e-8, dK_e and T_e change <= 1.8e-3; **declared gate G4 FAILED through T_i only** (-5% at t = 750 and -10% at 950 for v_q 0.03, nu_ei 1e-2: T_i is fed by the halved exchange; gate not amended) |
| H09 | 1D3V / 2D3V | partial: all runs are 1D3V; 2D3V not run |
| H10 | reduced envelope model | not started |

### R: research questions

| ID | Item | Status |
|---|---|---|
| R1 | finite coherence / reservoir | first step done 2026-10-18 ([results](results.md), gates declared in `studies/r1_common.py` before running): (a) mobile-ion self-consistent reservoir at the H05 seeded setup, v_q/v_te 0.01 and 0.03, U_D0/(n T_e V) = 0.1, 1, 10: kinetic U_D(t), W_D(t) follow the linear coupled-oscillator (Ebar, J, E_D, A_D) prediction to within 0.04 U_D0 over the whole claim windows (1000 and 940), beat times equal pi/eta, W_D follows the matched prescribed drive until t_split (52-500, equal to the linear t_split); no nonlinear detuning resolved. (b) finite coherence at v_q/v_te 0.01 (Lorentzian by phase diffusion, no phase resets, 6 kinetic realizations per tau_c, 400 linear): kinetic W_ext equals the linear same-drive W_ext within about 1% (no 10% departure) to t = 1000; the late rate saturates at about E0^2 tau_c omega_L^2/4 instead of growing. **Declaration defect:** the v1 coherence runs realized coherence time 2 tau_c (variance dt/tau_c, not 2 dt/tau_c); recorded, v2 declared and run with the corrected definition. Not done: Gaussian line, v_q/v_te >= 0.03 in (b), a regime where nonlinear detuning acts (needs stronger drive, beyond the certified windows), inhomogeneous dark fields **Second step (2026-10-20, gates in `studies/r1nl_gates.py` before any run; strong closure c = 64 + Lawson, Nx 32/64 x Nn 64/128 + c 32 pair, grid cross-check): certified windows 0.03: 1000, 0.05: 892, 0.07: 791.5 (limited by positivity of the Nx 64 members, not by pair disagreement). At the strongest drive v_q/v_te = 0.07: W_ext within 0.68% of the linear resonant law over the window (refinement spread 2e-4), mean-field frequency shift < 7e-6 omega_pe, finite reservoir (u = 1, 10) follows the linear coupled oscillator to 0.23% / 0.81% of U_D0, finite coherence (tau omega_pe = 100, 300, 6 realizations) R = 1.0050 +- 0.0058 / 1.0000 +- 0.0038: no saturation or detuning resolved, upper bound 0.7% (W_ext). The 0.7% deficit is independent of v_q (-1e-8 t^2) and scales as the seed amplitude squared (follow-up F1 pass; unseeded control 2e-9; hypothesis that it is a homogeneous-truncation effect rejected, F2 fail). Not done: windows beyond ~790 at 0.07 (blocked by the AW truncation instability), v_q/v_te > 0.07, Gaussian line, reservoir + coherence combined, other seed levels than 5e-4 (only scales 0, 0.1, 1 at t = 400).** |
| R2 | collisions / cosmology | done 2026-10-20 ([results](results.md#r2-sustained-absorption-with-collisions-vs-the-linear-drag-law-studiesr2_gatespy), gates declared in `studies/r2_gates.py` before running; 192 runs, EM and G beta = 0, v_q/v_te 0.01-0.05, nu_ei 1e-3-3e-2, omega resonant/0.9/1.1): the kinetic absorption equals the linear Drude drag law with nu_eff = (nu_ei/2)(1 + m_e/m_i): at nu_ei = 3e-2 the late rate is P_ss within 0.4% (steady state reached), elsewhere the kinetic work equals the exact transient-inclusive linear law within 5% except the nu_ei = 1e-3 detuned cases (+9-14%, independent of v_q, scales as seed^2: follow-up F pass); v_q exponent 1.99 +- 0.10; EM = G beta 0 in W_ext to 3e-6. Heating (T_e/T_e(0) up to 231, beyond v_te = 0.3 c past T_e/T_e(0) = 90) does not change the rate because nu is constant; ion heating small, e-to-i exchange 4e-5-1.6e-2 of W_ext. Declaration defect: T0 averaging window longer than its record (fails as declared), T0b passes. Not done (needed before any cosmological claim): Spitzer nu(T_e) ~ T_e^-3/2, radiative/Compton cooling, expansion, relativistic electrons, steady state at nu_ei <= 1e-2 resonant (needs t > 4000/nu-scale), k != 0 dark-photon coupling |
| R3 | inhomogeneous / magnetized | first step done 2026-10-20 ([results](results.md), gates declared in `studies/r3_gates.py` before running; follow-ups F and G declared in `studies/r3_followup_gates.py` before their runs): independent NumPy cold-fluid + Maxwell-Proca time-domain reference (energy drift 3.5e-10, refinement 4.5e-5, flux budget with the phi_D A_D term 3.4e-6 for transverse cases, without it 0.02-0.9 of E_in) and Dark-SPECTRAX 1D3V packets crossing a tanh density ramp (resonance omega_pe = Omega_D), kept in an exact discrete kinetic equilibrium (drift 3e-16), eta 0.15 and 0.3, both polarizations, B0 along/oblique/perpendicular. Transverse: kinetic equals the cold reference to 4e-4 (16 runs, ledger <= 2e-10 E_in); the simple Landau-Zener law FAILED R4 (2 of 4 gated cases, 0.035 / 0.045 at eta 0.15): its exponent is too large by 4.5 s^2, s = eta n_c/kbar^2 (0.99, 0.95, 0.89, 0.80, 0.61 at eta 0.05-0.3), accurate to 0.01 only for s <~ 0.1; the complex-turning-point (Dykhne) exponent reproduces T, reflected photon and reflected dark to 0.009 (k Lc >= 3, all eta). Longitudinal: the declared R5 prediction was wrong (WKB tunnelling exp(-2 int kappa) is the transmitted dark fraction, matched to 1.4e-3 for w >= 3, not the absorbed one); up to 28% of the packet stays at the cold resonance layer, whose phase-mixing wavenumber (about 40 at T = 330) needs Nx >= 2048 (K4/K5 FAILED at Nx 512/768; at Nx 3072 kinetic = cold to 6e-4 on propagating channels, layer energy 0.205 / 0.199, v_te 0.005-0.02 no resolved correction). B0 along the gradient or along the polarization (ordinary mode): no change; the transmitted fraction is insensitive to B0 (<= 0.004 at w >= 4), but oblique/perpendicular B0 turns the trapped longitudinal layer energy (0.20) into reflected ordinary photons (0.26-0.27) and moves 0.05-0.10 of transverse packets into a hybrid layer. Other failures recorded: R3 with-term budget 1.3e-3 in the 14 resonance cases (grid), K3 flux budget at Nx 512 (w 2: 1.5e-2), K5 4 of 15. Not done: non-relativistic packets, frequency-domain boundary-value solve (Beadle-Caputo-Ellis 2405.13882 read at abstract depth only), thermal absorption at the longitudinal layer (needs Nx >~ 12000, Nn >> 4), mobile ions, 2D, nonlinear amplitudes |
| R4 | observational (PSP) | not started here; the PSP chirp study lives in a separate repository |
| R5 | charged dark plasma | not started (the companion carries only the visible charge with mixing) |

## Literature: needed but not read

physics.md sets the rule: only full-text sources may set a benchmark.

| Source | Needed for | Access |
|---|---|---|
| Hakim, Francisquez, Juno, Hammett, arXiv:1903.08062 (Dougherty moments, N5) | B11, H08 conservative collisions | read 2026-10-16, Sec. 2 (operator, moments, d_v convention); multispecies coefficients deferred there |
| Francisquez, Juno, Hakim, Hammett, Ernst, arXiv:2109.10381 (multispecies Dougherty, LBO-G/EM/ET) | B11, H08 cross-species coefficients | read 2026-10-18; parent #71 = LBO-EM; LBO-G beta=0 added as parent #73 (opt-in); default and records unchanged |
| Juno, arXiv:2005.13539 (thesis, N11) | C06 continuum comparison, Weibel setups | open (arXiv) |
| Gkeyll documentation | C06 input matching | open (web) |
| O'Neil, Phys. Fluids 8, 2255 (1965) | B01 trapping time and bounce scaling | read 2026-10-15 (physics.md C1) |
| Gould, O'Neil, Malmberg, PRL 19, 219 (1967) | B06 echo theory | read 2026-10-15 (physics.md C2) |
| Zakharov, Sov. Phys. JETP 35, 908 (1972) | B10 | read 2026-10-15, Secs. 1-2 (physics.md C3) |
| Issan et al., arXiv:2312.05439, 2407.09626, 2412.07073 | anti-symmetric SW and square-root forms (design note option E) | read in full 2026-10-19; prototyped in studies/antisym (stage 2 fails) |
| Dai, arXiv:2608.09827 (N15) | stable AW Galerkin (design note option C) | held locally; prototyped in studies/dai (gate 1 fails in AW form) |

## Next steps, in order

1. **Dai stable-AW linear prototype: done, gate 1 failed** (lane "dai", `studies/dai/`, results.md section). Dai's
   eq. 27 form in AW coordinates is unstable in float64 at Nn >= 64 (frozen max Re lambda up to 1.1, Floquet up to 1.0,
   OTSI rates wrong); the same Galerkin method written in symmetric Hermite functions of width a/sqrt2 passes gates 1-2
   (max Re lambda <= 6e-15, OTSI within 0.96% of Silin). Keep option A (#66 closure); the next candidate is option B (SW),
   whose design tests 3-4 (H05 0.1 to t = 700/1000, energy ledger) are the follow-up.
   **Option B (SW) tests 3-4: done, rejected** (`studies/sw/`, results.md): all SW runs blow up at t = 637-738,
   the Nx16 Nn pair agrees only to t = 634 and the Nx16 Nn64 ledger hits 1.2e-3 W_ext before t = 600. A stays.
   **Anti-symmetric square-root SW (Issan et al., design note option E): done, rejected** (`studies/antisym/`, results.md):
   stage 1 linear gates pass (max Re lambda <= 4e-15, Silin 0.25%, 1.5x AW cost) but the H05 0.1 pilot departs from the grid at
   t = 391 (Nx 16) / 498 (Nx 32), earlier than SW (634-738), independent of Nn and dt. A stays.
2. **v_q/v_te = 0.1 grid reference with vmax > 32 v_te: done** (lane "gridref", batch 11, results.md). vmax 48/64
   moves edge contact only from t = 701.5 to 754.5 (tail accelerated past 64 v_te), and negativity at t = 645.5 is
   set by Nx, not the box. Grid reference certified for dK_e/W_ext to about 750, spectrum to about 640; Hermite c = 4
   agrees with it in dK_e/W_ext to >= 760 and in the spectrum to 555. A t = 1000 reference at 0.1 needs a different
   method (e.g. larger Nx with an adaptive or mapped v box); the Hermite window near 600 is a Hermite limit.
3. ~~Rerun the B00/B02/B03/B06 records at the current pin~~ done 2026-10-15: bit-identical (see Code health).
4. **Fetch and read the open-access papers** (1903.08062, 2005.13539). 2005.01515 was read 2026-10-15 (Sec. 1.1):
   the canonical map agrees, see physics.md. O'Neil 1965, Gould-O'Neil-Malmberg 1967 and Zakharov 1972
   were read 2026-10-15 (physics.md C1-C3).
5. ~~Cheap B rungs: B05 Buneman, B04 ion-acoustic~~ done 2026-10-15 (results.md); open: B04 damping at T_e/T_i = 20.
   Decisive: growth/damping vs kinetic roots to 1e-4, ordinary and dark.
6. ~~B08 magnetized rotation and EM waves~~ done 2026-10-15 for uniform B0 (results.md; transverse dark branch now tested);
   ~~B09 Weibel~~ linear done 2026-10-16; nonlinear first maximum reported 2026-10-16 with a magnetic field-scaled closure (Weibel only), but only under post-hoc amendments 1 and 3: declared criterion 1 and gate G3a FAILED; post-peak evolution and filamentation open. Open from B08: a warm (Bernstein) reference for the lower X branch near 2|Omega_ce|.
7. C09 done (2026-10-15). Open: gradient of a fitted Landau damping rate, through `run_adaptive` remaps, Diffrax checkpoint defect reported upstream as [diffrax#788](https://github.com/patrick-kidger/diffrax/issues/788). C08 CPU/GPU done for one case (see C08 row).
8. **Collisions (B11/H08/R2)**: B11 done 2026-10-16 with the parent Dougherty operator (#71). Next: collisional
   Landau damping vs a kinetic root, H08 HHS run with collisions, R2.

## HHS reference studies (hkr, relpic; results 2026-10-25, see results.md)

- **relpic done.** 12 production runs of the relativistic Boris PIC (Dark-JAX-in-Cell on JAX-in-Cell `2d693cb`) and Newtonian controls, 3 seeds each to T = 2000. Passed: G1/G2 (all runs), G3, G6, G7 (quintic shape), G9 (no-drive control), single-particle S1-S3. Failed as declared: G4 (ppc 412, t = 1000, +19%), G5 (dx 0.02, t = 400/600, +19%/+18%), G10 (Nx 32 Hermite record 1-2 orders below the PIC). Relativistic vs Newtonian dW_th separate by 2.6x at late time (>= 3 s.e. at every time); relativistic PIC is within 8-15% below the digitized HHS Fig. 2 curve (about 25 E_th at t = 1000); peak k != 0 field energy is 4 E_th. The ~50 E_th is not reproduced by the relativistic run; the Newtonian run gives 56 E_th in dW_th. Open: identify which HHS quantity carries ~50 E_th (not the k != 0 field energy or R's dW_th).
- **hkr done.** Grid Vlasov chains Nx 64-1024 (k lambda_De up to 1.69): the deterministic dE_e is ~2.3 E_th at t = 1000 for Nx >= 512 (Nx 256, 640, 1024 agree over their certified windows); the shot-noise seed loses positivity at t = 69-163 with dE_e = 0.013 E_th for every Nx (Q1: cutoff does not matter; 25 E_th not reached). Hermite c = 64 + Lawson: Nx 128 fails at t = 976.5 (max_steps), Nx 256 at 898 (positivity), Nx 512 stalled in the chunk 880-920 (stopped after 3 h wall). V1, V1b, V2, V2b and V2c at k lambda_De 1.0 failed as specified earlier (`studies/hkr/validation*.json`).
