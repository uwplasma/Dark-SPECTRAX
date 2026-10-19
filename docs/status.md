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
| B10 | Langmuir modulational / Zakharov | not started; Zakharov 1972 Secs. 1-2 read (physics.md C3): use eqs. 2.8/2.9 directly, the printed gamma_max = q w omega_p/16 does not follow from 2.8 (gives /4); no dark term at that order |
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
| H04 | relativistic drive | blocked: the model is nonrelativistic; needs a relativistic Hermite formulation or the PIC code |
| H05 | strong resonant drive | partial: certified windows v_q/v_te = 0.01 to 1000, 0.03 to about 960-1000, 0.1 to about 600 (grid reference to about 750); 0.1 to t = 1000 blocked by the AW truncation instability and the OTSI high-k stage. **Resolution extension on GPU (2026-10-18, gates in `studies/gpures_gates.py`) FAILED**: v_q 0.1 and 0.05 with Nx 64/128 x Nn 128/256 (c = 4) all end at the step budget inside the instability (0.1: t = 595, 383, 453, 328; 0.05: 752, 555, 671, 440), more Nn or Nx ends sooner; pairs agree to the end of the shorter run but G1 (to 1000) and G4 fail, no new window; going further needs a stable basis (design option B), not resolution; v_q 0.05 certified only to ~440-555 ([summary](../studies/gpures/summary.json)) |
| H06 | swept drive | partial: homogeneous check done; kinetic runs lose resolution before the crossing |
| H07 | finite dark reservoir | done (homogeneous, eta 1e-3 / 0.03) |
| H08 | collisions | pilot done ([summary](../studies/h08/summary.json), results.md H08): H05 nxconv setup with Dougherty e-e and e-i collisions, nu_ei 1e-3/1e-2 (nu_ee = nu_ei, nu_ii = nu_ei sqrt(m_e/m_i)), v_q/v_te 0.01/0.03, Nx 32/64 and Nn 64/128 agree to t = 1000; claims to 1000 (0.01) and 950 (0.03). Homogeneous controls vs the exact driven oscillator with drag to 3e-8, mean-field identity with the collisional current term to 3e-13, collisions do no net work. nu_ei = 1e-2 halves W_ext by t = 950 and heats electrons to 9-77 T_e0 (77 at v_q 0.03 is relativistic, outside the model). **Declared gate G1 FAILED** (nu = 0 rerun vs the nxconv record: 6e-6 in dK_e, gate 1e-6; cause found by the G1 follow-up study studies/h08_g1_cause.py: restart chunk length 240 (nxconv) vs 200 (H08), not the parent change; with chunk 240 the dd80b0f rerun reproduces the record exactly; the gate was mis-specified relative to restart sensitivity, recorded not amended). Limits: velocity-independent nu, beta convention unchecked, one seed realization. **LBO-G rerun** (`collision_model="dougherty_g"`, beta 0, studies/h08_lbog.py, [summary](../studies/h08_lbog/summary.json)): G0/G2/G3 pass; electron-ion exchange halves (-0.4999 to -0.4987), W_ext unchanged to 9e-8, dK_e and T_e change <= 1.8e-3; **declared gate G4 FAILED through T_i only** (-5% at t = 750 and -10% at 950 for v_q 0.03, nu_ei 1e-2: T_i is fed by the halved exchange; gate not amended) |
| H09 | 1D3V / 2D3V | partial: all runs are 1D3V; 2D3V not run |
| H10 | reduced envelope model | not started |

### R: research questions

| ID | Item | Status |
|---|---|---|
| R1 | finite coherence / reservoir | first step done 2026-10-18 ([results](results.md), gates declared in `studies/r1_common.py` before running): (a) mobile-ion self-consistent reservoir at the H05 seeded setup, v_q/v_te 0.01 and 0.03, U_D0/(n T_e V) = 0.1, 1, 10: kinetic U_D(t), W_D(t) follow the linear coupled-oscillator (Ebar, J, E_D, A_D) prediction to within 0.04 U_D0 over the whole claim windows (1000 and 940), beat times equal pi/eta, W_D follows the matched prescribed drive until t_split (52-500, equal to the linear t_split); no nonlinear detuning resolved. (b) finite coherence at v_q/v_te 0.01 (Lorentzian by phase diffusion, no phase resets, 6 kinetic realizations per tau_c, 400 linear): kinetic W_ext equals the linear same-drive W_ext within about 1% (no 10% departure) to t = 1000; the late rate saturates at about E0^2 tau_c omega_L^2/4 instead of growing. **Declaration defect:** the v1 coherence runs realized coherence time 2 tau_c (variance dt/tau_c, not 2 dt/tau_c); recorded, v2 declared and run with the corrected definition. Not done: Gaussian line, v_q/v_te >= 0.03 in (b), a regime where nonlinear detuning acts (needs stronger drive, beyond the certified windows), inhomogeneous dark fields |
| R2 | collisions / cosmology | unblocked: conservative Dougherty operator available (B11); not started |
| R3 | inhomogeneous / magnetized | not started (uniform-B0 B08 done 2026-10-15) |
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
| Dai, arXiv:2608.09827 (N15) | stable AW Galerkin (design note option C) | held locally; prototyped in studies/dai (gate 1 fails in AW form) |

## Next steps, in order

1. **Dai stable-AW linear prototype: done, gate 1 failed** (lane "dai", `studies/dai/`, results.md section). Dai's
   eq. 27 form in AW coordinates is unstable in float64 at Nn >= 64 (frozen max Re lambda up to 1.1, Floquet up to 1.0,
   OTSI rates wrong); the same Galerkin method written in symmetric Hermite functions of width a/sqrt2 passes gates 1-2
   (max Re lambda <= 6e-15, OTSI within 0.96% of Silin). Keep option A (#66 closure); the next candidate is option B (SW),
   whose design tests 3-4 (H05 0.1 to t = 700/1000, energy ledger) are the follow-up.
   **Option B (SW) tests 3-4: done, rejected** (`studies/sw/`, results.md): all SW runs blow up at t = 637-738,
   the Nx16 Nn pair agrees only to t = 634 and the Nx16 Nn64 ledger hits 1.2e-3 W_ext before t = 600. A stays.
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
