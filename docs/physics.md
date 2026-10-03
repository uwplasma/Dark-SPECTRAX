# Physics, units and numerical method

## Model

Each species obeys the collisionless Vlasov equation in the combined force field

$$
\partial_t f_s+\mathbf v\cdot\nabla f_s+\frac{q_s}{m_s}\left[\mathbf E+\eta\mathbf E_D+\mathbf E_{\rm drive}
+\mathbf v\times(\mathbf B+\eta\mathbf B_D)\right]\cdot\nabla_v f_s=0 .
$$

Ordinary fields: $\partial_t\mathbf E=c^2\nabla\times\mathbf B-\mathbf J/\epsilon_0$, $\partial_t\mathbf B=-\nabla\times\mathbf E$.
Canonical Proca fields (same convention as Dark-JAX-in-Cell):

$$
\partial_t\mathbf E_D=c^2\nabla\times\mathbf B_D+\Omega_D^2\mathbf A_D-\eta\mathbf J/\epsilon_0,\quad
\partial_t\mathbf B_D=-\nabla\times\mathbf E_D,\quad
\partial_t\mathbf A_D=-\mathbf E_D-\nabla\phi_D,\quad
\partial_t\phi_D=-c^2\nabla\cdot\mathbf A_D,
$$

with constraints $\nabla\cdot\mathbf E=\rho/\epsilon_0$, $\nabla\cdot\mathbf E_D+\Omega_D^2\phi_D/c^2=\eta\rho/\epsilon_0$,
$\mathbf B_D=\nabla\times\mathbf A_D$; $\rho$ includes the uniform background `rho_background` in both laws.

Modes: `ordinary` (no dark coupling), `prescribed_drive` (uniform $\mathbf E_{\rm drive}=\mathbf E_0\cos(\omega t+\varphi)$
and an external-work ledger, no dark reservoir), `self_consistent` (evolved Proca fields, complete energy).

## Units (parent normalization, SPECTRAX `ab87385`)

Velocities are in units of $c$ (the curl coefficient is 1). With `Om0 = Omega_cs[0]`, the parent equations are
$\dot{\mathbf E}=i\mathbf k\times\mathbf B-\mathbf J/\Omega_0$ and force factor $q_s\Omega_{cs}$, with
$n_s=\alpha_x\alpha_y\alpha_z C_{000}$ and $\mathbf J=\sum_s q_s\int\mathbf v f_s$. The energy theorem
$\dot K=\Omega_0\langle\mathbf J\cdot\mathbf E\rangle=-\dot U_\gamma$, $U_\gamma=\tfrac12\Omega_0^2\langle E^2+B^2\rangle$,
holds only if $m_s\Omega_{cs}=\Omega_0$ for every species; Dark-SPECTRAX therefore defines the species masses as
$m_s=\Omega_0/\Omega_{cs}$ (species 0 has unit mass). Then $\omega_L^2=\sum_s q_s^2n_s\Omega_{cs}/\Omega_0$.
The dark sector uses the same source times $\eta$ and the same prefactor:

$$
U_D=\tfrac12\Omega_0^2\left\langle E_D^2+B_D^2+\Omega_D^2(A_D^2+\phi_D^2)\right\rangle,\qquad
P_D=\Omega_0\eta\langle\mathbf J\cdot\mathbf E_D\rangle,\quad P_\gamma=\Omega_0\langle\mathbf J\cdot\mathbf E\rangle,\quad
P_{\rm ext}=\Omega_0\bar{\mathbf J}\cdot\mathbf E_{\rm drive}.
$$

$\langle\cdot\rangle$ is a box average evaluated from rfft coefficients with weights 1 (k_x = 0), 1 (Nyquist, even
N_x only) and 2 otherwise. $W_\gamma, W_D, W_{\rm ext}$ are integrated as extra ODE states, so they use the
same Runge-Kutta stages as the fields. The ledger is $K-K_0=W_\gamma+W_D+W_{\rm ext}$,
$U_\gamma-U_{\gamma0}=-W_\gamma$, $U_D-U_{D0}=-W_D$.

## Discretization

Velocity: Cartesian asymmetrically weighted Hermite functions in all three directions with independent orders
`(Nn, Nm, Np)` (parent storage order `(p, m, n)`), fixed widths `alpha_s` and centers `u_s`. Space: Fourier, rfft
along x, the parent 2/3 mask. One parent `Hermite_Fourier_system` call per RHS receives the physical force fields
$E+\eta E_D+E_{\rm drive}$, $B+\eta B_D$; `plasma_current` is evaluated once and feeds both Ampere laws. Ordinary
Maxwell uses only $(E,B)$. Time: Diffrax Dopri8 with a PID controller (complex state, as in the parent).
The parent closure `nu` (hypercollision $\propto n(n-1)(n-2)$) is numerical and leaves Hermite orders 0-2 unchanged.

Requirements found while testing:

* The kinetic-energy ledger closes only when every direction that is forced has Hermite order at least 3
  (indices 0, 1, 2). With order 2 in y and z, a run with $E_z\neq0$ showed ledger defects of $5\times10^{-7}$
  to $2\times10^{-4}$; with order 3 they fall to solver tolerance.
* Avoid grid sizes divisible by 3: the parent mask keeps $|m|\le\lfloor N/3\rfloor$, and for $N=3K$ the product
  of two $+K$ modes aliases into $-K$. A strict-mask upstream fix is in progress (separate SPECTRAX PR).
* Initial data must lie in the active band; the helpers here only populate retained modes.

## Initialization

* `maxwellian`: basis-matched drifting Maxwellians plus density perturbations (conjugate pair stored for k_x = 0).
* `consistent_fields`: $E_k=-i\mathbf k\rho_k/(\Omega_0k^2)$ plus a supplied mean field; Yukawa near field
  $\phi_{D,k}=\eta\rho_k/[\Omega_0(k^2+\Omega_D^2)]$, $E_D=-i\mathbf k\phi_D$, $A_D=B_D=0$. Nonzero mean charge
  is rejected; for $\Omega_D=0$ the zero mode uses the gauge $\phi_{D,0}=0$.
* `proca_mode`: free vacuum mode, $\omega^2=k^2+\Omega_D^2$, $\phi_D=\mathbf k\cdot\mathbf A_D/\omega$,
  $E_D=i\omega A_D-i\mathbf k\phi_D$, $B_D=i\mathbf k\times A_D$; at k = 0 the real mean field
  $A_D=\mathrm{Re}\,A$, $E_D=\mathrm{Re}(i\omega A)$.

## Independent re-derivation of the planning equations

Re-derived by hand and, where marked, checked numerically by the tests:

| Item | Verdict |
|---|---|
| Proca equations, Lorenz condition, dark Gauss law from $\partial_\mu F^{\mu\nu}+m^2A^\nu=j^\nu$ | correct (A02, A05) |
| Canonical mixing map $\eta=\chi/\sqrt{1-\chi^2}$, $m_D=m_0/\sqrt{1-\chi^2}$ | correct (algebra only) |
| $U_D$ with $+\Omega_D^2(A_D^2+\phi_D^2/c^2)$ and flux $\epsilon_0c^2E_D\times B_D+\epsilon_0\Omega_D^2\phi_DA_D$ | correct (A02, A03) |
| Hermite ladders $V_i$, $D_i$, moments $M_i$, $M_{ii}$, $M_{ij}$, Levi-Civita magnetic term | correct (A00 quadrature, 1e-12) |
| $E_{D,x}/E_x=\eta Q/(Q-\Omega_D^2)$, $D_L=(Q-\Omega_D^2)(1+\chi)+\eta^2Q\chi$ | correct; the elimination uses $A_D=-i\omega E_D/Q$, so it assumes $\omega\neq0$, $Q\neq0$ (B00) |
| Cold limits and homogeneous roots $(\omega^2-\omega_L^2)(\omega^2-\Omega_D^2)-\eta^2\omega_L^2\omega^2=0$ | correct (A03) |
| Mean-pump identity and $\bar E=-E_0\omega t\sin(\omega t)/2$ | correct; exact in the discrete system only if the closure leaves orders 0-2 untouched (A07) |
| B12 drag response and absorbed power | correct (algebra only) |
| Dougherty moments $R_{sr}$, $Q_{sr}$ with $\theta=\mathrm{tr}P/(3mn)$ | correct (algebra only) |
| Midpoint phase error $-\omega^3\Delta t^2t/12$ | correct |
| Free-streaming front $\lvert g_n\rvert^2\propto[(kv_tt)^2]^n e^{-(kv_tt)^2}/n!$ | correct only for a basis-matched Maxwellian, $a=\sqrt2v_t$; other widths change the prefactors |
| "n_s = alpha_x alpha_y alpha_z C000", "Ampere source -J/Omega_cs[0]", "energy prefactor Omega_cs[0]^2" | correct, but the energy theorem additionally needs $m_s\Omega_{cs}=\Omega_0$ (above) |

## Source-to-model table

Only sources whose equations or code were actually read for this repository are entered as used. Classification:
E = exact identity, A = controlled approximation, N = numerical finding.

| Source (version) | What was inspected | Field basis / mixing | Kinetic model, dims | Comparison made here | Applicability |
|---|---|---|---|---|---|
| SPECTRAX, uwplasma, commit ab87385 | `_model.py`, `_simulation.py`, `_initialization.py`, `_diagnostics.py` | ordinary Maxwell, parent normalization | Hermite-Fourier Vlasov, 3V Cartesian, 1-3D | A01 RHS (E) and trajectory (N) regression | parent; not an independent check |
| Dark-JAX-in-Cell, commit d547579 | README equations, `examples/dark_kinetic.py`, `physical_kinetic_{32,64,128}` records | canonical Proca, same eta convention | 1D3V PIC, quadratic shapes, Boris | C05 same-input Landau fits and roots (N) | independent kinetic discretization, same group conventions |
| Imported references, `studies/refs` (6781d80 era) | all of `code/`, B00/B02 records | none (ordinary) | NumPy semi-Lagrangian 1D1V VP; wofz roots; Gkeyll p=2 records | B00, B02 against Hermite rerun on ab87385 (N) | non-Hermite continuum, ordinary only |
| Plasma dispersion function (SciPy `wofz`, Faddeeva) | implementation used directly | - | linear Vlasov, Maxwellian populations | B00, B02, B03, C05 roots (E for the model) | linear, causal continuation, k > 0 |
| Canonical kinetic mixing (derived here, Section "Independent re-derivation") | own derivation | Lagrangian rotation | - | eta, m_D mapping (E, algebra only) | not yet compared with [D10] |
| HHS v1 (arXiv:2510.13956v1) | full text incl. App. A-C (literature check) | prescribed uniform drive, electron omega_p | 1D1V relativistic PIC | H00-H02, H07 inputs (studies/hhs_ladder.py) | nonrelativistic, deterministic seeds: inspired by, not a reproduction |

### Literature reading status (2026-10-03 check)

Depth: F = full text read for the stated items; A = abstract and metadata; M = bibliographic metadata only.
Only F sources may set a benchmark; A/M sources have not had their equations inspected and are not used for any claim.

| Ref | Corrected citation | Depth |
|---|---|---|
| D1 | Hook, Huang, Shalaby, arXiv:2510.13956v1 (only version); PRL 137, 071004 (2026), DOI 10.1103/98cx-7t43 | F for v1 text, eqs. 1-47, App. A-C; published PRL text and supplement not accessed |
| N1 | Parker, Dellar, arXiv:1407.1932; J. Plasma Phys., DOI 10.1017/S0022377814001287 | A |
| N2 | Issan, Chapurin, Koshkarov, Delzanno, arXiv:2412.07073v3 | A |
| N3 | Pagliantini et al., arXiv:2110.11511v3 | A |
| N4 | Pagliantini, Delzanno, Markidis, arXiv:2208.14373; JCP, DOI 10.1016/j.jcp.2023.112252 | A |
| N5 | Hakim, Francisquez, Juno, Hammett, "Conservative Discontinuous Galerkin Schemes for Nonlinear Fokker-Planck Collision Operators", arXiv:1903.08062; JPP, DOI 10.1017/S0022377820000586 | M |
| N10 | Shalaby, Broderick, Chang, Pfrommer, Lamberts, Puchwein, "SHARP: A Spatially Higher-order, Relativistic Particle-in-Cell Code", arXiv:1702.04732; ApJ, DOI 10.3847/1538-4357/aa6d13 | M |
| N11 | Juno, "A Deep Dive into the Distribution Function: Understanding Phase Space Dynamics with Continuum Vlasov-Maxwell Simulations", arXiv:2005.13539 | M |
| N14, N15 | Shao, Wang, Wu, arXiv:2605.17820v1; Dai, arXiv:2608.09827v2 | A |
| N18 | Koshkarov et al., arXiv:2004.12010; CPC, DOI 10.1016/j.cpc.2021.107866 | M |
| N22 | Bell, Campos Pinto, Possanner, Sonnendruecker, arXiv:2504.04929; J. Comput. Phys. 555, 114765 (2026), DOI 10.1016/j.jcp.2026.114765 (use this DOI, not the SSRN DOI on the arXiv record) | A |
| D3 | Witte, Rosauro-Alcaraz, McDermott, Poulin, "Dark Photon Dark Matter in the Presence of Inhomogeneous Structure", arXiv:2003.13698; JHEP 06 (2020) 132 | M |
| D5 | Brahma, Schutz, arXiv:2410.14771; related application: Brahma, Iles, Scherer, Schutz, "Resonant axion and dark photon production in magnetic white dwarfs", PRD 113, 083010 (2026), DOI 10.1103/9swg-ndgx | M |
| N6-N9, N12, N13, N16, N17, N19-N21, N23, N24, D2, D4, D6-D13 | as in the planning reading map (identifiers verified) | M |

HHS v1 facts used here (F): uniform k = 0 drive A0 cos(omega t) with no stated ramp; v_q^D/v_te = 0.03 and 1e-3 for the
resonant runs; m_i/m_e = 1836, T_e = T_i = 1e-3 m_e c^2, L = 40 c/omega_pe, 1000 cells; relativistic Vay pusher in 1D1V;
quiet co-located start; timestep not stated; the resonant omega (electron-only or total omega_p) is not stated. Their
eqs. (6), (7) give nu_ei tau_ei = m_p/(2 m_e), while footnote 5 states the opposite ordering (v1 only). The swept drive
must be read as theta(t) = (0.8 + 0.1 t/5000) t, instantaneous frequency 0.8 + 0.2 t/5000 (`sweep_drive` in `Model`).
Runs here are labelled HHS-v1-inspired and nonrelativistic; none is a reproduction.
