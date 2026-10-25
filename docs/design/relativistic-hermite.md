# Relativistic momentum-space Hermite model: design and 1D1V prototype

Status: prototype passes its declared gates (records `studies/relherm/gates_*.json`, code `studies/relherm/relherm.py`,
gates `studies/relherm/relherm_gates.py`). Not yet in SPECTRAX; the PR plan is in section 8.

## 1. Why this is small

Use the momentum variable $\mathbf u=\mathbf p/(mc)$ (per species) instead of $\mathbf v$. The Vlasov-Maxwell equation is

$$\partial_t f+\mathbf v(\mathbf u)\cdot\nabla f+\frac{q}{m}\left(\mathbf E+c\,\mathbf v\times\mathbf B\right)\cdot\nabla_u f=0,\qquad \mathbf v=\frac{\mathbf u}{\gamma},\ \gamma=\sqrt{1+u^2/c^2}.$$

With the SPECTRAX asymmetrically weighted basis $f=\sum_n C_n\psi_n(\xi)/\sqrt\pi$, $\xi=(u-u_0)/a$, $\psi_n=H_ne^{-\xi^2}/\sqrt{2^nn!}$, test functions
$\phi_n=H_n/\sqrt{2^nn!}$ (polynomials), the electric force term is the existing one, unchanged by relativity:
$\partial_tC_n\ni(q/m)(\sqrt{2n}/a)\,E\,C_{n-1}$ (it is $\int f\,\partial_u\phi_n$, and $\partial_u$ of a polynomial is exact). Only three objects contain $\mathbf v(\mathbf u)$:
streaming, the current, and the kinetic energy (with $\gamma-1$). In the basis, multiplication by $g(u)$ is the Galerkin matrix
$G_{nm}=\langle\phi_n,g\,\phi_m\rangle_w$; with the Hermite recurrence matrix $T=P\,\mathrm{diag}(\lambda_j)P^T$ (the eigendecomposition already used by the Lawson streaming, SPECTRAX PR #83; $\lambda_j$ are the Gauss-Hermite nodes), $G=g(u_0+aT)=P\,\mathrm{diag}\,g(u_0+a\lambda_j)P^T$ (Gauss-Hermite quadrature of the product). Hence

* streaming: $\partial_tC=-ik\,V C$, $V=P\,\mathrm{diag}\,v(u_0+a\lambda_j)P^T$ (nonrelativistic: $V=u_0+aT$, the existing ladder);
* current: $J=q\,a\,(VC)_0$ (existing: $q a(aC_1/\sqrt2+u_0C_0)$);
* kinetic energy density: $m c^2 a\,(\Gamma C)_0$, $\Gamma=P\,\mathrm{diag}(\gamma_j-1)P^T$.

The same $P$ serves every species, width and shift; only the nodal diagonal changes. $V$ is real symmetric, so $-ikV$ is skew-Hermitian and its exponential is unitary: the Lawson factor
$e^{-i\tau kV}=P\,\mathrm{diag}\,e^{-i\tau k v_j}P^T$ is exact and differs from PR #83 only by the diagonal phases ($k v(u_0+a\lambda_j)$ instead of $k(u_0+a\lambda_j)$).

## 2. Three dimensions

$v_i=u_i/\gamma(u)$ couples the axes through $\gamma$ (no tensor product of 1D factors). On the tensor Gauss-Hermite grid $v_i$ is still diagonal: $V_i=(P\otimes P\otimes P)\,\mathrm{diag}(v_i(\mathbf u_{jkl}))(P\otimes P\otimes P)^T$, applied as three successive $N\times N$ contractions, $O(N^4)$ per spatial mode against $O(N^3)$ for the ladder. The magnetic term $(\mathbf v\times\mathbf B)\cdot\nabla_uf=\nabla_u\!\cdot(f\,\mathbf v\times\mathbf B)$ (divergence-free because $\mathbf v$ is a gradient field in $u$) is the test-side lowering $D_j$ contracted with the nodal $V_i$ and $\epsilon_{ijk}B_k$ (order matters, they do not commute); sketch only, not prototyped. 1D2V/1D3V with a transverse $B$ are the next prototype step.

## 3. Projection error and aliasing

$v$ and $\gamma$ are analytic in $u$ with branch points at $u=\pm ic$. For a core of width $a\ll c$ the Hermite coefficients of $g(u_0+a\xi)$ decay like $\sqrt{n!}\,(a/\sqrt2\rho)^n$ with $\rho=|u_0\pm ic|$, i.e. down to $n\sim2\rho^2/a^2$ (about 4000 for $a=0.045$, $u_0=0$): the nodal projection of $v$ is spectrally accurate for any resolvable $N$. What the nodal product does not do is dealias the physics (that stays the 2/3 rule in $x$). The weight limits hot plasmas: Juttner tails $e^{-|u|/\theta}$ decay slower than the Gaussian weight's inverse, so the AW coefficients exist but reconstruction is only weak; expect trouble for $\theta\gtrsim0.3$ (not needed for HHS, $\theta=10^{-3}$). The large quiver is carried by the pump frame, which stays exact: a uniform $E$ translates $f$ in $u$ exactly, relativistic or not ($\dot u_0=(q/m)\bar E$).

## 4. Conservation

* Charge: the $n=0$ row has no force term and streaming is a total $x$-derivative; exact to round-off.
* Momentum: the row $u$ is linear; $\dot P=qnE$ exact (force), flux exact.
* Energy: streaming contributes $-ik\,e_0^T\Gamma VC$, a total derivative (Gamma and V commute and are symmetric): exact. The force contributes $(q/m)E\,e_0^T\Gamma D C$ with $D$ the lowering ladder, while the work is $E J=q E a\,e_0^TVC$. Their difference is $\epsilon=(q/m)E\sum_nr_nC_n$, $r_n=\sqrt{2(n+1)}\,\Gamma_{0,n+1}/a-V_{0n}$ for $n\le N-2$, $r_{N-1}=-V_{0,N-1}$: the derivative of the projected $\gamma$ differs from the projection of $v=\gamma'$ only in the truncated top modes. The semi-discrete work identity is therefore exact for polynomial $v$ (nonrelativistic, $N\ge3$) and holds to the spectral projection error otherwise; $\epsilon$ is measurable per RHS (one dot product) and should be reported as a diagnostic. Measured (section 6): $6.5\times10^{-10}$ (gate d) and $8\times10^{-9}$ (d2, $v/c=0.83$) of the energy scale at $N=24$.

## 5. Pump frame, what changes

The momentum-space shift stays exact (above) and the force term is unchanged, so the pump frame survives. What changes: (i) the streaming velocity of a shifted species is $v(u_0+a\lambda_j)$, not $u_0+a\lambda_j$: the mean drift is not $u_0$, and the effective thermal spread is $a/\gamma^3$ (parallel), so the drift and widths used for diagnostics must come from the nodal $v$; (ii) $V$ must be rebuilt when $u_0$ changes ($N^3$, once per RHS per species; negligible at $N\le128$); (iii) kinetic energy is not quadratic in $(u_0,C)$, so the ledger no longer closes to round-off by algebra; it closes to the integrator and to $\epsilon$; (iv) a relativistic moving frame is a Lorentz boost, not a Galilean one: only the *momentum* translation by a uniform force is exact; the box is not boosted. The pump frame with a uniform $\mathbf B\neq0$ is not covered here (the $\mathbf v\times\mathbf B$ force is not a pure $u$-translation).

## 6. Prototype results (1D1V, gates declared in the same commit as the code, before any run)

Details: `studies/relherm/relherm_gates.py` docstring, `studies/relherm/gates_*.json`.

| gate | what | result |
|---|---|---|
| a1 | $c\to\infty$ (`inv_c2=0`) vs `Hermite_Fourier_system` / `plasma_current` of SPECTRAX b668d87, $N=20$, $N_x=8$, random state | rel err $4.4\times10^{-16}$ (RHS), $7.3\times10^{-16}$ (current): PASS ($\le10^{-12}$) |
| a2 | `inv_c2=1e-8` | $1.0\times10^{-8}$: PASS ($\le10^{-6}$) |
| b | pump frame, $v_q=0.03$, $\omega_L$, $T=500$, $N=24$, RK4 dt 0.01 vs exact kinetic ODE | E $1.5\times10^{-8}$, $W_{ext}$ $1.1\times10^{-8}$, phase $1.4\times10^{-8}$ rad: PASS ($\le10^{-6}$); the nonrelativistic ODE misses $W_{ext}$ by 38% (discriminates) |
| b2 | lab frame, $v_q=0.01$, $T=300$, $N=48$ | E $1.2\times10^{-8}$, W $1.5\times10^{-9}$, phase $1.1\times10^{-8}$: PASS |
| c | eigenvalues of the linearised system vs relativistic Landau root | FAILED as declared: the truncated operator has a real undamped discrete spectrum ($\mathrm{Im}\,\omega\sim10^{-16}$), and the nonrelativistic control fails the same way (Re error $2\times10^{-3}$ at $k\lambda_D=0.3$); the declaration was wrong, not the model |
| c2 | declared after c: Schur-complement response function $\varepsilon_N(\omega)$ vs the Juttner $\varepsilon$ at $\mathrm{Im}\,\omega=0.1,0.2$, $k\lambda_D=0.2,0.3$, $N=128$ | max err $3.1\times10^{-5}$ (rel), control $3.8\times10^{-5}$; rel/nonrel $\varepsilon$ differ by $\ge2\times10^{-3}$: PASS ($\le10^{-4}$, discrimination $\ge5\times10^{-4}$) |
| d | ledger $K-K_0+\Delta U-W_{ext}$ with $(\gamma-1)$, nonlinear, $v_q=0.3$, $T=200$ | $6.6\times10^{-10}$ of $W_{ext}$: PASS ($\le10^{-5}$); but the relativistic detuning stops the drive ($u_0=0.13$ at $T$; nonrel run: 0.50) so the quiver stays moderate |
| d2 | declared after d: off-resonant drive, $\omega=3$, $v_q=100$ | max $|u_0|=1.52$, $v/c=0.83$; ledger $8\times10^{-9}$ of $\max|K-K_0|$: PASS |
| e | cost per kinetic RHS, CPU, 2 species, $N_x=32$ | rel 0.09/0.37/0.50 ms vs SPECTRAX ladder 0.09/0.07/0.16 ms at $N=32/64/128$: ratio 1-5; **noisy**, laptop load 35-85 during the run, to be repeated on an idle machine |

The impact of relativity is already large in the homogeneous problem: $W_{ext}(T=500)$ at $v_q=0.03$ is 0.0204 relativistic vs 0.0282 nonrelativistic (exact ODE), the detuning that `relgrid` found.

## 7. Literature (fetched/consulted)

* Juno, Hakim, Philippov et al., "Modeling of Relativistic Plasmas with a Conservative Discontinuous Galerkin Method", arXiv:2602.17487 (in `papers_dark/arxiv`): states that for spectral and finite-element conservative methods "there exists no obvious basis choice when we add relativity", because the invariants involve non-polynomial velocity integrals; their DG solution uses a velocity-space mapping. Our construction does not make the invariants polynomial; it makes them quadrature-exact in the retained space and measures the remainder ($\epsilon$, section 4).
* Hermite spectral Vlasov (nonrelativistic): Holloway 1996, Schumer-Holloway 1998, Delzanno 2015, Camporeale et al. arXiv:1311.2098 / 1312.4991, Koshkarov et al. 2004.12010, Pagliantini et al. 2110.11511, Issan et al. 2412.07073 (already read for the AW design). The search for "relativistic Vlasov Hermite spectral / Juttner basis / Gauss-Hermite in momentum" returned no prior relativistic Hermite Vlasov solver (arXiv searches 2026-10-24; not exhaustive). Vencels et al. (Hermite in 3V with the Vlasov-Poisson/Maxwell code), Roytershteyn and Delzanno are cited in 2602.17487 as nonrelativistic spectral work.

## 8. SPECTRAX PR plan (not started; no PR opened)

API: `Hermite_Fourier_system(..., inv_c2=0.0)` and `plasma_current(..., inv_c2=0.0)` with default `inv_c2=0.0` bitwise the present code path (static branch: `inv_c2 == 0` keeps the tridiagonal ladder; no dense $V$). New module `spectrax/_relativistic.py`: `hermite_eig(N)`, `streaming_matrix(u0, a, inv_c2, lam, P)`, `energy_matrix(...)`, `relativistic_work_defect(...)` ($\epsilon$). `Model(..., c=None)` in Dark-SPECTRAX mapping to `inv_c2 = 1/c^2` (velocities are in units of $c$ in SPECTRAX, so `inv_c2=1` is physical and the nonrelativistic limit is the current default). `_lawson_streaming.py`: replace $e^{-i\tau k(u+a\lambda_j)}$ by $e^{-i\tau k\,v(u+a\lambda_j)}$ when `inv_c2>0`. `_diagnostics.py`: kinetic energy $\Gamma$ row; ledger test.
Tests (1D1V first): (1) default path bitwise unchanged; (2) `inv_c2=0` dense route equals ladder to 1e-12 (gate a1); (3) homogeneous oscillator vs `scipy` ODE (gate b, cheap version T=100); (4) $\Gamma$-energy ledger with a pump-frame run; (5) response function vs Juttner $\varepsilon$ (gate c2) with `scipy.special.k1e`; (6) Lawson step equals the exponential of $-ikV$.
Out of scope for the first PR: 3V coupling through $\gamma$ ($O(N^4)$), magnetic term, relativistic Dougherty collisions (needs a relativistic LBO), the closure/hypercollision (unchanged: acts on $n\ge3$), Dark-SPECTRAX integration of $c$ into the Proca coupling.
Risks: (a) cost ratio 1-5x is unconfirmed on an idle machine; (b) the AW top-mode truncation instability is unchanged by relativity (the instability lives in $L_s+L_E$ non-normality; $V$ is symmetric, $L_E$ still nilpotent); (c) hot Juttner tails.
