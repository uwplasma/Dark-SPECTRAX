"""SPECTRAX ordinary-run adapter for electrostatic-unit cases.

Mapping (derived from spectrax/_model.py + _simulation.py at 6781d80):
  dC_n/dt = -i k a [sqrt((n+1)/2) C_{n+1} + sqrt(n/2) C_{n-1}] - i k u C_n + q Omega_c sqrt(2n)/a E C_{n-1}
  dE/dt  = i k x B - J/Omega_cs[0],   dB/dt = -i k x E,   J = sum_s q a^3 (a C_1/sqrt2 + u C_0)
  => f = sum C_n psi_n((v-u)/a), psi_n = e^{-xi^2} H_n/sqrt(pi 2^n n!), n_s = a_x a_y a_z C_000,
     sigma_v (thermal standard deviation) = a/sqrt(2); light speed = 1 (Maxwell curl terms);
     with q=-1, Omega_c = Omega_cs[0] = 1 and sum n_s = 1: omega_pe = 1 (time unit 1/omega_pe),
     velocity unit c, length unit c/omega_pe, E_code = e E /(m c omega_pe).
  Electrostatic case with v_te=1, lambda_D=1:  choose v_t/c = beta (free), a = sqrt(2)*beta*v_t_es,
     u_code = beta*u_es, k_code = k_es/beta, t identical, Lx = 2 pi m / k_code for mode index m.
  Pure longitudinal 1D (kx only, Ey=Ez=B=0 initially) never involves c, so results are beta-independent
  (checked numerically with two beta values).
Gauss-consistent initialization: i k E_k = q dn_k/Omega_cs[0].
"""
import numpy as np, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from spectrax import simulation
from diffrax import Dopri8

def run(species, L_es, Nx, Nn, tmax, nt, seeds, beta=0.1, nu=0.0, tol=1e-10, qforce=True, Ck0=None, Fk0=None):
    """species: list of (density_fraction, vt_es, u_es). seeds: {mode_index: complex density coefficient dn_k
    (rfft forward-normalized, same for each species weighted by fraction)}. Returns dict."""
    Ns = len(species)
    a = np.array([np.sqrt(2)*beta*s[1] for s in species])
    alpha_s = jnp.array(np.repeat(a, 3))
    u_s = jnp.array(np.concatenate([[beta*s[2], 0, 0] for s in species]))
    qs = jnp.array([-1.0 if qforce else 0.0]*Ns)
    Lx = L_es*beta
    if Ck0 is None:
        Ck0 = np.zeros((Ns*Nn, 1, Nx//2+1, 1), complex)
        Fk0 = np.zeros((6, 1, Nx//2+1, 1), complex)
        for i, (fr, vt, u) in enumerate(species):
            Ck0[i*Nn, 0, 0, 0] = fr/a[i]**3
            for m, dn in seeds.items():
                Ck0[i*Nn, 0, m, 0] = fr*dn/a[i]**3
        for m, dn in seeds.items():
            kc = 2*np.pi*m/Lx
            Fk0[0, 0, m, 0] = (-1.0)*dn/(1j*kc) if qforce else 0.0
    p = dict(Lx=Lx, Ly=1.0, Lz=1.0, mi_me=1.0, Ti_Te=1.0, qs=qs, alpha_s=alpha_s, u_s=u_s,
             Omega_cs=jnp.ones(Ns), nu=nu, D=0.0, t_max=float(tmax), ode_tolerance=tol,
             Ck_0=jnp.asarray(Ck0), Fk_0=jnp.asarray(Fk0))
    t0 = time.time()
    out = simulation(p, Nx=Nx, Ny=1, Nz=1, Nn=Nn, Nm=1, Np=1, Ns=Ns, timesteps=nt, dt=0.01, solver=Dopri8())
    jax.block_until_ready(out["Ck"])
    wall = time.time() - t0
    t = np.asarray(out["time"]); Fk = np.asarray(out["Fk"])[:, 0, 0, :, 0]
    Ck = np.asarray(out["Ck"])[:, :, 0, :, 0]
    st = {k: int(np.asarray(v)) for k, v in out["solver_stats"].items() if np.ndim(v) == 0}
    # E in electrostatic units: dv_es/dt = -E_es, v_es = v_code/beta  =>  E_es = E_code/beta
    Ck_last = np.asarray(out["Ck"])[-1]; Fk_last = np.asarray(out["Fk"])[-1]
    return dict(Ck_last=Ck_last, Fk_last=Fk_last, t=t, Ek=Fk/beta, Ck=Ck, wall=wall, stats=st, a=a, Lx=Lx, beta=beta,
                proj_res=float(out["initial_projection_residual"]))
