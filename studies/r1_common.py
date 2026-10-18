"""R1 (finite reservoir, finite temporal coherence): declared gates, shared setup and the linear theory.

Everything in this file is declared BEFORE any R1 run. Gates are never edited after results are seen; if a gate is
wrong, the failure is recorded and a follow-up study with a new gate is declared in a new commit.

Setup (as studies/h05_pump_frame.py): v_te = sqrt(1e-3) c, m_i/m_e = 1836, T_i = T_e, L = 40, Nx = 8, mobile ions,
declared seeds (5e-4, 5e-5j electron, 5e-4 ion), pump frame + run_adaptive (segment 20), rtol 1e-10, atol 1e-14,
omega = omega_L = sqrt(1 + 1/1836). `v_q/v_te` is the EFFECTIVE initial force amplitude: E0 = (v_q/v_te) v_te.

PART (a) finite reservoir (studies/r1_reservoir.py)
  Dark field: free k = 0 Proca mode with Omega_D = omega_L and eta E_D(0) = E0, so the prescribed drive of the
  H05 run at the same v_q is the eta -> 0 limit. Reservoir energy U_D0/(n T_e V) = 0.5 (v_q/v_te)^2 / eta^2 = u, with
  u in (0.1, 1, 10) hence eta = (v_q/v_te)/sqrt(2 u). v_q/v_te in (0.01, 0.03).
  Linear coupled-oscillator prediction (k = 0, mean field, exact for the mobile-ion mean-field system):
      x = (Ebar, J, E_D, A_D):  Ebar' = -J, J' = wL^2 (Ebar + eta E_D), E_D' = Omega_D^2 A_D - eta J, A_D' = -E_D.
  Comparison observables: U_D(t)/U_D0, W_D(t)/U_D0, Ebar(t).
PART (b) finite temporal coherence (studies/r1_coherence.py)
  Drive E0 cos(omega_L t + Phi(t)), E0 = 0.01 v_te, v_q/v_te = 0.01 only (certified to t = 1000).
  Phi'(t) = frequency noise delta(t): piecewise-linear interpolation (knots every h = 5) of iid N(0, s^2) samples,
  s^2 = 1/(tau_c h). Phi is continuous and C^1 (the drive is C^1: no phase resets, no energy injected outside the
  external-work ledger W_ext, which is integrated as a state and checked against the kinetic-energy change).
  Statistical definition: Phi is a zero-mean Gaussian process, Phi(0) = 0, phase diffusion Var[Phi(t)-Phi(s)] ->
  2|t-s|/tau_c for |t-s| >> h, hence <exp(i dPhi)> = exp(-|tau|/tau_c): LORENTZIAN line with half width 1/tau_c in
  angular frequency (FWHM 2/tau_c). The Gaussian-line case is NOT run. tau_c omega_pe in (30, 100, 300, inf);
  inf is the deterministic drive (one realization). 6 realizations (seeds 0..5) per finite tau_c for the kinetic runs;
  400 realizations of the linear mean-field ODE for the statistical check.
  Linear expectation (derived): mean-field response J(t) = wL^2 int_0^t cos(wL (t-s)) E_d(s) ds, W_ext = int J E_d dt,
      E[W_ext](T) = wL^2 int_0^T dt int_0^t ds cos(wL(t-s)) E[E_d(t) E_d(s)],
      E[E_d(t)E_d(s)] = (E0^2/2) [cos(w0 (t-s)) exp(-V_-/2) + cos(w0 (t+s)) exp(-V_+/2)],
  V_-/+ = Var[Phi(t) -/+ Phi(s)] evaluated exactly from the interpolation covariance (grid dt = 0.1).
  At resonance and t >> tau_c: dE[W]/dt -> E0^2 tau_c wL^2 / 4 (deterministic limit: W = E0^2 t^2 / 8).

GATES (declared; each reported as pass/fail, failures kept)
  G1 ledger: max |ledger defect| / max(sum |W components|) <= 1e-4 for every run (all three rows).
  G2 positivity: min K_species > 0 and min k=0 T_x,species > 0 over the saved times of every run; time of first
     violation caps the claim window.
  G3 refinement pairs, first t > 50 where |a-b| > 0.1 * running max of max(|a|,|b|) on dK_e and on the work observable
     (W_D in (a), W_ext in (b)), also on U_D in (a): Nn 32 vs 64 (nu = 0) for every case in (a) and for realization 0
     of every tau_c in (b); Nx 8 vs 16 (Nn 32) for u = 1 at both v_q in (a), and for tau_c in (100, inf) realization 0 in (b).
     Claim window of a case = min(resolved_until of its pairs, certified window (1000 at v_q 0.01, 940 at v_q 0.03), T).
  G4 mean-field identity (Nn 32 runs): residual of Ebar'' = -sum_s q_s^2 (Om_s/Om_0)[n_s0 (Ebar + eta Ebar_D + E_drive)
     + <dn_s dF_x>] evaluated from the code's own RHS (forward-mode derivative of the moments) at every 5th save,
     max residual / (wL^2 max |mean force|) <= 1e-6.
  G5 (a) linear coupled-oscillator prediction: max_{t<=claim window} |U_D^kin - U_D^lin| / U_D0 <= 0.1. Failure is
     informative: first exceedance time t_dev is recorded as the onset of departure from the linear reservoir.
  G6 (a) the 4x4 solution itself equals an independent solve_ivp integration to 1e-8 (code check of the prediction).
  G7 (a) matched prescribed drive: W_D(t) and W_ext(prescribed) agree within 10% (relative to W_ext, t > 50) until t_split;
     t_split must match the linear-theory t_split within a factor 1.5 (reported pass/fail).
  G8 (b) quadrature check: tau_c = inf expectation reproduces the exact fixed-ion W_ext to 2e-3.
  G9 (b) Monte-Carlo check: the 400-realization linear mean-field mean differs from the derived expectation by at most
     4 standard errors at all of 10 equally spaced times in [T/10, T], for each finite tau_c.
  G10 (b) kinetic vs linear (same drive, per realization): W_kin,r / W_lin,r - 1 within 10% (t > 50) up to the claim
     window means "no nonlinear detuning"; the ensemble-mean ratio R(t) = <W_kin>/<W_lin> and its late value
     (t in [0.8 T, T]) are reported with the standard error over realizations; claims on the late-time rate use only the
     claim window.
  Scope limit stated up front: no claim outside the certified windows; Gaussian line, v_q 0.03 and 0.1 in (b), Nx > 16
  and any t > 1000 are not attempted here.
"""

import time

import numpy as np

import darkspectrax as ds
from dataclasses import dataclass

vte = np.sqrt(1e-3)
a_e, a_i = np.sqrt(2) * vte, np.sqrt(2) * np.sqrt(1e-3 / 1836)
Lx, w_L = 40.0, np.sqrt(1 + 1 / 1836)
seeds = [(0, (1, 0, 0), 5e-4), (0, (2, 0, 0), 5e-5j), (1, (1, 0, 0), 5e-4)]
SEG, NSAVE, T_END = 20.0, 11, 1000.0
CERTIFIED = {0.01: 1000.0, 0.03: 940.0}
H_KNOT = 5.0
LEDGER_TOL, ID_TOL, TOL10 = 1e-4, 1e-6, 0.1


@dataclass(frozen=True, eq=False)
class PhasedModel(ds.Model):
    """Prescribed drive E0 cos(w_L t + Phi(t)) with Phi from the frequency-noise knots `z` (spacing `h`)."""
    z: tuple = ()
    h: float = H_KNOT

    def drive(self, t):
        import jax.numpy as jnp
        z = jnp.asarray(self.z, float)
        Phi_knots = jnp.concatenate([jnp.zeros(1), jnp.cumsum(self.h * 0.5 * (z[:-1] + z[1:]))])
        j = jnp.clip(jnp.floor(t / self.h).astype(int), 0, z.size - 2)
        s = t - j * self.h
        Phi = Phi_knots[j] + z[j] * s + (z[j + 1] - z[j]) * s ** 2 / (2 * self.h)
        return jnp.asarray(self.E_drive) * jnp.cos(self.omega_drive * t + Phi)


def noise_knots(tau_c, seed, T=T_END, h=H_KNOT):
    """Frequency-noise samples for realization `seed` (all zeros for tau_c = inf)."""
    n = int(np.ceil(T / h)) + 3
    if not np.isfinite(tau_c):
        return np.zeros(n)
    return np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(tau_c * h), n)


def phase_np(t, z, h=H_KNOT):
    """Numpy copy of Phi(t) (same formula as PhasedModel.drive)."""
    t = np.asarray(t, float)
    Phik = np.concatenate([[0.0], np.cumsum(h * 0.5 * (z[:-1] + z[1:]))])
    j = np.clip(np.floor(t / h).astype(int), 0, z.size - 2)
    s = t - j * h
    return Phik[j] + z[j] * s + (z[j + 1] - z[j]) * s ** 2 / (2 * h)


def mobile_model(Nx=8, Nn=32, nu=0.0, mode="prescribed_drive", E0=0.0, eta=0.0, Omega_D=w_L, cls=ds.Model, **kw):
    return cls(Nx=Nx, Nn=Nn, nu=nu, Lx=Lx, qs=(-1.0, 1.0), Omega_cs=(1.0, 1 / 1836), alpha_s=(a_e,) * 3 + (a_i,) * 3,
               u_s=(0.0,) * 6, mode=mode, E_drive=(E0, 0.0, 0.0) if mode == "prescribed_drive" else (0.0, 0.0, 0.0),
               omega_drive=w_L, eta=eta, Omega_D=Omega_D, frame="pump", **kw)


def initial(model, E_D0=None):
    y = ds.consistent_fields(model, ds.maxwellian(model, [1.0, 1.0], seeds))
    if E_D0 is not None:  # free k = 0 Proca mode with E_D(0) = E_D0, A_D(0) = 0
        y = ds.proca_mode(model, y, (0, 0, 0), [-1j * E_D0 / model.Omega_D, 0.0, 0.0])
    return y


def thermal(model, out):
    """Per save: kinetic energy per species and k = 0 x temperature n T = M_xx - M_x^2/n."""
    K, T = [], []
    for i in range(out["t"].size):
        st = {"Ck": out["Ck"][i], "Fk": out["Fk"][i], "Dk": out["Dk"][i], "W": out["W"][i].astype(complex),
              "B": out["B"][i].astype(complex)}
        K.append(np.asarray(ds.energies(model, st)["K_species"]))
        n, M, M2 = (np.asarray(x)[..., 0, 0, 0].real for x in ds.moments(model, st["Ck"], st["B"]))
        T.append(M2[:, 0, 0] - M[:, 0] ** 2 / n)
    return np.array(K), np.array(T)


def go(model, y0, T):
    tic = time.perf_counter()
    out = ds.run_adaptive(model, y0, T, SEG, n_save_segment=NSAVE, rtol=1e-10, atol=1e-14, max_steps=200_000)
    out["wall"] = time.perf_counter() - tic
    good = np.isfinite(out["t"]) & np.all(np.isfinite(out["W"]), axis=1)
    for k in ("t", "K", "W", "B", "Ck", "Fk", "Dk", "U_gamma", "U_D"):
        out[k] = out[k][good]
    out["ledger_defect"] = out["ledger_defect"][:, good]
    out["K_species"], out["Tx"] = thermal(model, out)
    return out


def identity_residual(model, out, every=5):
    """Mean-field identity residual (gate G4) from the code's own RHS; returns (t, residual, scale)."""
    import jax
    import jax.numpy as jnp
    om = np.asarray(model.Omega_cs) / model.Omega_cs[0]
    q = jnp.asarray(model.qs, float)

    def flux(Ck, B):
        _, M, _ = ds.moments(model, Ck, B)
        return jnp.real(jnp.tensordot(q, M[:, 0, 0, 0, 0], axes=1))

    @jax.jit
    def resid(t, Ck, Fk, Dk, W, B):
        y = {"Ck": Ck, "Fk": Fk, "Dk": Dk, "W": W, "B": B}
        dy = ds.rhs(t, y, model)
        _, dJ = jax.jvp(flux, (Ck, jnp.real(B)), (dy["Ck"], jnp.real(dy["B"])))
        n, _, _ = ds.moments(model, Ck, B)
        Fx = Fk[0] + model.eta * Dk[0] if model.mode == "self_consistent" else Fk[0]
        Fbar = jnp.real(Fx[0, 0, 0])
        Ed = model.drive(t)[0] if model.mode == "prescribed_drive" else 0.0
        tot = 0.0
        for s in range(model.Ns):
            c = ds.inner(model.Nx, n[s][None], Fx[None]) - jnp.real(n[s, 0, 0, 0]) * Fbar
            tot = tot + model.qs[s] ** 2 * om[s] * (jnp.real(n[s, 0, 0, 0]) * (Fbar + Ed) + c)
        return -dJ / model.Omega_cs[0] + tot, jnp.abs(Fbar + Ed)

    idx = np.arange(0, out["t"].size, every)
    R, S = [], []
    for i in idx:
        r, s = resid(float(out["t"][i]), jnp.asarray(out["Ck"][i]), jnp.asarray(out["Fk"][i]),
                     jnp.asarray(out["Dk"][i]), jnp.asarray(out["W"][i], complex), jnp.asarray(out["B"][i], complex))
        R.append(float(r)), S.append(float(s))
    return out["t"][idx], np.array(R), np.array(S)


def disagreement(t, a, b, tol=TOL10):
    """First t > 50 with |a - b| > tol * running max of max(|a|, |b|); None if never."""
    scale = np.maximum(np.maximum.accumulate(np.maximum(np.abs(a), np.abs(b))), 1e-12)
    bad = (np.abs(a - b) > tol * scale) & (t > 50)
    return float(t[np.argmax(bad)]) if bad.any() else None


def ledger_ok(out):
    scale = max(np.abs(out["W"]).sum(axis=1).max(), 1e-300)
    return float(np.abs(out["ledger_defect"]).max() / scale)


def first_nonpositive(out):
    bad = (out["K_species"] <= 0).any(axis=1) | (out["Tx"] <= 0).any(axis=1)
    return float(out["t"][np.argmax(bad)]) if bad.any() else None


# ---------------------------------------------------------------- part (a) linear theory
def coupled_matrix(eta, Omega_D=w_L, wL2=w_L ** 2):
    """x = (Ebar, J, E_D, A_D)."""
    return np.array([[0, -1, 0, 0], [wL2, 0, wL2 * eta, 0], [0, -eta, 0, Omega_D ** 2], [0, 0, -1, 0]], float)


def coupled_solution(t, eta, E_D0, Omega_D=w_L):
    from scipy.linalg import expm
    M = coupled_matrix(eta, Omega_D)
    x0 = np.array([0.0, 0.0, E_D0, 0.0])
    X = np.array([expm(M * tt) @ x0 for tt in t])
    U_D = 0.5 * (X[:, 2] ** 2 + Omega_D ** 2 * X[:, 3] ** 2)
    return X, U_D


def coupled_solution_ivp(t, eta, E_D0, Omega_D=w_L):
    from scipy.integrate import solve_ivp
    M = coupled_matrix(eta, Omega_D)
    s = solve_ivp(lambda tt, x: M @ x, (0, t[-1]), [0, 0, E_D0, 0], t_eval=t, rtol=1e-12, atol=1e-18, method="DOP853")
    return s.y.T


def prescribed_linear(t, E0, w0=w_L):
    """Linear mean-field prescribed-drive W_ext(t) = int J E_d, J' = wL^2 (Ebar + E_d), Ebar' = -J (ODE)."""
    from scipy.integrate import solve_ivp
    wL2 = w_L ** 2
    s = solve_ivp(lambda tt, x: [-x[1], wL2 * (x[0] + E0 * np.cos(w0 * tt)), x[1] * E0 * np.cos(w0 * tt)],
                  (0, t[-1]), [0, 0, 0], t_eval=t, rtol=1e-12, atol=1e-18, method="DOP853")
    return s.y[2]


# ---------------------------------------------------------------- part (b) linear theory
def linear_work_realization(t, z, E0, w0=w_L):
    """W_ext(t) of the linear mean-field ODE driven by E0 cos(w0 t + Phi(t)), integrated with solve_ivp."""
    from scipy.integrate import solve_ivp
    wL2 = w_L ** 2

    def f(tt, x):
        Ed = E0 * np.cos(w0 * tt + phase_np(tt, z))
        return [-x[1], wL2 * (x[0] + Ed), x[1] * Ed]
    s = solve_ivp(f, (0, t[-1]), [0, 0, 0], t_eval=t, rtol=1e-10, atol=1e-16, method="DOP853", max_step=1.0)
    return s.y[2]


def expected_work(tau_c, E0, T=T_END, dt=0.1, w0=w_L, out_every=100):
    """Derived linear expectation E[W_ext](t) for the interpolated frequency-noise drive (see module docstring).
    Returns (t_out, E[W](t_out), grid P(t)=dE[W]/dt on the full grid)."""
    t = np.arange(0.0, T + dt / 2, dt)
    N = t.size
    if np.isfinite(tau_c):
        h = H_KNOT
        nk = int(np.ceil(T / h)) + 3
        # G[i, j] = d Phi(t_i) / d z_j = int_0^{t_i} of the hat function of knot j (Lambda = max(0, 1 - |s - j h|/h))
        tk = np.arange(nk) * h

        def A(u):  # antiderivative of the hat function centred at 0
            u = np.clip(u, -h, h)
            return np.where(u < 0, (u + h) ** 2 / (2 * h), h / 2 + u - u ** 2 / (2 * h))
        G = A(t[:, None] - tk[None, :]) - A(-tk)[None, :]
        s2 = 1.0 / (tau_c * h)
        var = s2 * (G ** 2).sum(axis=1)                 # Var Phi(t)
    P = np.zeros(N)
    blk = 400
    for a in range(0, N, blk):
        ii = np.arange(a, min(a + blk, N))
        tt, ss = t[ii][:, None], t[None, :]
        if np.isfinite(tau_c):
            cov = s2 * (G[ii] @ G.T)
            Vm = var[ii][:, None] + var[None, :] - 2 * cov
            Vp = var[ii][:, None] + var[None, :] + 2 * cov
            C = 0.5 * E0 ** 2 * (np.cos(w0 * (tt - ss)) * np.exp(-Vm / 2) + np.cos(w0 * (tt + ss)) * np.exp(-Vp / 2))
        else:
            C = 0.5 * E0 ** 2 * (np.cos(w0 * (tt - ss)) + np.cos(w0 * (tt + ss)))
        K = np.cos(w_L * (tt - ss)) * C
        mask = (ss <= tt)
        wts = np.where(ss == tt, 0.5, 1.0) * mask
        wts[:, 0] *= 0.5
        P[ii] = w_L ** 2 * dt * (K * wts).sum(axis=1)
    W = np.concatenate([[0.0], np.cumsum(0.5 * dt * (P[1:] + P[:-1]))])
    return t[::out_every], W[::out_every], P
