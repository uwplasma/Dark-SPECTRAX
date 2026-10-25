"""Declared gates for the relativistic Hermite prototype (relherm). Thresholds below are fixed BEFORE any gate is run.

Usage: python studies/relherm/relherm_gates.py {a,b,b2,c,d,e} [path to SPECTRAX checkout for gate a/e]
Units: c = omega_pe = n = m_e = 1, m_i = 1836, u_te^2 = T_e/(m_e c^2) = 1e-3 (relgrid_ref conventions).

GATES (rejection criteria: a or b failing rejects the method)
 a1  inv_c2 = 0: kinetic RHS and current equal spectrax Hermite_Fourier_system / plasma_current; max rel err <= 1e-12.
 a2  inv_c2 = 1e-8: rel diff to the nonrelativistic RHS <= 1e-6 (expected O(inv_c2 u^2)).
 b   pump frame, homogeneous driven e-i oscillator, vq = 0.03, w = omega_L, T = 500, dt = 0.01 RK4, N = 24, vs
     relgrid_ref.homogeneous(rel=True): max|Ebar - ref|/max|ref| <= 1e-6; |W_ext - ref|/ref <= 1e-6;
     |phase - ref phase| <= 1e-6 rad on [0.8T, T]; and the nonrelativistic ODE must differ from the exact one by > 1e-3 in W_ext (discrimination).
 b2  lab frame (no basis shift), vq = 0.01, T = 300, dt = 0.01, N = 48: same three thresholds (1e-6).
 c   linearised Vlasov-Ampere eigenvalue (N = 128, Juttner theta = 1e-3) vs relgrid_ref.langmuir_root(rel=True) at k*lambda_D = 0.2, 0.3:
     |Re dw| <= 2e-4, |Im dw| <= 1.5e-3, and |w - w_rel| < |w - w_nonrel|.
 d   energy ledger  K - K0 + dU - W_ext  with K = sum m a [G C]_0 (gamma - 1), nonlinear pump-frame run, vq = 0.3, T = 200, dt = 0.01,
     N = 24, Nx = 16, density seed 1e-3 at k = 2 pi/40: |ledger| <= 1e-5 * W_ext(T).
 e   cost: ms per RHS (jit, CPU), N = 32, 64, 128, Nx = 32, 2 species; ratio relativistic / spectrax RHS reported (no pass/fail).
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import relherm as R
import relgrid_ref as G

MI = G.MI
VTE = G.VTE
VTI = G.VTI
OM = G.OMEGA
Q = [-1.0, 1.0]
M = [1.0, MI]
A = [np.sqrt(2) * VTE, np.sqrt(2) * VTI]


def out(name, d):
    p = os.path.join(HERE, "gates_%s.json" % name)
    json.dump(d, open(p, "w"), indent=1, default=float)
    print(json.dumps(d, indent=1, default=float))


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def spx_args(spx, N, Nx, Lx, a, u0, q_over_m):
    """Build arguments of spectrax._model.Hermite_Fourier_system for one species, 1D1V."""
    sys.path.insert(0, spx)
    from spectrax import _model, _simulation
    Ny = Nz = 1
    nk = Nx // 2 + 1
    ky, kx, kz = jnp.meshgrid(jnp.zeros(1), 2 * jnp.pi * jnp.arange(nk), jnp.zeros(1), indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    n = jnp.arange(N)[None, None, None, :, None, None, None]
    p = jnp.zeros((1, 1, 1, 1, 1, 1, 1))
    tabs = (jnp.sqrt(n + 1.0), jnp.sqrt(n * 1.0), jnp.sqrt(p + 1.0), jnp.sqrt(p), jnp.sqrt(p + 1.0), jnp.sqrt(p))
    mask = jnp.asarray(_simulation._twothirds_mask(Ny, Nx, Nz))
    return _model, tabs, kx, ky, kz, k2, mask, Ny, Nz


def gate_a(spx):
    N, Nx, Lx = 20, 8, 5.0
    rng = np.random.default_rng(1)
    nk = Nx // 2 + 1
    a, u0, qm = 0.7, 0.3, -1.3
    Ck = (rng.normal(size=(N, nk)) + 1j * rng.normal(size=(N, nk))) * np.exp(-0.1 * np.arange(N))[:, None]
    Ck[:, 0] = Ck[:, 0].real
    Ek = rng.normal(size=nk) + 1j * rng.normal(size=nk)
    Ek[0] = Ek[0].real
    _model, tabs, kx, ky, kz, k2, mask, Ny, Nz = spx_args(spx, N, Nx, Lx, a, u0, qm)
    nkx = nk
    Ckj = jnp.asarray(Ck).reshape(N, 1, nkx, 1)
    C = jnp.fft.irfftn(Ckj * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    Fk = jnp.zeros((6, Ny, nkx, Nz), complex).at[0, 0, :, 0].set(jnp.asarray(Ek))
    F = jnp.fft.irfftn(Fk * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    col = jnp.zeros((1, 1, N))
    alpha = jnp.array([a, 1.0, 1.0])
    us = jnp.array([u0, 0.0, 0.0])
    ref = _model.Hermite_Fourier_system(Ckj, C, F, kx, ky, kz, k2, col, *tabs, Lx, 1.0, 1.0, 0.0, 0.0, alpha, us,
                                        jnp.array([qm]), jnp.array([1.0]), N, 1, 1, 1, mask23=mask)
    ref = np.asarray(ref).reshape(N, nkx)
    Jref = np.asarray(_model.plasma_current(jnp.array([qm]), alpha, us, Ckj, N, 1, 1, 1))[0].reshape(-1)
    res = {}
    for name, ic2 in (("a1", 0.0), ("a2", 1e-8)):
        S = R.Setup(N, Nx, Lx, [qm], [1.0], [a], inv_c2=ic2)
        mine = np.asarray(R.kinetic_rhs(S, jnp.asarray(Ck)[None], jnp.asarray(Ek), jnp.array([u0])))[0]
        Jm = np.asarray(R.current_k(S, jnp.asarray(Ck)[None], jnp.array([u0])))
        res[name] = {"rhs_rel_err": rel(mine, ref), "current_rel_err": rel(Jm * 1.0, Jref * 1.0 / 1.0)}
    # SPECTRAX current is in units q*a*(...): plasma_current multiplies by q*a0*a1*a2, ours q*a (a1=a2=1) -> same with q=1
    res["a1"]["pass"] = bool(max(res["a1"].values()) <= 1e-12)
    res["a2"]["pass"] = bool(res["a2"]["rhs_rel_err"] <= 1e-6)
    out("a", res)


def oscillator(vq, T, dt, N, pump, tag):
    Nx = 4
    S = R.Setup(N, Nx, 40.0, Q, M, A, inv_c2=1.0)
    nk = S.nk
    C0 = np.zeros((2, N, nk), complex)
    for s in range(2):
        C0[s, 0, 0] = 1.0 / A[s]
    E0 = vq * VTE
    drive = lambda t: E0 * jnp.cos(OM * t)
    y0 = (jnp.asarray(C0), jnp.zeros(nk, complex), jnp.zeros(2), jnp.zeros(()))
    ds = 0.5
    nsave = int(round(T / ds))
    S2 = S
    y, outs = R.run(S, y0, drive, pump, T, dt, nsave)
    Eb = np.array([float(o[0]) for o in outs])
    Wx = np.array([float(o[1]) for o in outs])
    t = ds * np.arange(1, nsave + 1)
    sol = G.homogeneous(vq, T, w=OM, rel=True)
    sol_nr = G.homogeneous(vq, T, w=OM, rel=False)
    Er, Wr = sol.sol(t)[2], sol.sol(t)[3]
    t0 = 0.8 * T
    ph, am = G.demod_phase(t, Eb, OM, t0, T)
    phr, amr = G.demod_phase(t, Er, OM, t0, T)
    d = {"vq": vq, "T": T, "dt": dt, "N": N, "pump": pump,
         "E_rel_err": rel(Eb, Er), "W_rel_err": float(abs(Wx[-1] - Wr[-1]) / abs(Wr[-1])),
         "phase_err_rad": float(abs(ph - phr)), "amp_rel_err": float(abs(am - amr) / amr),
         "W_exact": float(Wr[-1]), "W_nonrel_ode": float(sol_nr.sol(T)[3]),
         "nonrel_vs_exact_W": float(abs(sol_nr.sol(T)[3] - Wr[-1]) / abs(Wr[-1]))}
    d["pass"] = bool(d["E_rel_err"] <= 1e-6 and d["W_rel_err"] <= 1e-6 and d["phase_err_rad"] <= 1e-6)
    d["discriminates"] = bool(d["nonrel_vs_exact_W"] > 1e-3) if tag == "b" else None
    out(tag, d)


def project_species(theta, a, N, rel):
    if rel:
        f = lambda u: G.juttner_pdf(u, theta)
    else:
        f = lambda u: np.exp(-u * u / (2 * theta)) / np.sqrt(2 * np.pi * theta)
    return R.project(f, a, 0.0, N, nq=8192, span=20.0)


def gate_c():
    N = 128
    res = {"N": N}
    for kld in (0.2, 0.3):
        k = kld / VTE
        row = {}
        for rel_ in (True, False):
            ic2 = 1.0 if rel_ else 0.0
            S = R.Setup(N, 4, 2 * np.pi / k, Q, M, A, inv_c2=ic2)
            thetas = [VTE ** 2, VTE ** 2 / MI]
            C0 = [np.asarray(project_species(th, A[s], N, rel_)) for s, th in enumerate(thetas)]
            u0 = jnp.zeros(2)
            Mx = np.zeros((2 * N + 1, 2 * N + 1), complex)
            for s in range(2):
                V = np.asarray(S.V(u0, s))
                sl = slice(s * N, (s + 1) * N)
                Mx[sl, sl] = -1j * k * V
                low = np.concatenate([[0.0], C0[s][:-1]])
                Mx[sl, 2 * N] = (Q[s] / M[s]) * np.asarray(S.sq2n) / A[s] * low
                Mx[2 * N, sl] = -Q[s] * A[s] * V[0]
            lam = np.linalg.eigvals(Mx)
            w = 1j * lam
            wr = G.langmuir_root(k, rel=True)
            wn = G.langmuir_root(k, rel=False)
            target = wr if rel_ else wn
            i = np.argmin(np.abs(w - target))
            row["rel" if rel_ else "nonrel"] = {"omega": [w[i].real, w[i].imag], "root_rel": [wr.real, wr.imag], "root_nonrel": [wn.real, wn.imag],
                                                 "dRe_vs_own_root": float(w[i].real - target.real), "dIm_vs_own_root": float(w[i].imag - target.imag),
                                                 "max_Im_growth": float(np.max(w.imag))}
        r = row["rel"]
        w = complex(*r["omega"])
        row["pass"] = bool(abs(r["dRe_vs_own_root"]) <= 2e-4 and abs(r["dIm_vs_own_root"]) <= 1.5e-3
                           and abs(w - complex(*r["root_rel"])) < abs(w - complex(*r["root_nonrel"])))
        res["kld_%g" % kld] = row
    res["pass"] = bool(all(res[k]["pass"] for k in res if k.startswith("kld")))
    out("c", res)


def gate_d():
    N, Nx, Lx = 24, 16, 40.0
    vq, T, dt = 0.3, 200.0, 0.01
    res = {}
    for name, ic2 in (("rel", 1.0), ("nonrel", 0.0)):
        S = R.Setup(N, Nx, Lx, Q, M, A, inv_c2=ic2)
        nk = S.nk
        C0 = np.zeros((2, N, nk), complex)
        k1 = 2 * np.pi / Lx
        for s in range(2):
            C0[s, 0, 0] = 1.0 / A[s]
        C0[0, 0, 1] = 1e-3 / (2 * A[0])
        rho1 = -1.0 * A[0] * C0[0, 0, 1]
        Ek = np.zeros(nk, complex)
        Ek[1] = -1j * rho1 / k1
        E0 = vq * VTE
        drive = lambda t: E0 * jnp.cos(OM * t)
        y0 = (jnp.asarray(C0), jnp.asarray(Ek), jnp.zeros(2), jnp.zeros(()))
        K0 = float(R.kinetic_energy(S, y0[0], y0[2]))
        U0 = float(R.field_energy(y0[1]))
        y, outs = R.run(S, y0, drive, True, T, dt, 20)
        K = np.array([float(o[2]) for o in outs])
        U = np.array([float(o[3]) for o in outs])
        W = np.array([float(o[1]) for o in outs])
        led = K - K0 + (U - U0) - W
        d = {"ledger_max_rel_to_W": float(np.max(np.abs(led) / np.abs(W))), "ledger_end_rel": float(abs(led[-1]) / abs(W[-1])),
             "W_end": float(W[-1]), "K_end_minus_K0": float(K[-1] - K0), "u0_end": [float(x) for x in np.asarray(y[2])],
             "Ek_end_energy_over_W": float(U[-1] / abs(W[-1]))}
        res[name] = d
    res["pass"] = bool(res["rel"]["ledger_end_rel"] <= 1e-5)
    out("d", res)


def gate_e(spx):
    res = {}
    Nx, Lx = 32, 40.0
    for N in (32, 64, 128):
        nk = Nx // 2 + 1
        S = R.Setup(N, Nx, Lx, Q, M, A, inv_c2=1.0)
        rng = np.random.default_rng(0)
        C = jnp.asarray((rng.normal(size=(2, N, nk)) + 1j * rng.normal(size=(2, N, nk))) * np.exp(-0.05 * np.arange(N))[None, :, None])
        Ek = jnp.asarray(rng.normal(size=nk) + 1j * rng.normal(size=nk))
        u0 = jnp.array([0.3, 0.0])
        f = jax.jit(lambda C, Ek, u0: R.kinetic_rhs(S, C, Ek, u0))
        S0 = R.Setup(N, Nx, Lx, Q, M, A, inv_c2=0.0, dense_nonrel=True)
        f0 = jax.jit(lambda C, Ek, u0: R.kinetic_rhs(S0, C, Ek, u0))
        def tm(fn, *args, rep=50):
            fn(*args).block_until_ready()
            t = time.perf_counter()
            for _ in range(rep):
                r = fn(*args)
            r.block_until_ready()
            return (time.perf_counter() - t) / rep * 1e3
        row = {"rel_ms": tm(f, C, Ek, u0), "dense_nonrel_ms": tm(f0, C, Ek, u0)}
        # spectrax ladder RHS, same Ns, N, Nx (two species)
        _model, tabs, kx, ky, kz, k2, mask, Ny, Nz = spx_args(spx, N, Nx, Lx, 1, 0, 1)
        Ck = jnp.asarray(np.asarray(C).transpose(0, 1, 2).reshape(2 * N, 1, nk, 1))
        Cx = jnp.fft.irfftn(Ck * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
        Fk = jnp.zeros((6, Ny, nk, Nz), complex).at[0, 0, :, 0].set(Ek)
        F = jnp.fft.irfftn(Fk * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
        col = jnp.zeros((1, 1, N))
        al = jnp.array([A[0], 1, 1, A[1], 1, 1]); us = jnp.array([0.3, 0, 0, 0, 0, 0])
        g = lambda Ck, Cx, F: _model.Hermite_Fourier_system(Ck, Cx, F, kx, ky, kz, k2, col, *tabs, Lx, 1.0, 1.0, 0.0, 0.0, al, us,
                                                            jnp.array(Q), jnp.array([1.0, 1.0 / MI]), N, 1, 1, 2, mask23=mask)
        gj = jax.jit(g)
        row["spectrax_ms"] = tm(gj, Ck, Cx, F)
        row["ratio_rel_over_spectrax"] = row["rel_ms"] / row["spectrax_ms"]
        res["N%d" % N] = row
    out("e", res)


if __name__ == "__main__":
    which = sys.argv[1]
    spx = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "..", "rh-spx")
    if which == "a": gate_a(spx)
    elif which == "b": oscillator(0.03, 500.0, 0.01, 24, True, "b")
    elif which == "b2": oscillator(0.01, 300.0, 0.01, 48, False, "b2")
    elif which == "c": gate_c()
    elif which == "d": gate_d()
    elif which == "e": gate_e(spx)
